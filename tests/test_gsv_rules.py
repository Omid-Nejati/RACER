"""CPU regression tests for RACER's proposed strict GSV patch.

Place this file in RACER/tests or run from the RACER repo root with:
  PYTHONPATH=. python -m unittest -v /path/to/test_gsv_rules.py
"""
import unittest
from unittest.mock import patch
import torch
import saliency_qwen as g


class DummyTokenizer:
    eos_token_id = 99
    def decode(self, ids, skip_special_tokens=False):
        return str(ids[0])


class GSVRulesTests(unittest.TestCase):
    def cfg(self, **kwargs):
        base = dict(visual_lambda=0.3, strict_grounding=True, tau_visual=0.25,
                    tau_text=0.25, exclude_neutral_history=True)
        base.update(kwargs)
        return g.SaliencyConfig(**base)

    def test_zero_visual_fails_despite_weighted_score(self):
        state = g.SaliencyState(txt_hist=[1.0], vis_hist=[1.0])
        parts = dict(txt=1.0, vis=0.0, neutral=False)
        score, _ = g._combined_score(parts, state, self.cfg())
        self.assertAlmostEqual(score, 0.7)
        self.assertGreater(score, 0.6)  # the reviewer's counterexample
        self.assertFalse(g._support_pass(parts, state, self.cfg()))

    def test_missing_visual_cannot_be_verified(self):
        state = g.SaliencyState(txt_hist=[1.0], vis_hist=[1.0])
        parts = dict(txt=1.0, vis=None, neutral=False)
        self.assertFalse(g._support_pass(parts, state, self.cfg()))
        self.assertEqual(g._combined_score(parts, state, self.cfg())[0], float('-inf'))

    def test_text_gate_is_independent(self):
        state = g.SaliencyState(txt_hist=[1.0], vis_hist=[1.0])
        parts = dict(txt=0.0, vis=10.0, neutral=False)
        self.assertFalse(g._support_pass(parts, state, self.cfg()))

    def test_visual_only_first_steps(self):
        state = g.SaliencyState()
        self.assertFalse(g._support_pass(dict(txt=1.0, vis=0.0, neutral=True), state, self.cfg()))
        self.assertTrue(g._support_pass(dict(txt=1.0, vis=0.01, neutral=True), state, self.cfg()))
        self.assertEqual(g._combined_score(dict(txt=1.0, vis=0.01, neutral=True), state, self.cfg())[0], 1.0)

    def run_selection(self, parts_map, config):
        seq = torch.tensor([[10, 11]], dtype=torch.long)
        mask = torch.ones_like(seq)
        logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        state = g.SaliencyState(txt_hist=[1.0], vis_hist=[1.0])
        def fake_calc(*args, **kwargs):
            return dict(parts_map[int(args[4])])
        with patch.object(g, 'get_language_layers', return_value=[None] * 9), \
             patch.object(g, 'compute_candidate_saliency_parts', side_effect=fake_calc):
            chosen, meta = g.select_sgrs_token(
                model=object(), tokenizer=DummyTokenizer(), base_inputs={},
                seq=seq, attention_mask=mask, logits=logits, prompt_len=2,
                state=state, config=config, visual_positions=torch.tensor([0]),
                generator=torch.Generator().manual_seed(1))
        return chosen, meta, state

    def test_all_zero_visual_abstains_without_histories(self):
        chosen, meta, state = self.run_selection({
            0: dict(txt=1.0, vis=0.0, neutral=False),
            1: dict(txt=1.0, vis=0.0, neutral=False),
        }, self.cfg(max_resample=0))
        self.assertEqual(chosen, 99)
        self.assertTrue(meta['abstained'])
        self.assertFalse(meta['gsv_verified'])
        self.assertFalse(meta['support_pass'])
        self.assertEqual(state.txt_hist, [1.0])
        self.assertEqual(state.vis_hist, [1.0])

    def test_fallback_does_not_select_unsupported_token(self):
        chosen, meta, state = self.run_selection({
            0: dict(txt=1.0, vis=0.0, neutral=False),
            1: dict(txt=0.30, vis=0.30, neutral=False),
        }, self.cfg(max_resample=0))
        self.assertEqual(chosen, 1)
        self.assertTrue(meta['fallback'])
        self.assertTrue(meta['support_pass'])
        self.assertFalse(meta['gsv_verified'])  # support floor passes, combined score does not

    def test_legacy_port_unchanged(self):
        state = g.SaliencyState(txt_hist=[1.0], vis_hist=[1.0])
        cfg = g.SaliencyConfig(visual_lambda=0.3, strict_grounding=False)
        score, _ = g._combined_score(dict(txt=1.0, vis=0.0, neutral=False), state, cfg)
        self.assertAlmostEqual(score, 0.7)
        self.assertTrue(g._support_pass(dict(txt=1.0, vis=0.0, neutral=False), state, cfg))


if __name__ == '__main__':
    unittest.main()
