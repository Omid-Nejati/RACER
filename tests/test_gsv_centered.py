"""CPU regression tests for RACER directional GSV, run after applying patch.

    PYTHONPATH=. python -m unittest discover -s tests -p 'test_gsv_centered.py' -v

These do not replace an end-to-end Qwen2.5-VL evaluation.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import saliency_qwen as sal


class ToyAttentionModel(torch.nn.Module):
    """Synthetic differentiable attention, not a pretrained vision model."""
    def __init__(self, candidate_direction):
        super().__init__()
        self.candidate_direction = float(candidate_direction)
        self.z = torch.nn.Parameter(torch.zeros(1, 1, 5, 5, dtype=torch.float64))

    def forward(self, input_ids, attention_mask, **kwargs):
        t = input_ids.shape[-1]
        z = self.z[..., :t, :t]
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=z.device))
        attn = torch.softmax(z.masked_fill(~causal, -1e9), dim=-1)
        # Class 1 relies positively or negatively on image key 0 at predictor row.
        class_1 = self.candidate_direction * attn[0, 0, t-2, 0]
        other = class_1 * 0.0
        pred = torch.stack([other, class_1])
        logits = pred.view(1, 1, 2).expand(1, t, 2)
        return SimpleNamespace(attentions=(attn,), logits=logits)


class GSVCenteredTests(unittest.TestCase):
    def test_softmax_centering_matches_autograd(self):
        z = torch.tensor([0.2, 0.4, -1.3], dtype=torch.float64, requires_grad=True)
        a = torch.softmax(z, dim=0)
        loss = (a * torch.tensor([1.7, 0.4, -0.2], dtype=torch.float64)).sum()
        g_a, g_z = torch.autograd.grad(loss, (a, z))
        centered = a * (g_a - (a * g_a).sum())
        torch.testing.assert_close(centered, g_z, rtol=1e-10, atol=1e-10)

    def test_real_candidate_saliency_function_supportive(self):
        model = ToyAttentionModel(+3.0)
        seq = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        mask = torch.ones_like(seq)
        x = sal.compute_candidate_saliency_parts(
            model, {'input_ids': seq, 'attention_mask': mask}, seq, mask,
            candidate_id=1, prompt_len=2, target_layers=[0],
            visual_positions=torch.tensor([0]), measure_first_visual=True,
            signed_saliency=True,
        )
        self.assertGreater(x['vis'], x['vis_opp'])
        self.assertGreater(x['vis'], 0.0)

    def test_real_candidate_saliency_function_opposing(self):
        model = ToyAttentionModel(-3.0)
        seq = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        mask = torch.ones_like(seq)
        x = sal.compute_candidate_saliency_parts(
            model, {'input_ids': seq, 'attention_mask': mask}, seq, mask,
            candidate_id=1, prompt_len=2, target_layers=[0],
            visual_positions=torch.tensor([0]), measure_first_visual=True,
            signed_saliency=True,
        )
        self.assertGreater(x['vis_opp'], x['vis'])

    def test_polarity_gate(self):
        yes = sal._directionally_supportive
        self.assertTrue(yes(0.9, 0.1))
        self.assertFalse(yes(0.1, 0.9))
        self.assertFalse(yes(0.0, 0.0))
        self.assertFalse(yes(None, 0.1))
        self.assertFalse(yes(float('nan'), 0.2))
        self.assertFalse(yes(0.5, 0.5))

    def test_strict_gate_checks_both_visual_and_textual_channels(self):
        cfg = sal.SaliencyConfig(strict_grounding=True, visual_lambda=0.3,
                                 signed_saliency=True, direction_min_ratio=0.5)
        state = sal.SaliencyState()
        parts = {'vis':0.6,'vis_opp':0.1,'txt':0.4,'txt_opp':0.1,'neutral':False}
        self.assertTrue(sal._support_pass(parts, state, cfg))
        self.assertFalse(sal._support_pass({**parts, 'vis_opp':2}, state, cfg))
        self.assertFalse(sal._support_pass({**parts, 'txt_opp':2}, state, cfg))
        self.assertFalse(sal._support_pass({**parts, 'vis':None}, state, cfg))

    def test_legacy_port_is_not_subject_to_directional_gate(self):
        cfg = sal.SaliencyConfig(strict_grounding=False, signed_saliency=False)
        self.assertTrue(sal._support_pass({'vis': None, 'txt':0.3}, sal.SaliencyState(), cfg))

    @unittest.skipUnless(hasattr(sal, 'select_sgrs_token'), 'needs the full RACER module')
    def test_fallback_rejects_unopposed_candidate(self):
        cfg = sal.SaliencyConfig(top_k=2, max_resample=0, target_layers='0',
            strict_grounding=True, visual_lambda=0.3, signed_saliency=True)
        state = sal.SaliencyState()
        data = {
            0:{'vis':0.1, 'vis_opp':0.8, 'txt':0.6, 'txt_opp':0.1, 'neutral':False},
            1:{'vis':0.5, 'vis_opp':0.1, 'txt':0.6, 'txt_opp':0.1, 'neutral':False},
        }
        model = object()
        tok = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kwargs: str(ids[0]))
        def measure(*args, **kwargs):
            return data[args[4]]
        with patch.object(sal, 'get_language_layers', return_value=[object()]), \
             patch.object(sal, 'compute_candidate_saliency_parts', side_effect=measure):
            chosen, meta = sal.select_sgrs_token(
                model=model, tokenizer=tok, base_inputs={},
                seq=torch.tensor([[0,1,2]]), attention_mask=torch.ones(1,3),
                logits=torch.tensor([[2.,1.]]), prompt_len=2, state=state,
                config=cfg, visual_positions=torch.tensor([0]))
        self.assertEqual(chosen, 1)
        self.assertTrue(meta['support_pass'])
        self.assertTrue(meta['fallback'])

    @unittest.skipUnless(hasattr(sal, 'select_sgrs_token'), 'needs the full RACER module')
    def test_fallback_abstains_when_all_visual_evidence_opposes(self):
        cfg = sal.SaliencyConfig(top_k=2, max_resample=0, target_layers='0',
            strict_grounding=True, visual_lambda=0.3, signed_saliency=True)
        data = {i:{'vis':0.1, 'vis_opp':0.9, 'txt':0.6, 'txt_opp':0.1, 'neutral':False}
                for i in (0,1)}
        tok = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kwargs: str(ids[0]))
        def measure(*args, **kwargs):
            return data[args[4]]
        with patch.object(sal, 'get_language_layers', return_value=[object()]), \
             patch.object(sal, 'compute_candidate_saliency_parts', side_effect=measure):
            chosen, meta = sal.select_sgrs_token(
                model=object(), tokenizer=tok, base_inputs={},
                seq=torch.tensor([[0,1,2]]), attention_mask=torch.ones(1,3),
                logits=torch.tensor([[2.,1.]]), prompt_len=2,
                state=sal.SaliencyState(), config=cfg,
                visual_positions=torch.tensor([0]))
        self.assertEqual(chosen, 99)
        self.assertTrue(meta['abstained'])
        self.assertFalse(meta['gsv_verified'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
