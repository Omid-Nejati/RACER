"""Unit tests for the MedicalHall novelties (CPU, tiny random Qwen2.5-VL).

Run:  cd tests && python -m pytest -q test_novelties.py
Also: bash tests/check_port_equivalence.sh  (old pipeline == --preset port)
"""
import math
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(1, str(HERE.parent))

from tiny_qwen import make_inputs, make_tiny  # noqa: E402
from mcot_plus import AVEG, AVEGConfig, parse_info_layers  # noqa: E402
from ott_qwen import OTTConfig, QwenOTTController, parse_layer_range  # noqa: E402
from risk_router import RiskRouter, RouterConfig  # noqa: E402
from saliency_qwen import (  # noqa: E402
    LocoREConfig, LocoREController, SaliencyConfig, SaliencyState,
    _combined_score, _threshold, adaptive_locore_beta,
)


@pytest.fixture(scope="module")
def tiny():
    return make_tiny(seed=0), make_inputs(grid=(1, 8, 8))


def test_layer_ranges():
    assert parse_layer_range("auto", 36, "crc") == list(range(18))
    assert parse_layer_range("auto", 36, "svc") == [18]
    assert parse_layer_range("0-2,5,-1", 10, "x") == [0, 1, 2, 5, 9]
    assert parse_info_layers("-1,-6", 37) == [36, 31]


def test_ott_decode_only_keeps_prompt_untouched(tiny):
    """N2.b: image/prompt representations are identical with and without OTT."""
    model, inp = tiny
    ott = QwenOTTController(model, OTTConfig(decode_only=True, max_visual_tokens=8,
                                            visual_select="attention", svc_ref_space="mlp",
                                            relevance_gate=True))
    ott.install()
    try:
        ott.prepare_reference(inp)
        with torch.no_grad():
            base = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
            ott.enable()
            steered = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
        P = inp["input_ids"].shape[-1]
        for b, s in zip(base, steered):
            assert torch.equal(b[0, : P - 1], s[0, : P - 1])       # prompt rows untouched
        assert not torch.equal(base[-1][0, P - 1], steered[-1][0, P - 1])  # decode row steered
        assert ott.visual_priority_ranked.numel() == 8
    finally:
        ott.disable(); ott.remove()


def test_ott_risk_gate_zero_is_noop_for_svc(tiny):
    """N2.f: with gate 0 and no CRC layers, SVC-only OTT leaves outputs unchanged."""
    model, inp = tiny
    ott = QwenOTTController(model, OTTConfig(use_crc=False, svc_layers="2", decode_only=True,
                                            default_svc_gate=0.0, max_visual_tokens=8))
    ott.install()
    try:
        ott.prepare_reference(inp)
        with torch.no_grad():
            a = model(**inp, use_cache=False).logits
            ott.enable()
            b = model(**inp, use_cache=False).logits
            ott.set_position_gate(inp["input_ids"].shape[-1] - 1, 1.0)
            c = model(**inp, use_cache=False).logits
        assert torch.allclose(a, b)
        assert not torch.allclose(a[0, -1], c[0, -1])
    finally:
        ott.disable(); ott.remove()


def _loop_bias(q_len, kv_len, P, W, row_beta, base_beta, anchors, vbeta):
    """Reference (slow) implementation of LocoRE+ bias."""
    bias = torch.zeros(q_len, kv_len)
    for ql in range(q_len):
        q = kv_len - q_len + ql
        if q < P:
            continue
        b = row_beta.get(q, base_beta)
        for k in range(max(P, q - W), q):
            bias[ql, k] += math.log1p(b)
        for k in anchors:
            bias[ql, k] += math.log1p(vbeta)
    return bias


def test_locore_vectorised_matches_loop(tiny):
    """N4.a/N4.b: vectorised per-row + visual-anchor bias equals the reference loop."""
    model, _ = tiny
    loc = LocoREController(model, LocoREConfig(beta=0.2, window=3, visual_beta=0.1, visual_anchor_k=3))
    loc.set_prompt_len(10)
    loc.set_row_beta(12, 0.5)
    loc.set_row_beta(14, 0.3)
    loc.set_visual_anchors(torch.tensor([4, 2, 7, 9]))
    value, _ = loc._build(16, 16, torch.device("cpu"), torch.float32)
    ref = _loop_bias(16, 16, 10, 3, {12: 0.5, 14: 0.3}, 0.2, [4, 2, 7], 0.1)
    assert torch.allclose(value, ref, atol=1e-6)


def test_adaptive_beta_rises_when_saliency_drops():
    cfg = LocoREConfig(beta=0.2, beta_max=0.6)
    st = SaliencyState(txt_hist=[0.10, 0.10, 0.10], last_txt=0.10)
    assert adaptive_locore_beta(cfg, st, 5) == pytest.approx(0.2)
    st.last_txt = 0.05
    assert adaptive_locore_beta(cfg, st, 5) == pytest.approx(0.4)
    st.last_txt = 0.001
    assert adaptive_locore_beta(cfg, st, 5) == pytest.approx(0.6)


def test_sgrs_relative_equals_absolute_when_lambda_zero():
    cfg = SaliencyConfig(alpha=0.6, visual_lambda=0.0)
    st = SaliencyState(txt_hist=[0.2, 0.4])
    for txt in (0.05, 0.17, 0.19, 0.5):
        s, _ = _combined_score({"txt": txt}, st, cfg)
        assert (s >= _threshold(0, cfg, st, None)) == (txt >= 0.6 * (0.2 + 0.4) / 2)


def test_entropy_coupled_threshold_is_stricter_for_diffuse_tokens():
    cfg = SaliencyConfig(alpha=0.6, entropy_kappa=0.5, entropy_gamma=0.5, visual_lambda=0.3)
    ent = torch.tensor([0.2, 0.9])
    st = SaliencyState(txt_hist=[1.0], vis_hist=[1.0])
    assert _threshold(1, cfg, st, ent) > _threshold(0, cfg, st, ent)


def test_aveg_soft_gate_and_phase(tiny):
    model, inp = tiny
    aveg = AVEG(model, AVEGConfig(info_layers="-1,-3", candidate_k=3, soft_gate=True,
                                  think_alpha_scale=0.5, answer_alpha_scale=1.0))
    logits = aveg.prepare(inp)
    _, m_think = aveg.apply(logits, "think")
    _, m_ans = aveg.apply(logits, "answer")
    assert 0.0 <= m_think["gate_value"] <= 1.0
    assert m_ans["penalty_strength"] == pytest.approx(2 * m_think["penalty_strength"])


def test_router_modes():
    r = RiskRouter(RouterConfig(mode="risk", risk_threshold=0.5))
    low = r.risk(gate_value=0.0, last_relative=2.0, p_max=0.99)
    high = r.risk(gate_value=1.0, last_relative=0.1, p_max=0.3)
    assert low < 0.5 <= high
    assert r.decide(low, 0.99)["sgrs"] is False
    assert r.decide(high, 0.3) == {"risky": True, "retrace": True, "sgrs": True}
    assert RiskRouter(RouterConfig(mode="always")).decide(0.0, 1.0)["sgrs"] is True
