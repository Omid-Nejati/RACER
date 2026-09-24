"""
AVEG: Adaptive Visual-Entropy Gating (novel extension of MCoT activation decoding).

Original MCoT (ASGO-MM/MCoT-hallucination, `generate_chair.py`):
  * visual entropy H[v] is read from ONE layer (default -1) by projecting the
    visual-token hidden states through lm_head (no final norm for other layers);
  * the gate is a HARD threshold on the entropy of the GREEDY token only;
  * the penalty strength alpha is constant over the whole <think>/<answer> output.

AVEG changes (each switchable, so every change is its own ablation row):
  N1.a  Multi-layer logit-lens entropy ensemble: H is averaged over several
        decoder layers, and intermediate layers are passed through the model's
        final RMSNorm before lm_head (proper logit lens).  Late layers alone
        over-smooth the visual distribution; mid layers still carry
        spatially-local evidence.
  N1.b  Candidate-expected entropy: the gate uses E_{c~p_topK}[H[c]] over the
        Top-K candidates instead of H[argmax].  The later SGRS stage samples from
        Top-K, so a gate that only inspects the greedy token can miss the token
        that is actually emitted.
  N1.c  Soft (sigmoid) gate g = sigmoid((E[H]-gamma)/T): removes the
        discontinuity at gamma and yields a calibrated [0,1] risk signal that the
        risk router (risk_router.py) consumes.
  N1.d  Phase-aware strength: alpha is scaled separately inside <think> and
        <answer>.  Reasoning text contains many function/abstract words with
        naturally high visual entropy; penalising them as hard as object words in
        the answer degrades reasoning.

With the "port" settings (info_layers="-1", candidate_k=1, soft_gate=False,
phase scales 1.0) AVEG is numerically identical to the original MCoT gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AVEGConfig:
    alpha: float = 0.75
    threshold: float = 0.5            # gamma in MCoT
    info_layers: str = "-1"           # e.g. "-1,-6,-12"  (N1.a)
    candidate_k: int = 1              # 1 = greedy token only (original)  (N1.b)
    soft_gate: bool = False           # (N1.c)
    gate_temperature: float = 0.05
    think_alpha_scale: float = 1.0    # (N1.d)
    answer_alpha_scale: float = 1.0
    disable_gate: bool = False        # MCoT --disable_activation_gate


def _find_final_norm(model):
    for getter in (
        lambda m: m.model.norm,
        lambda m: m.model.language_model.norm,
        lambda m: m.language_model.model.norm,
        lambda m: m.language_model.norm,
    ):
        try:
            norm = getter(model)
            if norm is not None:
                return norm
        except Exception:
            pass
    return None


def parse_info_layers(spec: str, n_hidden: int) -> List[int]:
    """Indices into outputs.hidden_states (len = n_layers + 1)."""
    out: List[int] = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0:
            idx = n_hidden + idx
        if idx < 1 or idx >= n_hidden:
            raise ValueError(f"info layer {raw} maps to {idx}, outside [1, {n_hidden - 1}]")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("at least one info layer is required")
    return out


class PhaseTracker:
    """Tracks whether decoding is inside <think> or <answer> (N1.d)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.phase = "think"

    def update(self, generated_ids: Sequence[int]) -> str:
        if self.phase == "answer" or not generated_ids:
            return self.phase
        # Only the recent tail can contain a newly completed tag.
        tail = self.tokenizer.decode(list(generated_ids[-8:]), skip_special_tokens=False)
        if "<answer>" in tail or "</think>" in tail:
            self.phase = "answer"
        return self.phase


class AVEG:
    def __init__(self, model, config: AVEGConfig):
        self.model = model
        self.cfg = config
        self.entropy_vec: Optional[torch.Tensor] = None
        self.baseline_logits: Optional[torch.Tensor] = None
        self.visual_positions: Optional[torch.Tensor] = None

    @torch.no_grad()
    def prepare(self, base_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the per-vocabulary visual entropy vector and the first logits."""
        model = self.model
        outputs = model(**base_inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("model forward did not return hidden states")

        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("model.config.image_token_id is required for visual-only entropy")
        visual_mask = base_inputs["input_ids"][0] == int(image_token_id)
        visual_count = int(visual_mask.sum().item())
        if visual_count <= 1:
            raise RuntimeError("at least two visual positions are required to compute normalized entropy")
        self.visual_positions = torch.nonzero(visual_mask, as_tuple=False).flatten()

        lm_head = model.get_output_embeddings()
        norm = _find_final_norm(model)
        n_hidden = len(hidden_states)
        layers = parse_info_layers(self.cfg.info_layers, n_hidden)

        acc = None
        for idx in layers:
            h = hidden_states[idx][0][visual_mask]
            # HF appends the post-norm state as the last hidden state; every other
            # entry is pre-norm, so apply the final norm for a proper logit lens.
            if idx != n_hidden - 1 and norm is not None:
                h = norm(h)
            logits = h @ lm_head.weight.T
            if getattr(lm_head, "bias", None) is not None:
                logits = logits + lm_head.bias
            probs = torch.softmax(logits.transpose(0, 1).float(), dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1) / math.log(visual_count)
            acc = ent if acc is None else acc + ent
            del logits, probs
        self.entropy_vec = acc / float(len(layers))
        self.baseline_logits = outputs.logits[:, -1, :]
        return self.baseline_logits

    def expected_entropy(self, step_logits: torch.Tensor) -> Tuple[float, int, List[int]]:
        k = max(1, int(self.cfg.candidate_k))
        probs = torch.softmax(step_logits.float(), dim=-1)[0]
        top_p, top_ids = torch.topk(probs, k=min(k, probs.numel()))
        greedy = int(top_ids[0].item())
        if k == 1:
            return float(self.entropy_vec[greedy].item()), greedy, [greedy]
        w = top_p / top_p.sum().clamp_min(1e-12)
        ent = (w * self.entropy_vec[top_ids].float()).sum()
        return float(ent.item()), greedy, [int(x) for x in top_ids.tolist()]

    def gate_value(self, expected_entropy: float) -> float:
        if self.cfg.disable_gate:
            return 1.0
        if self.cfg.soft_gate:
            t = max(float(self.cfg.gate_temperature), 1e-6)
            return float(1.0 / (1.0 + math.exp(-(expected_entropy - self.cfg.threshold) / t)))
        return 1.0 if expected_entropy > self.cfg.threshold else 0.0

    def apply(self, step_logits: torch.Tensor, phase: str = "think") -> Tuple[torch.Tensor, dict]:
        if self.entropy_vec is None:
            raise RuntimeError("AVEG.prepare() must be called before apply()")
        exp_h, greedy, _ = self.expected_entropy(step_logits)
        g = self.gate_value(exp_h)
        scale = self.cfg.answer_alpha_scale if phase == "answer" else self.cfg.think_alpha_scale
        strength = float(self.cfg.alpha) * float(scale) * g
        if strength != 0.0:
            adjusted = step_logits - strength * self.entropy_vec.unsqueeze(0).to(step_logits.dtype)
        else:
            adjusted = step_logits
        return adjusted, {
            "current_token_id": greedy,
            "current_token_entropy": float(self.entropy_vec[greedy].item()),
            "expected_entropy": exp_h,
            "gate_value": g,
            "gate_applied": bool(g > 0.0),
            "phase": phase,
            "penalty_strength": strength,
        }
