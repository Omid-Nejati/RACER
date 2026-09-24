"""
LVLMs-Saliency (SGRS + LocoRE) for Qwen2.5-VL with novel extensions.

Original method (zhangbaijin/LVLMs-Saliency, "Hallucination Begins Where
Saliency Drops"):
  * SGRS: saliency = |A * dL/dA| on attention to previously generated OUTPUT
    tokens only; fixed threshold alpha * mean(recent accepted saliency);
    fallback = max-saliency candidate;
  * LocoRE: attention to the last W output tokens is multiplied by (1 + beta)
    with constant beta.

Novel changes (all switchable; the "port" preset reproduces the previous port):
  N3.a  Dual (text + visual) grounding saliency: the candidate score also
        measures gradient-weighted attention to IMAGE keys, not only to the text
        history.  Both terms are made scale-free by dividing by their own
        running means, so they can be mixed with one weight lambda.
  N3.b  Entropy-coupled per-candidate threshold: candidates whose AVEG visual
        entropy H[c] is high must clear a stricter saliency threshold,
        alpha_c = clip(alpha * (1 + kappa * (H[c] - gamma))).  This couples
        MCoT's "is this token visually supported?" with the saliency test.
  N3.c  Calibrated fallback: when all samples are rejected, pick
        argmax_c log p_c + eta * log s_c instead of pure max-saliency, so a
        near-zero-probability token cannot win on saliency alone.
  N3.d  Warm-start fix: the neutral placeholder score used before any history
        exists is not written into the history (it otherwise inflates the
        threshold for the next W steps and causes spurious rejections).
  N4.a  Saliency-adaptive LocoRE: each decode row gets its own gain
        beta_t = beta * clip(mean_saliency / last_saliency, 1, beta_max/beta):
        reinforcement grows exactly when saliency drops.
  N4.b  Visual-anchored LocoRE (LocoRE-V): the same additive gain is applied to
        the top visually-prioritised image tokens (shared with OTT N2.c), so
        LocoRE reinforces both output coherence and visual grounding.
  (eng.) LocoRE bias is built once per forward pass (vectorised) instead of a
        Python loop over rows in every layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def get_language_layers(model):
    """Locate the text decoder layers across common Qwen2/2.5-VL wrappers."""
    for getter in (
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
    ):
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except Exception:
            pass
    raise RuntimeError(
        "Could not locate Qwen language decoder layers. "
        "Inspect model.named_modules() and extend get_language_layers()."
    )


def parse_layer_spec(spec: str | Sequence[int], n_layers: int) -> List[int]:
    """Parse '6,7,8', 'all', or a list into validated layer indices."""
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text == "all":
            return list(range(n_layers))
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        values = [int(x) for x in spec]
    out: List[int] = []
    for idx in values:
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"layer index {idx} is outside [0, {n_layers - 1}]")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("at least one target layer is required")
    return out


# =============================================================== configs
@dataclass
class SaliencyConfig:
    top_k: int = 5
    max_resample: int = 3
    alpha: float = 0.6
    history_window: int = 5
    target_layers: str = "6,7,8"
    query_mode: str = "predictor"      # "predictor" | "repo"
    # --- novel options; defaults reproduce the previous port ---
    visual_lambda: float = 0.0         # N3.a  weight of visual-grounding saliency
    entropy_kappa: float = 0.0         # N3.b
    entropy_gamma: float = 0.5
    alpha_min: float = 0.3
    alpha_max: float = 0.95
    fallback_eta: Optional[float] = None   # N3.c  None = pure max-saliency
    exclude_neutral_history: bool = False  # N3.d


@dataclass
class LocoREConfig:
    beta: float = 0.20
    window: int = 5
    layers: str = "all"
    strict_mask: bool = True
    # --- novel options ---
    adaptive: bool = False             # N4.a
    beta_max: float = 0.60
    visual_beta: float = 0.0           # N4.b
    visual_anchor_k: int = 32


@dataclass
class SaliencyState:
    """Per-sample SGRS history."""
    txt_hist: List[float] = field(default_factory=list)
    vis_hist: List[float] = field(default_factory=list)
    last_relative: Optional[float] = None   # used by the risk router
    last_txt: Optional[float] = None

    def window(self, values: List[float], w: int) -> List[float]:
        return values[-int(w):] if w > 0 else values

    def mean_txt(self, w: int) -> Optional[float]:
        h = self.window(self.txt_hist, w)
        return sum(h) / len(h) if h else None

    def mean_vis(self, w: int) -> Optional[float]:
        h = self.window(self.vis_hist, w)
        return sum(h) / len(h) if h else None


# ================================================================ LocoRE
class LocoREController:
    """
    Portable Qwen2.5-VL LocoRE: adds log(1 + beta) to the additive causal mask
    (softmax-equivalent to multiplying the attention probability by 1 + beta).
    Requires eager attention and a 4-D additive mask at decoder-layer entry.
    """

    def __init__(self, model, config: LocoREConfig):
        self.model = model
        self.cfg = config
        self.layers = get_language_layers(model)
        self.layer_ids = parse_layer_spec(config.layers, len(self.layers))
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.enabled = False
        self.prompt_len: Optional[int] = None
        self.applied_calls = 0
        self.skipped_mask_calls = 0
        self.row_beta: Dict[int, float] = {}            # N4.a
        self.visual_anchors: Optional[torch.Tensor] = None  # N4.b (absolute key positions)
        self._version = 0
        self._cache = None

    # ---- setup
    def set_prompt_len(self, prompt_len: int):
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive")
        self.prompt_len = int(prompt_len)
        self.row_beta.clear()
        self.visual_anchors = None
        self._bump()

    def set_row_beta(self, abs_row: int, beta: float):
        self.row_beta[int(abs_row)] = float(beta)
        self._bump()

    def set_visual_anchors(self, positions: Optional[torch.Tensor]):
        if positions is None or int(self.cfg.visual_anchor_k) <= 0:
            self.visual_anchors = None
        else:
            self.visual_anchors = positions[: int(self.cfg.visual_anchor_k)].detach().clone()
        self._bump()

    def _bump(self):
        self._version += 1
        self._cache = None

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def install(self):
        if self.handles:
            return
        for idx in self.layer_ids:
            self.handles.append(self.layers[idx].register_forward_pre_hook(self._pre_hook, with_kwargs=True))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ---- bias construction (once per forward)
    def _simple(self) -> bool:
        return not self.row_beta and (self.visual_anchors is None or self.cfg.visual_beta <= 0)

    def _build(self, q_len: int, kv_len: int, device, dtype):
        key = (q_len, kv_len, self._version, str(device), dtype)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        P, W = int(self.prompt_len), int(self.cfg.window)
        q_abs = torch.arange(kv_len - q_len, kv_len, device=device).view(-1, 1)
        k = torch.arange(kv_len, device=device).view(1, -1)
        recent = (q_abs >= P) & (k >= P) & (k < q_abs) & (k >= q_abs - W)
        if self._simple():
            value = recent  # boolean; added with a python scalar (bit-identical to the port)
        else:
            betas = torch.full((q_len,), float(self.cfg.beta), dtype=torch.float32)
            base = kv_len - q_len
            for row, b in self.row_beta.items():
                if base <= row < kv_len:
                    betas[row - base] = b
            gains = torch.log1p(betas.clamp_min(0.0)).to(device).view(-1, 1)
            bias = recent.float() * gains
            if self.visual_anchors is not None and self.cfg.visual_beta > 0:
                vis = torch.zeros(kv_len, dtype=torch.bool, device=device)
                anchors = self.visual_anchors.to(device)
                vis[anchors[anchors < kv_len]] = True
                decode_rows = q_abs >= P
                bias = bias + (decode_rows & vis.view(1, -1)).float() * math.log1p(float(self.cfg.visual_beta))
            value = bias
        changed = bool(value.any().item())
        self._cache = (key, (value, changed))
        return value, changed

    def _pre_hook(self, module, args, kwargs):
        if not self.enabled or self.prompt_len is None:
            return args, kwargs
        if self.cfg.beta <= 0 and self.cfg.visual_beta <= 0 and not self.row_beta:
            return args, kwargs
        if self.cfg.window <= 0 and self.cfg.visual_beta <= 0:
            return args, kwargs
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None or not torch.is_tensor(attention_mask) or attention_mask.ndim != 4:
            self.skipped_mask_calls += 1
            return args, kwargs
        q_len, kv_len = int(attention_mask.shape[-2]), int(attention_mask.shape[-1])
        value, changed = self._build(q_len, kv_len, attention_mask.device, attention_mask.dtype)
        if not changed:
            return args, kwargs
        if value.dtype == torch.bool:
            gain = math.log1p(float(self.cfg.beta))
            new_mask = torch.where(value.view(1, 1, q_len, kv_len), attention_mask + gain, attention_mask)
        else:
            new_mask = (attention_mask.float() + value.view(1, 1, q_len, kv_len)).to(attention_mask.dtype)
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = new_mask
        self.applied_calls += 1
        return args, kwargs

    def validate_active(self, generated_tokens: int):
        if not self.enabled or self.cfg.beta <= 0 or generated_tokens <= 1:
            return
        if self.cfg.strict_mask and self.applied_calls == 0:
            raise RuntimeError(
                "LocoRE did not observe a 4-D additive causal mask. "
                "Run with --attn_implementation eager and transformers==4.49.0, "
                "or use --locore_non_strict while debugging."
            )


def adaptive_locore_beta(cfg: LocoREConfig, state: SaliencyState, window: int) -> float:
    """N4.a: beta_t = beta * clip(mean / last, 1, beta_max / beta)."""
    mu = state.mean_txt(window)
    last = state.last_txt
    if mu is None or last is None or last <= 0 or cfg.beta <= 0:
        return float(cfg.beta)
    ratio = min(max(mu / last, 1.0), cfg.beta_max / cfg.beta)
    return float(cfg.beta * ratio)


# ============================================================ saliency
def _full_inputs(base_inputs, input_ids, attention_mask):
    inputs = dict(base_inputs)
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    return inputs


def compute_candidate_saliency_parts(
    model,
    base_inputs: Dict[str, torch.Tensor],
    seq: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_id: int,
    prompt_len: int,
    target_layers: Sequence[int],
    query_mode: str = "predictor",
    visual_positions: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """
    Gradient-aware attention saliency |A * dL/dA| (head-mean, row-L2-normalised)
    of one candidate.  Returns the text-history part (original method) and,
    if visual_positions is given, the visual-grounding part (N3.a).
    """
    if query_mode not in {"predictor", "repo"}:
        raise ValueError("query_mode must be 'predictor' or 'repo'")
    if seq.shape[-1] <= prompt_len:
        return {"txt": 1.0, "vis": None, "neutral": True}

    device = seq.device
    cand = torch.tensor([[int(candidate_id)]], device=device, dtype=seq.dtype)
    seq2 = torch.cat([seq, cand], dim=-1)
    mask2 = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
        dim=-1,
    )

    model.zero_grad(set_to_none=True)
    neutral = False
    with torch.enable_grad():
        outputs = model(**_full_inputs(base_inputs, seq2, mask2), use_cache=False,
                        output_attentions=True, return_dict=True)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise RuntimeError("Model did not return attentions. SGRS requires --attn_implementation eager.")
        pred_logits = outputs.logits[:, -2, :].float()
        loss = -F.log_softmax(pred_logits, dim=-1)[0, int(candidate_id)]

        txt_scores: List[torch.Tensor] = []
        vis_scores: List[torch.Tensor] = []
        for layer_idx in target_layers:
            attn = attentions[layer_idx]
            if attn is None:
                continue
            grad = torch.autograd.grad(loss, attn, retain_graph=True, allow_unused=True, create_graph=False)[0]
            if grad is None:
                continue
            sal = (attn.float() * grad.float()).abs()
            tq, tk = sal.shape[-2], sal.shape[-1]
            tri = torch.tril(torch.ones((tq, tk), device=sal.device, dtype=sal.dtype))
            sal = sal * tri.view(1, 1, tq, tk)
            sal = sal.mean(dim=1)
            sal = sal / sal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)
            q_idx = tq - 2 if query_mode == "predictor" else tq - 1
            if q_idx < 0:
                continue
            hist_start, hist_end = int(prompt_len), q_idx
            if hist_start >= hist_end:
                txt_scores.append(sal.new_tensor(1.0))
                neutral = True
            else:
                txt_scores.append(sal[0, q_idx, hist_start:hist_end].mean())
            if visual_positions is not None and visual_positions.numel() > 0:
                vis_scores.append(sal[0, q_idx].index_select(0, visual_positions.to(sal.device)).mean())

        if not txt_scores:
            raise RuntimeError("No usable attention gradients were produced for SGRS (check eager attention).")
        txt = float(torch.stack(txt_scores).mean().detach().cpu().item())
        vis = float(torch.stack(vis_scores).mean().detach().cpu().item()) if vis_scores else None

    model.zero_grad(set_to_none=True)
    return {"txt": txt, "vis": vis, "neutral": neutral}


def compute_candidate_saliency(model, base_inputs, seq, attention_mask, candidate_id, prompt_len,
                               target_layers, query_mode="predictor") -> float:
    """Backward-compatible scalar API (text-history saliency only)."""
    return float(compute_candidate_saliency_parts(
        model, base_inputs, seq, attention_mask, candidate_id, prompt_len, target_layers, query_mode
    )["txt"])


def _combined_score(parts: Dict[str, object], state: SaliencyState, cfg: SaliencyConfig) -> Tuple[float, Optional[float]]:
    """Returns (score used for the test, relative score for logging/router)."""
    mu_t = state.mean_txt(cfg.history_window)
    txt = float(parts["txt"])
    rel_t = txt / mu_t if mu_t and mu_t > 0 else None
    lam = float(cfg.visual_lambda)
    if lam <= 0:
        return txt, rel_t
    mu_v = state.mean_vis(cfg.history_window)
    vis = parts.get("vis")
    rel_v = (float(vis) / mu_v) if (vis is not None and mu_v and mu_v > 0) else None
    if rel_t is None and rel_v is None:
        return float("inf"), None
    if rel_t is None:
        return rel_v, rel_v
    if rel_v is None:
        return rel_t, rel_t
    rel = (1.0 - lam) * rel_t + lam * rel_v
    return rel, rel


def _candidate_alpha(cid: int, cfg: SaliencyConfig, entropy_vec: Optional[torch.Tensor]) -> float:
    if cfg.entropy_kappa <= 0 or entropy_vec is None:
        return float(cfg.alpha)
    h = float(entropy_vec[int(cid)].item())
    a = cfg.alpha * (1.0 + cfg.entropy_kappa * (h - cfg.entropy_gamma))
    return float(min(max(a, cfg.alpha_min), cfg.alpha_max))


def _threshold(cid: int, cfg: SaliencyConfig, state: SaliencyState, entropy_vec) -> float:
    a_c = _candidate_alpha(cid, cfg, entropy_vec)
    if cfg.visual_lambda > 0:
        return a_c  # relative (scale-free) test
    mu_t = state.mean_txt(cfg.history_window)
    return a_c * mu_t if mu_t is not None else 0.0


def select_sgrs_token(
    model,
    tokenizer,
    base_inputs: Dict[str, torch.Tensor],
    seq: torch.Tensor,
    attention_mask: torch.Tensor,
    logits: torch.Tensor,
    prompt_len: int,
    state: SaliencyState,
    config: SaliencyConfig,
    generator: Optional[torch.Generator] = None,
    visual_positions: Optional[torch.Tensor] = None,
    entropy_vec: Optional[torch.Tensor] = None,
) -> Tuple[int, Dict]:
    """Top-K -> (dual) saliency rejection with per-candidate threshold -> calibrated fallback."""
    layers = get_language_layers(model)
    target_layers = parse_layer_spec(config.target_layers, len(layers))
    vis_pos = visual_positions if config.visual_lambda > 0 else None

    probs = F.softmax(logits.float(), dim=-1).squeeze(0)
    k = min(int(config.top_k), int(probs.numel()))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    top_probs = top_probs / top_probs.sum().clamp_min(1e-12)
    candidates = [int(x) for x in top_ids.tolist()]
    prob_map = {cid: float(p) for cid, p in zip(candidates, top_probs.tolist())}

    def evaluate(cid):
        return compute_candidate_saliency_parts(
            model, base_inputs, seq, attention_mask, cid, prompt_len, target_layers, config.query_mode, vis_pos
        )

    remaining = list(candidates)
    parts: Dict[int, Dict] = {}
    scores: Dict[int, float] = {}
    rel: Dict[int, Optional[float]] = {}
    taus: Dict[int, float] = {}
    rejected: List[int] = []
    chosen: Optional[int] = None
    fallback = False

    for _ in range(int(config.max_resample)):
        if not remaining:
            break
        weights = torch.tensor([prob_map[c] for c in remaining], device=logits.device, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        pick = int(torch.multinomial(weights, 1, generator=generator).item())
        cid = remaining[pick]
        parts[cid] = evaluate(cid)
        scores[cid], rel[cid] = _combined_score(parts[cid], state, config)
        taus[cid] = _threshold(cid, config, state, entropy_vec)
        if scores[cid] >= taus[cid]:
            chosen = cid
            break
        rejected.append(cid)
        remaining.pop(pick)

    if chosen is None:
        fallback = True
        for cid in candidates:
            if cid not in parts:
                parts[cid] = evaluate(cid)
                scores[cid], rel[cid] = _combined_score(parts[cid], state, config)
                taus[cid] = _threshold(cid, config, state, entropy_vec)
        if config.fallback_eta is None:
            chosen = max(candidates, key=lambda c: scores[c])
        else:
            eta = float(config.fallback_eta)
            chosen = max(
                candidates,
                key=lambda c: math.log(max(prob_map[c], 1e-12)) + eta * math.log(max(scores[c], 1e-12)),
            )

    p = parts.get(chosen, {"txt": 1.0, "vis": None, "neutral": True})
    if not (p.get("neutral") and config.exclude_neutral_history):
        state.txt_hist.append(float(p["txt"]))
        if p.get("vis") is not None:
            state.vis_hist.append(float(p["vis"]))
    if not p.get("neutral"):
        state.last_txt = float(p["txt"])
        state.last_relative = rel.get(chosen)

    meta = {
        "chosen_id": int(chosen),
        "chosen_token": tokenizer.decode([int(chosen)], skip_special_tokens=False).replace("\n", "\\n"),
        "chosen_saliency": float(p["txt"]),
        "chosen_visual_saliency": p.get("vis"),
        "chosen_relative": rel.get(chosen),
        "fallback": fallback,
        "rejected_ids": rejected,
        "candidates": [
            {
                "id": cid,
                "token": tokenizer.decode([cid], skip_special_tokens=False).replace("\n", "\\n"),
                "topk_prob": prob_map[cid],
                "saliency": parts[cid]["txt"] if cid in parts else None,
                "visual_saliency": parts[cid].get("vis") if cid in parts else None,
                "score": scores.get(cid),
                "threshold": taus.get(cid),
            }
            for cid in candidates
        ],
    }
    return int(chosen), meta


@torch.no_grad()
def attention_visual_priority(model, base_inputs: Dict[str, torch.Tensor], image_token_id: int, tail: int = 4) -> torch.Tensor:
    """
    Image-token positions ranked by the attention they receive from the last
    prompt rows (head/layer mean).  Used for LocoRE-V anchors when OTT is off;
    with OTT on, OTT's own ranking (N2.c) is reused so no extra forward is spent.
    """
    input_ids = base_inputs["input_ids"]
    visual_pos = torch.nonzero(input_ids[0].eq(int(image_token_id)), as_tuple=False).flatten()
    out = model(**base_inputs, output_attentions=True, use_cache=False, return_dict=True)
    if out.attentions is None or out.attentions[0] is None:
        return visual_pos
    t = min(int(tail), int(input_ids.shape[-1]))
    score = torch.zeros(visual_pos.numel(), device=visual_pos.device, dtype=torch.float32)
    for att in out.attentions:
        score += att[0, :, -t:, :].index_select(-1, visual_pos).float().mean(dim=(0, 1))
    return visual_pos[torch.argsort(score, descending=True)]
