"""
Qwen2.5-VL OTT with novel extensions: "TRACE" (Triggered, Relevance-gated,
Attention-prioritised Context rEtracing).

Original OTT (Fazhan-cs/OTT, pope_fast.py, LLaVA only):
  * CRC steering is applied in early layers and SVC (visual retracing) fires once
    at a fixed layer (ending_layer + 1 = 16); the entropy-triggered variant and
    the relevance gate exist only as commented-out code;
  * the steering is applied to every sequence position.

The previous MedicalHall port applied CRC and SVC in *every* layer to *every*
position, subsampled visual references uniformly (linspace), and took SVC
references from the residual stream while blending them into MLP outputs.

Novel changes (all switchable; the "port" preset reproduces the previous port):
  N2.a  Layer-scheduled intervention: CRC and SVC get separate layer sets
        ("auto": CRC in the first half of the layers, SVC in the next layer),
        restoring OTT's schedule on a new architecture instead of stacking both
        everywhere.
  N2.b  Decode-only steering: prompt positions (including all image tokens) are
        never modified, so the visual evidence that AVEG's entropy and SGRS's
        saliency are computed from stays clean.
  N2.c  Attention-prioritised visual memory: the M visual references are the
        image tokens that receive the most attention from the end of the prompt,
        not a uniform subsample.  The same indices are exported as visual anchors
        for LocoRE-V (saliency_qwen.py), coupling OTT with the saliency method.
  N2.d  Space-consistent retracing: SVC references are MLP outputs of the visual
        tokens (the space the retraced MLP output lives in) instead of
        residual-stream states.
  N2.e  Relevance gate: the SVC ratio of each position is scaled by
        (1 + cos(h, mean visual ref)) / 2  (activates an idea OTT left commented out).
  N2.f  Risk-triggered retracing: SVC strength is a per-position gate written by
        the risk router.  A position keeps the gate it had when its token was
        produced, so full-sequence recomputation stays self-consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class OTTConfig:
    use_crc: bool = True
    use_svc: bool = True
    crc_lambda: float = 0.1111
    svc_ratio: float = 0.06
    max_visual_tokens: int = 128
    # --- novel options; defaults reproduce the previous port ---
    crc_layers: str = "all"            # N2.a  e.g. "0-17" or "auto"
    svc_layers: str = "all"            # N2.a  e.g. "18" or "auto"
    decode_only: bool = False          # N2.b
    visual_select: str = "uniform"     # N2.c  "uniform" | "attention"
    svc_ref_space: str = "residual"    # N2.d  "residual" | "mlp"
    relevance_gate: bool = False       # N2.e
    default_svc_gate: float = 1.0      # N2.f  gate for positions the router never set


def _get_language_layers(model):
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
        "Inspect model.named_modules() and add the correct path in _get_language_layers()."
    )


def _get_image_token_id(model) -> int:
    for x in (
        getattr(getattr(model, "config", None), "image_token_id", None),
        getattr(getattr(getattr(model, "config", None), "vision_config", None), "image_token_id", None),
    ):
        if x is not None:
            return int(x)
    raise RuntimeError("model.config.image_token_id was not found.")


def parse_layer_range(spec: str, n_layers: int, kind: str) -> List[int]:
    """'all', 'none', 'auto', '3', '0-15', '2,5,9-11'; negative indices allowed."""
    text = str(spec).strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text in ("none", ""):
        return []
    if text == "auto":
        # OTT on LLaVA-1.5 (32 layers): CRC in layers 0..15, SVC at layer 16.
        split = max(1, int(round(n_layers * 16 / 32)))
        return list(range(split)) if kind == "crc" else [min(split, n_layers - 1)]
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            left, right = part[1:].split("-", 1)
            lo, hi = int(part[0] + left), int(right)
            lo = lo + n_layers if lo < 0 else lo
            hi = hi + n_layers if hi < 0 else hi
            vals = list(range(lo, hi + 1))
        else:
            v = int(part)
            vals = [v + n_layers if v < 0 else v]
        for v in vals:
            if v < 0 or v >= n_layers:
                raise ValueError(f"{kind} layer {v} outside [0, {n_layers - 1}]")
            if v not in out:
                out.append(v)
    return out


class QwenOTTController:
    """Hook-based OTT for Qwen2.5-VL (no transformers internals are patched)."""

    def __init__(self, model, config: OTTConfig):
        self.model = model
        self.cfg = config
        self.layers = _get_language_layers(model)
        n = len(self.layers)
        self.crc_layer_set = set(parse_layer_range(config.crc_layers, n, "crc")) if config.use_crc else set()
        self.svc_layer_set = set(parse_layer_range(config.svc_layers, n, "svc")) if config.use_svc else set()
        self.handles = []
        self.enabled = False

        self.visual_refs: Dict[int, torch.Tensor] = {}
        self.crc_refs: Dict[int, torch.Tensor] = {}
        self.visual_means: Dict[int, torch.Tensor] = {}
        self.prompt_len: Optional[int] = None
        self.visual_priority: Optional[torch.Tensor] = None         # absolute positions, sorted by position (N2.c)
        self.visual_priority_ranked: Optional[torch.Tensor] = None  # same tokens, most-attended first
        self.position_svc_gate: Dict[int, float] = {}         # N2.f
        self._gate_cache = None
        self.svc_applications = 0

    # ------------------------------------------------------------------ hooks
    def install(self):
        if self.handles:
            return
        for idx, layer in enumerate(self.layers):
            if idx not in self.crc_layer_set and idx not in self.svc_layer_set:
                continue
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"Decoder layer {idx} has no .mlp module.")
            self.handles.append(mlp.register_forward_hook(self._make_hook(idx)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    # ------------------------------------------------------- risk gate (N2.f)
    def set_position_gate(self, abs_pos: int, value: float):
        self.position_svc_gate[int(abs_pos)] = float(value)
        self._gate_cache = None

    def get_position_gate(self, abs_pos: int) -> float:
        return self.position_svc_gate.get(int(abs_pos), float(self.cfg.default_svc_gate))

    def _row_weights(self, seq_len: int, device) -> Optional[torch.Tensor]:
        """Per-position multipliers: row 0 = CRC on/off, row 1 = SVC gate."""
        if not self.cfg.decode_only and not self.position_svc_gate and self.cfg.default_svc_gate == 1.0:
            return None  # previous-port behaviour: uniform over positions
        key = (seq_len, str(device))
        if self._gate_cache is not None and self._gate_cache[0] == key:
            return self._gate_cache[1]
        crc_w = torch.ones(seq_len, dtype=torch.float32)
        svc_w = torch.full((seq_len,), float(self.cfg.default_svc_gate), dtype=torch.float32)
        first_decode = 0
        if self.cfg.decode_only and self.prompt_len is not None:
            # Row prompt_len-1 predicts the first generated token, so it is a decode row.
            first_decode = max(0, self.prompt_len - 1)
            crc_w[:first_decode] = 0.0
            svc_w[:first_decode] = 0.0
        for pos, val in self.position_svc_gate.items():
            if first_decode <= pos < seq_len:
                svc_w[pos] = val
        w = torch.stack([crc_w, svc_w], dim=0).to(device)
        self._gate_cache = (key, w)
        return w

    # ------------------------------------------------------------ reference
    @torch.no_grad()
    def prepare_reference(self, base_inputs: dict):
        """Builds per-layer visual references from the image prompt (once per sample)."""
        self.disable()
        self.position_svc_gate.clear()
        self._gate_cache = None
        self.svc_applications = 0

        input_ids = base_inputs["input_ids"]
        self.prompt_len = int(input_ids.shape[-1])
        visual_mask = input_ids[0].eq(_get_image_token_id(self.model))
        if not torch.any(visual_mask):
            raise RuntimeError("No visual placeholder tokens found. Check processor/model compatibility.")
        text_mask = ~visual_mask
        visual_pos = torch.nonzero(visual_mask, as_tuple=False).flatten()

        # Capture MLP outputs of SVC layers if references live in MLP space (N2.d).
        mlp_out: Dict[int, torch.Tensor] = {}
        tmp = []
        if self.cfg.svc_ref_space == "mlp":
            for idx, layer in enumerate(self.layers):
                if idx in self.svc_layer_set:
                    def cap(module, inp, out, idx=idx):
                        mlp_out[idx] = out[0].detach()
                    tmp.append(layer.mlp.register_forward_hook(cap))
        want_attn = self.cfg.visual_select == "attention"
        try:
            outputs = self.model(
                **base_inputs,
                output_hidden_states=True,
                output_attentions=want_attn,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for h in tmp:
                h.remove()
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen forward did not return hidden_states.")

        # ---- choose which visual tokens form the memory (N2.c)
        m = int(self.cfg.max_visual_tokens)
        n_vis = int(visual_pos.numel())
        attns = getattr(outputs, "attentions", None) if want_attn else None
        ranked = None
        if attns is not None and len(attns) > 0 and attns[0] is not None:
            # Attention mass the last prompt rows (the question / generation
            # prompt) put on each image token, averaged over heads and layers.
            tail = min(4, int(input_ids.shape[-1]))
            score = torch.zeros(n_vis, device=visual_pos.device, dtype=torch.float32)
            for att in attns:
                score += att[0, :, -tail:, :].index_select(-1, visual_pos).float().mean(dim=(0, 1))
            ranked = torch.argsort(score, descending=True)
        if n_vis <= m:
            local_sel = torch.arange(n_vis, device=visual_pos.device)
        elif ranked is not None:
            local_sel = ranked[:m].sort().values
        else:
            local_sel = torch.linspace(0, n_vis - 1, m, device=visual_pos.device).long()
        self.visual_priority = visual_pos[local_sel]
        self.visual_priority_ranked = visual_pos[ranked[: min(m, n_vis)]] if ranked is not None else self.visual_priority
        del outputs, attns

        self.visual_refs.clear()
        self.crc_refs.clear()
        self.visual_means.clear()
        n_layers = min(len(self.layers), len(hidden_states) - 1)
        for idx in range(n_layers):
            hs = hidden_states[idx + 1][0].detach()  # residual stream after layer idx
            visual_all = hs[visual_mask]
            if idx in self.svc_layer_set:
                src = mlp_out[idx][visual_mask] if idx in mlp_out else visual_all
                ref = src.index_select(0, local_sel)
                self.visual_refs[idx] = ref
                self.visual_means[idx] = F.normalize(ref.float().mean(dim=0, keepdim=True), dim=-1)
            if idx in self.crc_layer_set:
                visual = visual_all.index_select(0, local_sel)
                visual_mean = visual.float().mean(dim=0, keepdim=True)
                if torch.any(text_mask):
                    direction = visual_mean - hs[text_mask].float().mean(dim=0, keepdim=True)
                else:
                    direction = visual_mean
                self.crc_refs[idx] = F.normalize(direction, p=2, dim=-1)

    # ----------------------------------------------------------------- hook
    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self.enabled or not torch.is_tensor(output):
                return output
            y = output
            original_dtype = y.dtype
            yf = y.float()
            weights = self._row_weights(int(y.shape[-2]), y.device)

            if layer_idx in self.crc_refs:
                d = self.crc_refs[layer_idx].to(y.device, dtype=torch.float32)
                norm = torch.norm(yf, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
                yn = F.normalize(yf, p=2, dim=-1)
                steered = F.normalize(yn + self.cfg.crc_lambda * d, p=2, dim=-1) * norm
                if weights is None:
                    yf = steered
                else:
                    yf = torch.where(weights[0].view(1, -1, 1) > 0, steered, yf)

            if layer_idx in self.visual_refs:
                visual = self.visual_refs[layer_idx].to(y.device, dtype=torch.float32)
                if visual.numel() > 0:
                    qn = F.normalize(yf, p=2, dim=-1)
                    kn = F.normalize(visual, p=2, dim=-1)
                    attn = torch.softmax(torch.matmul(qn, kn.transpose(-1, -2)), dim=-1)
                    context = torch.matmul(attn, visual)
                    context_norm = torch.norm(context, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
                    target_norm = torch.norm(yf, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
                    context = context / context_norm * target_norm
                    r = self.cfg.svc_ratio
                    if self.cfg.relevance_gate:
                        vm = self.visual_means[layer_idx].to(y.device)
                        r = r * (1.0 + (qn * vm).sum(dim=-1, keepdim=True)) / 2.0
                    if weights is not None:
                        r = r * weights[1].view(1, -1, 1)
                    yf = (1.0 - r) * yf + r * context
                    self.svc_applications += 1

            return yf.to(original_dtype)

        return hook
