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


def _get_language_layers(model):
    candidates = [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
    ]
    for name, getter in candidates:
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
    candidates = [
        getattr(getattr(model, "config", None), "image_token_id", None),
        getattr(getattr(getattr(model, "config", None), "vision_config", None), "image_token_id", None),
    ]
    for x in candidates:
        if x is not None:
            return int(x)
    raise RuntimeError("model.config.image_token_id was not found.")


class QwenOTTController:
    """
    Qwen-compatible OTT-style intervention.

    The original OTT release is coupled to LLaVA/LlamaMLP.  This controller
    keeps the two mechanisms used by the released OTT run script:
      - CRC-like normalized visual steering
      - SVC-like visual-context retracing

    It applies them with forward hooks, so no transformers internals are
    monkey-patched and the MCoT Qwen2.5-VL environment can remain unchanged.
    """

    def __init__(self, model, config: OTTConfig):
        self.model = model
        self.cfg = config
        self.layers = _get_language_layers(model)
        self.handles = []
        self.enabled = False

        self.visual_refs: Dict[int, torch.Tensor] = {}
        self.crc_refs: Dict[int, torch.Tensor] = {}

    def install(self):
        if self.handles:
            return
        for idx, layer in enumerate(self.layers):
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

    @torch.no_grad()
    def prepare_reference(self, base_inputs: dict):
        """
        Build per-layer visual references from the original image prompt.
        Must be called once per sample before generation.
        """
        self.disable()

        outputs = self.model(
            **base_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen forward did not return hidden_states.")

        input_ids = base_inputs["input_ids"]
        image_token_id = _get_image_token_id(self.model)
        visual_mask = input_ids[0].eq(image_token_id)

        if not torch.any(visual_mask):
            raise RuntimeError(
                "No visual placeholder tokens found. "
                "Check processor/model compatibility and image_token_id."
            )

        text_mask = ~visual_mask

        self.visual_refs.clear()
        self.crc_refs.clear()

        n_layers = min(len(self.layers), len(hidden_states) - 1)
        for idx in range(n_layers):
            # hidden_states[idx+1] is the representation after decoder layer idx.
            hs = hidden_states[idx + 1][0].detach()
            visual = hs[visual_mask]

            if visual.shape[0] > self.cfg.max_visual_tokens:
                ids = torch.linspace(
                    0, visual.shape[0] - 1,
                    self.cfg.max_visual_tokens,
                    device=visual.device,
                ).long()
                visual = visual.index_select(0, ids)

            self.visual_refs[idx] = visual

            # A Qwen-safe visual steering direction.  Centering by the textual
            # prompt prevents a hard-coded LLaVA hidden size or LLaVA-only
            # positive/negative prompt machinery.
            visual_mean = visual.float().mean(dim=0, keepdim=True)
            if torch.any(text_mask):
                text_mean = hs[text_mask].float().mean(dim=0, keepdim=True)
                direction = visual_mean - text_mean
            else:
                direction = visual_mean

            self.crc_refs[idx] = F.normalize(direction, p=2, dim=-1)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self.enabled or not torch.is_tensor(output):
                return output

            y = output
            original_dtype = y.dtype
            yf = y.float()

            # CRC: normalized direction addition while approximately preserving
            # the original activation norm, matching OTT's released behavior.
            if self.cfg.use_crc and layer_idx in self.crc_refs:
                d = self.crc_refs[layer_idx].to(y.device, dtype=torch.float32)
                norm = torch.norm(yf, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
                yn = F.normalize(yf, p=2, dim=-1)
                yf = F.normalize(
                    yn + self.cfg.crc_lambda * d,
                    p=2,
                    dim=-1,
                ) * norm

            # SVC: retrieve visual context for current hidden states and blend it
            # back into the MLP output.
            if self.cfg.use_svc and layer_idx in self.visual_refs:
                visual = self.visual_refs[layer_idx].to(y.device, dtype=torch.float32)
                if visual.numel() > 0:
                    q = yf
                    qn = F.normalize(q, p=2, dim=-1)
                    kn = F.normalize(visual, p=2, dim=-1)

                    scores = torch.matmul(qn, kn.transpose(-1, -2))
                    weights = torch.softmax(scores, dim=-1)
                    context = torch.matmul(weights, visual)

                    context_norm = torch.norm(
                        context, p=2, dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    target_norm = torch.norm(
                        yf, p=2, dim=-1, keepdim=True
                    ).clamp_min(1e-6)

                    context = context / context_norm * target_norm
                    r = self.cfg.svc_ratio
                    yf = (1.0 - r) * yf + r * context

            return yf.to(original_dtype)

        return hook
