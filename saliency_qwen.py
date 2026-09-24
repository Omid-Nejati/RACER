from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn.functional as F


def get_language_layers(model):
    """Locate the text decoder layers across common Qwen2/2.5-VL wrappers."""
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
    ]
    for getter in candidates:
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


@dataclass
class SaliencyConfig:
    top_k: int = 5
    max_resample: int = 3
    alpha: float = 0.6
    history_window: int = 5
    target_layers: str = "6,7,8"
    # predictor = use the attention row that actually predicts the candidate.
    # repo = reproduce the public do_reject_sample.py row choice (often zero in
    # a standard causal decoder because the candidate row cannot affect its own
    # already-computed likelihood).
    query_mode: str = "predictor"


@dataclass
class LocoREConfig:
    beta: float = 0.20
    window: int = 5
    layers: str = "all"
    strict_mask: bool = True


class LocoREController:
    """
    Portable Qwen2.5-VL LocoRE implementation.

    The paper multiplies attention to recent output keys by (1 + beta).  Rather
    than forking Transformers' Qwen attention implementation, this controller
    injects log(1 + beta) into the additive causal attention mask.  After the
    softmax this is the normalized equivalent of multiplying those attention
    probabilities by (1 + beta).

    This is a Qwen2.5-VL port, not byte-for-byte code from the authors' Qwen2-VL
    fork.  It requires eager attention and a 4-D additive causal mask at decoder
    layer entry.
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

    def set_prompt_len(self, prompt_len: int):
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive")
        self.prompt_len = int(prompt_len)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def install(self):
        if self.handles:
            return
        for idx in self.layer_ids:
            layer = self.layers[idx]
            self.handles.append(
                layer.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _pre_hook(self, module, args, kwargs):
        if not self.enabled or self.cfg.beta <= 0 or self.cfg.window <= 0:
            return args, kwargs
        if self.prompt_len is None:
            return args, kwargs

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 0 and torch.is_tensor(args[0]):
            hidden_states = args[0]

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None or not torch.is_tensor(attention_mask):
            self.skipped_mask_calls += 1
            if self.cfg.strict_mask and self.applied_calls == 0:
                # Do not raise on the prompt-only first pass; there is nothing to
                # reinforce yet.  Once generated positions exist, the generator
                # calls validate_active() for a clear error if no mask was seen.
                pass
            return args, kwargs

        if attention_mask.ndim != 4:
            self.skipped_mask_calls += 1
            return args, kwargs

        q_len = int(hidden_states.shape[-2]) if hidden_states is not None else int(attention_mask.shape[-2])
        kv_len = int(attention_mask.shape[-1])
        q_mask_len = int(attention_mask.shape[-2])
        if q_mask_len != q_len:
            # Use the mask's query dimension if wrappers change hidden-state
            # packing.  This keeps indexing tied to the actual attention mask.
            q_len = q_mask_len

        # With full-sequence decoding, q_abs_start = 0.  With a KV cache, the
        # query rows correspond to the tail of the key sequence.
        q_abs_start = max(0, kv_len - q_len)
        prompt_len = self.prompt_len
        gain_bias = math.log1p(float(self.cfg.beta))

        new_mask = attention_mask.clone()
        changed = False
        for q_local in range(q_len):
            q_abs = q_abs_start + q_local
            if q_abs < prompt_len:
                continue
            key_start = max(prompt_len, q_abs - int(self.cfg.window))
            key_end = q_abs  # previous outputs only; exclude self
            if key_start >= key_end:
                continue
            new_mask[..., q_local, key_start:key_end] = (
                new_mask[..., q_local, key_start:key_end] + gain_bias
            )
            changed = True

        if changed:
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
                "Run with --attn_implementation eager and transformers==4.49.0. "
                "If your local Qwen build uses a different masking path, use "
                "--locore_non_strict to continue without LocoRE while debugging."
            )


def _full_inputs(base_inputs: Dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor):
    inputs = dict(base_inputs)
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    return inputs


def compute_candidate_saliency(
    model,
    base_inputs: Dict[str, torch.Tensor],
    seq: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_id: int,
    prompt_len: int,
    target_layers: Sequence[int],
    query_mode: str = "predictor",
) -> float:
    """
    Gradient-aware attention saliency for one candidate token.

    Implements |A * dL/dA|, head averaging, row-wise L2 normalization, and
    averaging over previous generated output keys and selected layers.

    query_mode='predictor' uses the causal attention row whose logits predict
    the candidate (T-2 after appending the candidate).  query_mode='repo'
    reproduces the row choice in the public do_reject_sample.py (T-1).
    """
    if query_mode not in {"predictor", "repo"}:
        raise ValueError("query_mode must be 'predictor' or 'repo'")

    device = seq.device
    cand = torch.tensor([[int(candidate_id)]], device=device, dtype=seq.dtype)
    seq2 = torch.cat([seq, cand], dim=-1)
    mask2 = torch.cat(
        [
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
        ],
        dim=-1,
    )

    # If there are no previously generated outputs, follow the public code's
    # convention and accept with a neutral/high saliency score.
    if seq.shape[-1] <= prompt_len:
        return 1.0

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = model(
            **_full_inputs(base_inputs, seq2, mask2),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise RuntimeError(
                "Model did not return attentions. SGRS requires eager attention: "
                "use --attn_implementation eager."
            )

        # In an autoregressive LM, position T-2 predicts the appended candidate.
        pred_logits = outputs.logits[:, -2, :].float()
        loss = -F.log_softmax(pred_logits, dim=-1)[0, int(candidate_id)]

        valid_scores: List[torch.Tensor] = []
        for layer_idx in target_layers:
            attn = attentions[layer_idx]
            if attn is None:
                continue
            grad = torch.autograd.grad(
                loss,
                attn,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )[0]
            if grad is None:
                continue

            sal = (attn.float() * grad.float()).abs()
            # causal lower triangle, then average heads
            tq, tk = sal.shape[-2], sal.shape[-1]
            tri = torch.tril(torch.ones((tq, tk), device=sal.device, dtype=sal.dtype))
            sal = sal * tri.view(1, 1, tq, tk)
            sal = sal.mean(dim=1)  # (B, Tq, Tk)
            sal = sal / sal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)

            if query_mode == "predictor":
                q_idx = tq - 2
            else:
                q_idx = tq - 1

            if q_idx < 0:
                continue
            # Previous generated outputs begin immediately after the fixed
            # multimodal prompt.  Exclude the predictor/current token itself.
            hist_start = int(prompt_len)
            hist_end = q_idx
            if hist_start >= hist_end:
                score = sal.new_tensor(1.0)
            else:
                score = sal[0, q_idx, hist_start:hist_end].mean()
            valid_scores.append(score)

        if not valid_scores:
            raise RuntimeError(
                "No usable attention gradients were produced for SGRS. "
                "Confirm eager attention and that model parameters/attention "
                "operations participate in autograd."
            )

        score = torch.stack(valid_scores).mean()

    model.zero_grad(set_to_none=True)
    return float(score.detach().cpu().item())


def select_sgrs_token(
    model,
    tokenizer,
    base_inputs: Dict[str, torch.Tensor],
    seq: torch.Tensor,
    attention_mask: torch.Tensor,
    logits: torch.Tensor,
    prompt_len: int,
    accepted_saliencies: List[float],
    config: SaliencyConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, Dict]:
    """Paper-style Top-K -> saliency rejection -> best-saliency fallback."""
    layers = get_language_layers(model)
    target_layers = parse_layer_spec(config.target_layers, len(layers))

    probs = F.softmax(logits.float(), dim=-1).squeeze(0)
    k = min(int(config.top_k), int(probs.numel()))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    top_probs = top_probs / top_probs.sum().clamp_min(1e-12)
    candidates = [int(x) for x in top_ids.tolist()]
    prob_map = {cid: float(p) for cid, p in zip(candidates, top_probs.tolist())}

    if accepted_saliencies:
        if config.history_window > 0:
            hist = accepted_saliencies[-int(config.history_window):]
        else:
            hist = accepted_saliencies
        tau = float(config.alpha) * (sum(hist) / max(len(hist), 1))
    else:
        tau = 0.0

    remaining = list(candidates)
    scores: Dict[int, float] = {}
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
        score = compute_candidate_saliency(
            model=model,
            base_inputs=base_inputs,
            seq=seq,
            attention_mask=attention_mask,
            candidate_id=cid,
            prompt_len=prompt_len,
            target_layers=target_layers,
            query_mode=config.query_mode,
        )
        scores[cid] = score
        if score >= tau:
            chosen = cid
            break
        rejected.append(cid)
        remaining.pop(pick)

    if chosen is None:
        fallback = True
        # Paper/public-code fallback: evaluate missing Top-K candidates and take
        # the candidate with maximum saliency.
        for cid in candidates:
            if cid not in scores:
                scores[cid] = compute_candidate_saliency(
                    model=model,
                    base_inputs=base_inputs,
                    seq=seq,
                    attention_mask=attention_mask,
                    candidate_id=cid,
                    prompt_len=prompt_len,
                    target_layers=target_layers,
                    query_mode=config.query_mode,
                )
        chosen = max(candidates, key=lambda c: scores[c])

    chosen_sal = float(scores.get(chosen, 1.0))
    accepted_saliencies.append(chosen_sal)
    meta = {
        "tau": tau,
        "chosen_id": int(chosen),
        "chosen_token": tokenizer.decode([int(chosen)], skip_special_tokens=False).replace("\n", "\\n"),
        "chosen_saliency": chosen_sal,
        "fallback": fallback,
        "rejected_ids": rejected,
        "candidates": [
            {
                "id": cid,
                "token": tokenizer.decode([cid], skip_special_tokens=False).replace("\n", "\\n"),
                "topk_prob": prob_map[cid],
                "saliency": scores.get(cid),
            }
            for cid in candidates
        ],
    }
    return int(chosen), meta
