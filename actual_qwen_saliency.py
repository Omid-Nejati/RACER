#!/usr/bin/env python3
"""
Actual causal token-to-token saliency for Qwen2.5-VL.

This reproduces the core diagnostic idea from:
"Hallucination Begins Where Saliency Drops" (ICLR 2026)

For a chosen generated target token y_t:
    1) Build the prefix that ends immediately before y_t.
    2) Run a gradient-enabled forward pass.
    3) Compute CE loss for y_t from the last-position logits.
    4) For each layer/head:
           S = tril(abs(A * dL/dA))
    5) Sum heads, L2-normalize each layer, average selected layers.
    6) Optionally remove visual placeholder tokens only for visualization.
    7) Save .npy + PNG + PDF.

Example:
python actual_qwen_saliency.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --image /path/to/image.jpg \
  --question "Describe the image in detail." \
  --target-step 12 \
  --max-new-tokens 40 \
  --text-only \
  --output-prefix sample12
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def clean_token(tok: str) -> str:
    replacements = {
        "Ġ": "▁",
        "Ċ": "↵",
        "ĉ": "⇥",
    }
    for a, b in replacements.items():
        tok = tok.replace(a, b)
    return tok


def l2_normalize_matrix(x: torch.Tensor, eps: float = 1e-12):
    return x / (torch.linalg.vector_norm(x) + eps)


def get_text_keep_indices(tokenizer, token_ids):
    """
    Remove Qwen visual placeholder/marker tokens only from the DISPLAY matrix.
    The actual saliency is still computed using the full multimodal sequence.
    """
    special_names = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    special_ids = set()
    for name in special_names:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != tokenizer.unk_token_id:
            special_ids.add(tid)

    keep = [i for i, tid in enumerate(token_ids) if tid not in special_ids]
    return keep


def plot_saliency(matrix, labels, target_text, out_png, out_pdf):
    n = matrix.shape[0]
    masked = np.ma.array(
        matrix,
        mask=np.triu(np.ones((n, n), dtype=bool), k=1),
    )

    cmap = plt.get_cmap("RdPu").copy()
    cmap.set_bad("#fffaf7")

    # Scale figure size to token count, but keep sane bounds.
    side = min(max(8.0, n * 0.22), 22.0)
    fig, ax = plt.subplots(figsize=(side, side))

    im = ax.imshow(
        masked,
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )

    # Show token labels only when the sequence is not excessively long.
    if n <= 100:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        step = max(1, n // 50)
        inds = np.arange(0, n, step)
        ax.set_xticks(inds)
        ax.set_yticks(inds)
        ax.set_xticklabels([labels[i] for i in inds], rotation=90, fontsize=5)
        ax.set_yticklabels([labels[i] for i in inds], fontsize=5)

    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    ax.set_title(f"Causal token-to-token saliency → target: {target_text!r}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized |attention × gradient|")

    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--target-step", type=int, default=-1,
                        help="0-based index inside generated tokens. -1 = last generated token.")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=-1,
                        help="Exclusive. -1 means all layers.")
    parser.add_argument("--text-only", action="store_true",
                        help="Remove visual marker/pad tokens only from the plotted matrix.")
    parser.add_argument("--output-prefix", default="actual_saliency")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(f"Loading model: {args.model}")

    # IMPORTANT:
    # We force eager attention because FlashAttention/SDPA often does not expose
    # the full differentiable attention probability matrices required here.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
    ).to(args.device)

    model.eval()

    # Reducing max_pixels can drastically reduce attention-memory use.
    processor = AutoProcessor.from_pretrained(
        args.model,
        max_pixels=512 * 28 * 28,
    )
    tokenizer = processor.tokenizer

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}

    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # -----------------------------------------------------------
    # PASS 1: normal generation. No gradients needed here.
    # -----------------------------------------------------------
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = generated[:, prompt_len:]
    gen_list = generated_ids[0].tolist()

    if len(gen_list) == 0:
        raise RuntimeError("The model generated zero tokens.")

    print("\nGenerated answer:")
    print(tokenizer.decode(gen_list, skip_special_tokens=True))

    print("\nGenerated tokens:")
    for i, tid in enumerate(gen_list):
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"{i:4d}  id={tid:6d}  token={tok!r}")

    target_step = args.target_step
    if target_step < 0:
        target_step = len(gen_list) - 1
    if not 0 <= target_step < len(gen_list):
        raise ValueError(
            f"--target-step must be in [0, {len(gen_list)-1}], got {target_step}"
        )

    target_id = generated_ids[:, target_step]
    target_token = tokenizer.convert_ids_to_tokens(int(target_id.item()))
    target_text = tokenizer.decode([int(target_id.item())])

    # Prefix ends immediately BEFORE the selected generated target token.
    if target_step == 0:
        prefix_ids = prompt_ids
    else:
        prefix_ids = torch.cat(
            [prompt_ids, generated_ids[:, :target_step]],
            dim=1,
        )

    prefix_len = prefix_ids.shape[1]
    attention_mask = torch.ones_like(prefix_ids)

    print("\nSelected target:")
    print(f"  target_step : {target_step}")
    print(f"  token id    : {int(target_id.item())}")
    print(f"  token       : {target_token!r}")
    print(f"  decoded     : {target_text!r}")
    print(f"  prefix len  : {prefix_len}")

    # -----------------------------------------------------------
    # PASS 2: gradient-enabled teacher-forced target prediction.
    #
    # logits[:, -1] predicts the NEXT token after prefix_ids,
    # which is exactly target_id.
    # -----------------------------------------------------------
    forward_kwargs = {
        "input_ids": prefix_ids,
        "attention_mask": attention_mask,
        "output_attentions": True,
        "use_cache": False,
        "return_dict": True,
    }

    # Reuse the image tensors from the original processor output.
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in inputs:
            forward_kwargs[key] = inputs[key]

    # Do NOT put this inside torch.no_grad().
    with torch.enable_grad():
        outputs = model(**forward_kwargs)

        if outputs.attentions is None:
            raise RuntimeError(
                "The model returned no attentions. "
                "Make sure attn_implementation='eager' and output_attentions=True."
            )

        usable_attentions = [a for a in outputs.attentions if a is not None]
        if not usable_attentions:
            raise RuntimeError("No usable attention tensors were returned.")

        if not usable_attentions[0].requires_grad:
            raise RuntimeError(
                "Returned attention tensors do not require gradients. "
                "Your Transformers/model implementation is not exposing differentiable "
                "attention probabilities. In that case use the modified Qwen attention "
                "implementation from the LVLMs-Saliency repository."
            )

        logits = outputs.logits[:, -1, :].float()
        loss = F.cross_entropy(logits, target_id)

        # Get dL/dA directly without filling parameter .grad tensors.
        grads = torch.autograd.grad(
            loss,
            usable_attentions,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        n_layers = len(usable_attentions)
        layer_end = n_layers if args.layer_end < 0 else min(args.layer_end, n_layers)
        layer_start = max(0, args.layer_start)

        if layer_start >= layer_end:
            raise ValueError(
                f"Invalid layer range [{layer_start}, {layer_end}) for {n_layers} layers."
            )

        selected_layer_maps = []

        for layer_idx in range(layer_start, layer_end):
            attn = usable_attentions[layer_idx]
            grad = grads[layer_idx]

            if grad is None:
                print(f"[WARN] Layer {layer_idx}: gradient is None; skipping.")
                continue

            # attn/grad: [batch, heads, query, key]
            sal = torch.abs(attn * grad)
            sal = torch.tril(sal)

            # Paper-style head aggregation:
            # sum over heads, then L2-normalize each layer matrix.
            head_sum = sal.sum(dim=1)[0].float()  # [Q, K]
            layer_map = l2_normalize_matrix(head_sum)
            selected_layer_maps.append(layer_map)

        if not selected_layer_maps:
            raise RuntimeError("No layer saliency maps were produced.")

        # Average selected layers.
        saliency = torch.stack(selected_layer_maps, dim=0).mean(dim=0)

    saliency_np = saliency.detach().cpu().numpy()

    prefix_token_ids = prefix_ids[0].detach().cpu().tolist()
    labels = [
        clean_token(tokenizer.convert_ids_to_tokens(tid))
        for tid in prefix_token_ids
    ]

    if args.text_only:
        keep = get_text_keep_indices(tokenizer, prefix_token_ids)
        saliency_np = saliency_np[np.ix_(keep, keep)]
        labels = [labels[i] for i in keep]

    # Display normalization only. Keep raw normalized-layer average separately too.
    max_val = float(saliency_np.max())
    display_saliency = saliency_np / max_val if max_val > 0 else saliency_np

    npy_path = f"{args.output_prefix}.npy"
    png_path = f"{args.output_prefix}.png"
    pdf_path = f"{args.output_prefix}.pdf"
    tokens_path = f"{args.output_prefix}_tokens.txt"

    np.save(npy_path, saliency_np)

    with open(tokens_path, "w", encoding="utf-8") as f:
        for i, tok in enumerate(labels):
            f.write(f"{i}\t{tok}\n")

    plot_saliency(
        display_saliency,
        labels,
        target_text,
        png_path,
        pdf_path,
    )

    print("\nSaved:")
    print(f"  {npy_path}")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {tokens_path}")
    print(f"\nTarget CE loss: {float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
