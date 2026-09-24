from __future__ import annotations

import argparse
import json
import os
import torch
from tqdm import tqdm

import generate_chair as mcot
from ott_qwen import OTTConfig, QwenOTTController


def greedy_generate_combined(
    model,
    processor,
    base_inputs,
    max_new_tokens,
    activation_alpha,
    activation_info_layer,
    activation_threshold,
    ott: QwenOTTController,
):
    # Reference extraction uses the untouched Qwen model.
    ott.prepare_reference(base_inputs)
    ott.enable()

    current_input_ids = base_inputs["input_ids"].clone()
    current_attention_mask = base_inputs["attention_mask"].clone()
    eos_token_ids = mcot.get_eos_token_ids(processor.tokenizer)

    generated_token_ids = []
    gate_trace = []

    try:
        # Compute MCoT activation context after OTT is enabled so the entropy
        # gate and OTT operate on the same effective model.
        if activation_alpha is not None:
            activation_context = mcot.compute_activation_context(
                model=model,
                base_inputs=base_inputs,
                activation_info_layer=activation_info_layer,
            )
            step_logits = activation_context.baseline_logits
        else:
            activation_context = None
            with torch.no_grad():
                outputs = model(
                    **base_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            step_logits = outputs.logits[:, -1, :]

        for step_idx in range(max_new_tokens):
            if activation_context is not None:
                adjusted_logits, gate_meta = mcot.maybe_apply_activation(
                    step_logits=step_logits,
                    entropy_vec=activation_context.entropy_vec,
                    activation_alpha=activation_alpha,
                    activation_threshold=activation_threshold,
                )
            else:
                adjusted_logits = step_logits
                current_token_id = int(step_logits.argmax(dim=-1).item())
                gate_meta = {
                    "current_token_id": current_token_id,
                    "current_token_entropy": None,
                    "gate_applied": False,
                }

            next_token_id = int(adjusted_logits.argmax(dim=-1).item())
            generated_token_ids.append(next_token_id)

            gate_trace.append(
                {
                    "step": step_idx,
                    "current_token_id": gate_meta["current_token_id"],
                    "current_token": mcot.decode_token(
                        processor.tokenizer, gate_meta["current_token_id"]
                    ),
                    "current_token_entropy": gate_meta["current_token_entropy"],
                    "mcot_gate_applied": gate_meta["gate_applied"],
                    "selected_token_id": next_token_id,
                    "selected_token": mcot.decode_token(
                        processor.tokenizer, next_token_id
                    ),
                }
            )

            if next_token_id in eos_token_ids:
                break

            next_token = torch.tensor(
                [[next_token_id]],
                device=current_input_ids.device,
                dtype=current_input_ids.dtype,
            )
            current_input_ids = torch.cat(
                [current_input_ids, next_token], dim=-1
            )

            next_mask = torch.ones(
                (1, 1),
                device=current_attention_mask.device,
                dtype=current_attention_mask.dtype,
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, next_mask], dim=-1
            )

            with torch.no_grad():
                outputs = model(
                    **mcot.model_inputs_for_step(
                        base_inputs,
                        current_input_ids,
                        current_attention_mask,
                    ),
                    use_cache=False,
                    return_dict=True,
                )
            step_logits = outputs.logits[:, -1, :]

        if generated_token_ids:
            generated_tensor = torch.tensor(
                [generated_token_ids],
                device=current_input_ids.device,
            )
            text = processor.batch_decode(
                generated_tensor,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            text = ""

        return text, gate_trace
    finally:
        ott.disable()


def process_records(
    model,
    processor,
    records,
    image_root,
    output_path,
    max_new_tokens,
    activation_alpha,
    activation_info_layer,
    activation_threshold,
    ott,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = []

    progress = tqdm(records, total=len(records), desc="MCoT + OTT")
    for record in progress:
        image_id = record.get("image_id")
        if not image_id:
            raise KeyError("each record must contain image_id")

        question = record.get("instruction", "")
        image_path = mcot.resolve_image_path(image_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        base_inputs = mcot.prepare_inputs(
            processor=processor,
            image_path=image_path,
            question=question,
            device=str(model.device),
        )

        generated_text, trace = greedy_generate_combined(
            model=model,
            processor=processor,
            base_inputs=base_inputs,
            max_new_tokens=max_new_tokens,
            activation_alpha=activation_alpha,
            activation_info_layer=activation_info_layer,
            activation_threshold=activation_threshold,
            ott=ott,
        )

        out = dict(record)
        out["image_src"] = image_path
        out["question"] = question
        out["model_answer"] = generated_text
        out["activation_gate_trace"] = trace
        out["combined_method"] = "MCoT+OTT"
        results.append(out)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        progress.set_postfix_str(str(image_id))

    return results


def main():
    p = argparse.ArgumentParser()

    # Same host-side inputs as MCoT-hallucination.
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--chunk_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # MCoT.
    p.add_argument("--activation_alpha", type=float, default=0.75)
    p.add_argument("--activation_info_layer", type=int, default=-1)
    p.add_argument("--activation_threshold", type=float, default=0.5)
    p.add_argument("--disable_activation_gate", action="store_true")

    # OTT defaults follow OTT's released run_pope.sh.
    p.add_argument("--crc", action="store_true")
    p.add_argument("--svc", action="store_true")
    p.add_argument("--crc_lambda", type=float, default=0.1111)
    p.add_argument("--svc_ratio", type=float, default=0.06)
    p.add_argument("--max_visual_tokens", type=int, default=128)

    args = p.parse_args()

    model, processor = mcot.load_model_and_processor(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        device=args.device,
    )

    ott = QwenOTTController(
        model,
        OTTConfig(
            use_crc=args.crc,
            use_svc=args.svc,
            crc_lambda=args.crc_lambda,
            svc_ratio=args.svc_ratio,
            max_visual_tokens=args.max_visual_tokens,
        ),
    )
    ott.install()

    try:
        num_samples = args.num_samples if args.num_samples > 0 else None
        records = mcot.randomize_records(mcot.load_records(args.input))
        records = mcot.select_chunk(
            records=records,
            num_chunks=args.num_chunks,
            chunk_index=args.chunk_index,
            num_samples=num_samples,
        )
        if not records:
            raise ValueError("no records selected")

        threshold = (
            None if args.disable_activation_gate
            else args.activation_threshold
        )

        process_records(
            model=model,
            processor=processor,
            records=records,
            image_root=args.image_root,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            activation_alpha=args.activation_alpha,
            activation_info_layer=args.activation_info_layer,
            activation_threshold=threshold,
            ott=ott,
        )
    finally:
        ott.remove()


if __name__ == "__main__":
    main()
