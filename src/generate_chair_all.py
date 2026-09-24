from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Set

import torch
from tqdm import tqdm

import generate_chair as mcot
from ott_qwen import OTTConfig, QwenOTTController
from saliency_qwen import (
    LocoREConfig,
    LocoREController,
    SaliencyConfig,
    select_sgrs_token,
)


def parse_methods(text: str) -> Set[str]:
    aliases = {"saliency": {"sgrs", "locore"}, "all": {"mcot", "ott", "sgrs", "locore"}}
    out: Set[str] = set()
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key in aliases:
            out |= aliases[key]
        else:
            out.add(key)
    allowed = {"mcot", "ott", "sgrs", "locore"}
    bad = out - allowed
    if bad:
        raise ValueError(f"Unknown methods: {sorted(bad)}. Allowed: {sorted(allowed)}")
    return out


def full_forward(model, base_inputs, input_ids, attention_mask):
    return model(
        **mcot.model_inputs_for_step(base_inputs, input_ids, attention_mask),
        use_cache=False,
        return_dict=True,
    )


def generate_one(
    model,
    processor,
    base_inputs: Dict[str, torch.Tensor],
    methods: Set[str],
    max_new_tokens: int,
    activation_alpha: float,
    activation_info_layer: int,
    activation_threshold: float,
    ott: QwenOTTController | None,
    locore: LocoREController | None,
    sgrs_cfg: SaliencyConfig,
    seed: int,
):
    input_ids = base_inputs["input_ids"].clone()
    attention_mask = base_inputs["attention_mask"].clone()
    prompt_len = int(input_ids.shape[-1])
    eos_ids = mcot.get_eos_token_ids(processor.tokenizer)

    generated_ids: List[int] = []
    mcot_trace: List[dict] = []
    saliency_trace: List[dict] = []
    accepted_saliencies: List[float] = []

    rng = torch.Generator(device=input_ids.device)
    rng.manual_seed(int(seed))

    if locore is not None:
        locore.set_prompt_len(prompt_len)
        locore.enable()

    if ott is not None:
        ott.prepare_reference(base_inputs)
        ott.enable()

    try:
        if "mcot" in methods:
            activation_context = mcot.compute_activation_context(
                model=model,
                base_inputs=base_inputs,
                activation_info_layer=activation_info_layer,
            )
            step_logits = activation_context.baseline_logits
        else:
            activation_context = None
            with torch.no_grad():
                step_logits = model(
                    **base_inputs,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]

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
                greedy_id = int(step_logits.argmax(dim=-1).item())
                gate_meta = {
                    "current_token_id": greedy_id,
                    "current_token_entropy": None,
                    "gate_applied": False,
                }

            if "sgrs" in methods:
                next_id, sal_meta = select_sgrs_token(
                    model=model,
                    tokenizer=processor.tokenizer,
                    base_inputs=base_inputs,
                    seq=input_ids,
                    attention_mask=attention_mask,
                    logits=adjusted_logits,
                    prompt_len=prompt_len,
                    accepted_saliencies=accepted_saliencies,
                    config=sgrs_cfg,
                    generator=rng,
                )
                sal_meta["step"] = step_idx
                saliency_trace.append(sal_meta)
            else:
                next_id = int(adjusted_logits.argmax(dim=-1).item())

            generated_ids.append(next_id)
            mcot_trace.append(
                {
                    "step": step_idx,
                    "current_token_id": gate_meta["current_token_id"],
                    "current_token": mcot.decode_token(processor.tokenizer, gate_meta["current_token_id"]),
                    "current_token_entropy": gate_meta["current_token_entropy"],
                    "gate_applied": gate_meta["gate_applied"],
                    "selected_token_id": next_id,
                    "selected_token": mcot.decode_token(processor.tokenizer, next_id),
                }
            )

            if next_id in eos_ids:
                break

            next_token = torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            next_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            with torch.no_grad():
                step_logits = full_forward(model, base_inputs, input_ids, attention_mask).logits[:, -1, :]

            if locore is not None:
                locore.validate_active(len(generated_ids))

        if generated_ids:
            generated_tensor = torch.tensor([generated_ids], device=input_ids.device)
            text = processor.batch_decode(
                generated_tensor,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            text = ""

        return text, mcot_trace, saliency_trace
    finally:
        if ott is not None:
            ott.disable()
        if locore is not None:
            locore.disable()


def main():
    p = argparse.ArgumentParser(description="MCoT + OTT + LVLMs-Saliency Qwen2.5-VL integration")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--methods", default="mcot,ott,sgrs,locore")
    p.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--chunk_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=1994)
    p.add_argument("--shuffle", action="store_true", help="Shuffle input records deterministically with --seed")

    # MCoT defaults from the public MCoT repository usage.
    p.add_argument("--activation_alpha", type=float, default=0.75)
    p.add_argument("--activation_info_layer", type=int, default=-1)
    p.add_argument("--activation_threshold", type=float, default=0.5)

    # OTT defaults from the released OTT run setup used in the previous port.
    p.add_argument("--crc_lambda", type=float, default=0.1111)
    p.add_argument("--svc_ratio", type=float, default=0.06)
    p.add_argument("--max_visual_tokens", type=int, default=128)

    # LVLMs-Saliency SGRS defaults from public code/paper.
    p.add_argument("--sgrs_top_k", type=int, default=5)
    p.add_argument("--sgrs_max_resample", type=int, default=3)
    p.add_argument("--sgrs_alpha", type=float, default=0.6)
    p.add_argument("--sgrs_history_window", type=int, default=5)
    p.add_argument("--sgrs_target_layers", default="6,7,8")
    p.add_argument("--saliency_query_mode", choices=["predictor", "repo"], default="predictor")

    # Qwen2-VL paper ablation reports beta=0.20; window is configurable.
    p.add_argument("--locore_beta", type=float, default=0.20)
    p.add_argument("--locore_window", type=int, default=5)
    p.add_argument("--locore_layers", default="all")
    p.add_argument("--locore_non_strict", action="store_true")

    args = p.parse_args()
    methods = parse_methods(args.methods)

    if ("sgrs" in methods or "locore" in methods) and args.attn_implementation != "eager":
        raise ValueError("SGRS/LocoRE require --attn_implementation eager in this portable Qwen2.5-VL port")

    model, processor = mcot.load_model_and_processor(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        device=args.device,
    )

    ott = None
    if "ott" in methods:
        ott = QwenOTTController(
            model,
            OTTConfig(
                use_crc=True,
                use_svc=True,
                crc_lambda=args.crc_lambda,
                svc_ratio=args.svc_ratio,
                max_visual_tokens=args.max_visual_tokens,
            ),
        )
        ott.install()

    locore = None
    if "locore" in methods:
        locore = LocoREController(
            model,
            LocoREConfig(
                beta=args.locore_beta,
                window=args.locore_window,
                layers=args.locore_layers,
                strict_mask=not args.locore_non_strict,
            ),
        )
        locore.install()

    sgrs_cfg = SaliencyConfig(
        top_k=args.sgrs_top_k,
        max_resample=args.sgrs_max_resample,
        alpha=args.sgrs_alpha,
        history_window=args.sgrs_history_window,
        target_layers=args.sgrs_target_layers,
        query_mode=args.saliency_query_mode,
    )

    try:
        records = mcot.load_records(args.input)
        if args.shuffle:
            rng = random.Random(args.seed)
            records = list(records)
            rng.shuffle(records)
        records = mcot.select_chunk(
            records=records,
            num_chunks=args.num_chunks,
            chunk_index=args.chunk_index,
            num_samples=args.num_samples if args.num_samples > 0 else None,
        )
        if not records:
            raise ValueError("No records selected")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results: List[dict] = []
        bar = tqdm(records, total=len(records), desc=" + ".join(sorted(methods)) or "baseline")

        for sample_idx, record in enumerate(bar):
            image_id = record.get("image_id")
            if not image_id:
                raise KeyError("each record must contain image_id")
            question = record.get("instruction", "")
            image_path = mcot.resolve_image_path(args.image_root, image_id)
            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)

            base_inputs = mcot.prepare_inputs(
                processor=processor,
                image_path=image_path,
                question=question,
                device=str(model.device),
            )
            text, gate_trace, sal_trace = generate_one(
                model=model,
                processor=processor,
                base_inputs=base_inputs,
                methods=methods,
                max_new_tokens=args.max_new_tokens,
                activation_alpha=args.activation_alpha,
                activation_info_layer=args.activation_info_layer,
                activation_threshold=args.activation_threshold,
                ott=ott,
                locore=locore,
                sgrs_cfg=sgrs_cfg,
                seed=args.seed + sample_idx,
            )

            out = dict(record)
            out["image_src"] = image_path
            out["question"] = question
            out["model_answer"] = text
            out["methods"] = sorted(methods)
            out["activation_gate_trace"] = gate_trace
            out["saliency_trace"] = sal_trace
            results.append(out)

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            bar.set_postfix_str(str(image_id))

        print(f"Saved {len(results)} records to {args.output}")
        if locore is not None:
            print(f"LocoRE mask applications: {locore.applied_calls}")
            print(f"LocoRE skipped-mask calls: {locore.skipped_mask_calls}")
    finally:
        if ott is not None:
            ott.remove()
        if locore is not None:
            locore.remove()


if __name__ == "__main__":
    main()
