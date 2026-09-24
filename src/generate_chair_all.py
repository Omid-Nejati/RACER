"""
MCoT + OTT + LVLMs-Saliency on Qwen2.5-VL / GRIT, with the MedicalHall novelties.

  --preset port   reproduces the previous plain-stacking integration exactly
  --preset ours   enables every novelty (AVEG, TRACE, SGRS+, LocoRE+, risk router)

Every novelty also has its own flag, so ablation rows are "ours minus X".
See NOVELTY.md for the list and the paper-ready description.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Set

import torch
from tqdm import tqdm

import generate_chair as mcot
from mcot_plus import AVEG, AVEGConfig, PhaseTracker
from ott_qwen import OTTConfig, QwenOTTController, _get_image_token_id
from risk_router import RiskRouter, RouterConfig
from saliency_qwen import (
    LocoREConfig,
    LocoREController,
    SaliencyConfig,
    SaliencyState,
    adaptive_locore_beta,
    attention_visual_priority,
    select_sgrs_token,
)

PRESETS: Dict[str, Dict[str, object]] = {
    "port": dict(
        aveg_info_layers="-1", aveg_candidate_k=1, aveg_soft_gate=False, aveg_gate_temperature=0.05,
        aveg_think_scale=1.0, aveg_answer_scale=1.0,
        ott_crc_layers="all", ott_svc_layers="all", ott_decode_only=False, ott_visual_select="uniform",
        ott_svc_ref_space="residual", ott_relevance_gate=False,
        sgrs_visual_lambda=0.0, sgrs_entropy_kappa=0.0, sgrs_fallback_eta="none", sgrs_exclude_neutral=False,
        locore_adaptive=False, locore_beta_max=0.60, locore_visual_beta=0.0, locore_visual_anchor_k=32,
        router="always", risk_threshold=0.5, risk_w_entropy=0.5, risk_w_saliency=0.3, risk_w_conf=0.2,
        risk_confident_skip=1.01, risk_retrace=True,
    ),
    "ours": dict(
        aveg_info_layers="-1,-6,-12", aveg_candidate_k=5, aveg_soft_gate=True, aveg_gate_temperature=0.05,
        aveg_think_scale=0.5, aveg_answer_scale=1.0,
        ott_crc_layers="auto", ott_svc_layers="auto", ott_decode_only=True, ott_visual_select="attention",
        ott_svc_ref_space="mlp", ott_relevance_gate=True,
        sgrs_visual_lambda=0.3, sgrs_entropy_kappa=0.5, sgrs_fallback_eta="1.0", sgrs_exclude_neutral=True,
        locore_adaptive=True, locore_beta_max=0.60, locore_visual_beta=0.10, locore_visual_anchor_k=32,
        router="risk", risk_threshold=0.5, risk_w_entropy=0.5, risk_w_saliency=0.3, risk_w_conf=0.2,
        risk_confident_skip=0.95, risk_retrace=True,
    ),
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_methods(text: str) -> Set[str]:
    aliases = {"saliency": {"sgrs", "locore"}, "all": {"mcot", "ott", "sgrs", "locore"}}
    out: Set[str] = set()
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        out |= aliases.get(key, {key})
    bad = out - {"mcot", "ott", "sgrs", "locore"}
    if bad:
        raise ValueError(f"Unknown methods: {sorted(bad)}")
    return out


def full_forward(model, base_inputs, input_ids, attention_mask):
    return model(**mcot.model_inputs_for_step(base_inputs, input_ids, attention_mask),
                 use_cache=False, return_dict=True)


def generate_one(
    model,
    processor,
    base_inputs: Dict[str, torch.Tensor],
    methods: Set[str],
    max_new_tokens: int,
    aveg: AVEG | None,
    ott: QwenOTTController | None,
    locore: LocoREController | None,
    sgrs_cfg: SaliencyConfig,
    router: RiskRouter,
    seed: int,
):
    tokenizer = processor.tokenizer
    input_ids = base_inputs["input_ids"].clone()
    attention_mask = base_inputs["attention_mask"].clone()
    prompt_len = int(input_ids.shape[-1])
    eos_ids = mcot.get_eos_token_ids(tokenizer)
    image_token_id = _get_image_token_id(model)
    visual_positions = torch.nonzero(input_ids[0].eq(image_token_id), as_tuple=False).flatten()

    generated_ids: List[int] = []
    gate_trace: List[dict] = []
    saliency_trace: List[dict] = []
    state = SaliencyState()
    phase = PhaseTracker(tokenizer)
    cost = {"forwards": 0, "retrace_steps": 0, "sgrs_steps": 0, "steps": 0}
    t0 = time.perf_counter()

    rng = torch.Generator(device=input_ids.device)
    rng.manual_seed(int(seed))

    if locore is not None:
        locore.set_prompt_len(prompt_len)
        locore.enable()
    if ott is not None:
        ott.prepare_reference(base_inputs)
        cost["forwards"] += 1
        ott.enable()
    if locore is not None and locore.cfg.visual_beta > 0:
        if ott is not None and ott.visual_priority_ranked is not None:
            anchors = ott.visual_priority_ranked
        else:
            with torch.no_grad():
                anchors = attention_visual_priority(model, base_inputs, image_token_id)
            cost["forwards"] += 1
        locore.set_visual_anchors(anchors)

    try:
        if aveg is not None:
            step_logits = aveg.prepare(base_inputs)
        else:
            with torch.no_grad():
                step_logits = model(**base_inputs, use_cache=False, return_dict=True).logits[:, -1, :]
        cost["forwards"] += 1

        for step_idx in range(max_new_tokens):
            cost["steps"] += 1
            row = int(input_ids.shape[-1]) - 1          # row that predicts the next token
            cur_phase = phase.update(generated_ids)

            def adjust(logits):
                if aveg is not None:
                    return aveg.apply(logits, cur_phase)
                g = int(logits.argmax(dim=-1).item())
                return logits, {"current_token_id": g, "current_token_entropy": None,
                                "gate_applied": False, "gate_value": None}

            adjusted_logits, gate_meta = adjust(step_logits)
            p_max = float(torch.softmax(step_logits.float(), dim=-1).max().item())
            risk = router.risk(gate_meta.get("gate_value"),
                               state.last_relative if "sgrs" in methods else None, p_max)
            decision = router.decide(risk, p_max)

            retraced = False
            if decision["retrace"] and ott is not None and ott.svc_layer_set:
                ott.set_position_gate(row, 1.0)                     # N2.f
                with torch.no_grad():
                    step_logits = full_forward(model, base_inputs, input_ids, attention_mask).logits[:, -1, :]
                cost["forwards"] += 1
                cost["retrace_steps"] += 1
                retraced = True
                adjusted_logits, gate_meta = adjust(step_logits)

            used_sgrs = "sgrs" in methods and decision["sgrs"]
            if used_sgrs:
                next_id, sal_meta = select_sgrs_token(
                    model=model, tokenizer=tokenizer, base_inputs=base_inputs,
                    seq=input_ids, attention_mask=attention_mask, logits=adjusted_logits,
                    prompt_len=prompt_len, state=state, config=sgrs_cfg, generator=rng,
                    visual_positions=visual_positions,
                    entropy_vec=aveg.entropy_vec if aveg is not None else None,
                )
                sal_meta["step"] = step_idx
                saliency_trace.append(sal_meta)
                cost["sgrs_steps"] += 1
            else:
                next_id = int(adjusted_logits.argmax(dim=-1).item())

            generated_ids.append(next_id)
            gate_trace.append({
                "step": step_idx,
                "current_token_id": gate_meta["current_token_id"],
                "current_token": mcot.decode_token(tokenizer, gate_meta["current_token_id"]),
                "current_token_entropy": gate_meta["current_token_entropy"],
                "expected_entropy": gate_meta.get("expected_entropy"),
                "gate_value": gate_meta.get("gate_value"),
                "gate_applied": gate_meta["gate_applied"],
                "phase": cur_phase,
                "p_max": p_max,
                "risk": risk,
                "retraced": retraced,
                "sgrs": used_sgrs,
                "selected_token_id": next_id,
                "selected_token": mcot.decode_token(tokenizer, next_id),
            })

            if next_id in eos_ids:
                break

            next_token = torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device,
                                            dtype=attention_mask.dtype)], dim=-1)

            # N4.a: after a risky, saliency-checked step, reinforce the new row harder.
            if locore is not None and locore.cfg.adaptive and used_sgrs:
                locore.set_row_beta(int(input_ids.shape[-1]) - 1,
                                    adaptive_locore_beta(locore.cfg, state, sgrs_cfg.history_window))

            with torch.no_grad():
                step_logits = full_forward(model, base_inputs, input_ids, attention_mask).logits[:, -1, :]
            cost["forwards"] += 1

            if locore is not None:
                locore.validate_active(len(generated_ids))

        if generated_ids:
            text = processor.batch_decode(torch.tensor([generated_ids], device=input_ids.device),
                                          skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            text = ""
        cost["seconds"] = round(time.perf_counter() - t0, 3)
        return text, gate_trace, saliency_trace, cost, generated_ids
    finally:
        if ott is not None:
            ott.disable()
        if locore is not None:
            locore.disable()


def build_parser():
    p = argparse.ArgumentParser(description="MedicalHall: MCoT + OTT + LVLMs-Saliency (Qwen2.5-VL) with novelties")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--methods", default="mcot,ott,sgrs,locore")
    p.add_argument("--preset", choices=sorted(PRESETS), default="port")
    p.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--chunk_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=1994)
    p.add_argument("--shuffle", action="store_true")

    # MCoT (original)
    p.add_argument("--activation_alpha", type=float, default=0.75)
    p.add_argument("--activation_info_layer", type=int, default=-1, help="used when --aveg_info_layers is unset in port")
    p.add_argument("--activation_threshold", type=float, default=0.5)
    p.add_argument("--disable_activation_gate", type=str2bool, default=False)
    # N1 AVEG
    p.add_argument("--aveg_info_layers", default=None)
    p.add_argument("--aveg_candidate_k", type=int, default=None)
    p.add_argument("--aveg_soft_gate", type=str2bool, default=None)
    p.add_argument("--aveg_gate_temperature", type=float, default=None)
    p.add_argument("--aveg_think_scale", type=float, default=None)
    p.add_argument("--aveg_answer_scale", type=float, default=None)

    # OTT (original)
    p.add_argument("--crc_lambda", type=float, default=0.1111)
    p.add_argument("--svc_ratio", type=float, default=0.06)
    p.add_argument("--max_visual_tokens", type=int, default=128)
    # N2 TRACE
    p.add_argument("--ott_crc_layers", default=None)
    p.add_argument("--ott_svc_layers", default=None)
    p.add_argument("--ott_decode_only", type=str2bool, default=None)
    p.add_argument("--ott_visual_select", choices=["uniform", "attention"], default=None)
    p.add_argument("--ott_svc_ref_space", choices=["residual", "mlp"], default=None)
    p.add_argument("--ott_relevance_gate", type=str2bool, default=None)

    # SGRS (original)
    p.add_argument("--sgrs_top_k", type=int, default=5)
    p.add_argument("--sgrs_max_resample", type=int, default=3)
    p.add_argument("--sgrs_alpha", type=float, default=0.6)
    p.add_argument("--sgrs_history_window", type=int, default=5)
    p.add_argument("--sgrs_target_layers", default="6,7,8")
    p.add_argument("--saliency_query_mode", choices=["predictor", "repo"], default="predictor")
    # N3 SGRS+
    p.add_argument("--sgrs_visual_lambda", type=float, default=None)
    p.add_argument("--sgrs_entropy_kappa", type=float, default=None)
    p.add_argument("--sgrs_fallback_eta", default=None, help="float, or 'none' for pure max-saliency")
    p.add_argument("--sgrs_exclude_neutral", type=str2bool, default=None)

    # LocoRE (original)
    p.add_argument("--locore_beta", type=float, default=0.20)
    p.add_argument("--locore_window", type=int, default=5)
    p.add_argument("--locore_layers", default="all")
    p.add_argument("--locore_non_strict", action="store_true")
    # N4 LocoRE+
    p.add_argument("--locore_adaptive", type=str2bool, default=None)
    p.add_argument("--locore_beta_max", type=float, default=None)
    p.add_argument("--locore_visual_beta", type=float, default=None)
    p.add_argument("--locore_visual_anchor_k", type=int, default=None)

    # N5 risk router
    p.add_argument("--router", choices=["always", "risk"], default=None)
    p.add_argument("--risk_threshold", type=float, default=None)
    p.add_argument("--risk_w_entropy", type=float, default=None)
    p.add_argument("--risk_w_saliency", type=float, default=None)
    p.add_argument("--risk_w_conf", type=float, default=None)
    p.add_argument("--risk_confident_skip", type=float, default=None)
    p.add_argument("--risk_retrace", type=str2bool, default=None)
    return p


def resolve_args(args):
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.preset == "port" and args.aveg_info_layers == "-1":
        args.aveg_info_layers = str(args.activation_info_layer)
    eta = str(args.sgrs_fallback_eta).strip().lower()
    args.sgrs_fallback_eta = None if eta in {"none", ""} else float(eta)
    return args


def build_components(model, args, methods):
    aveg = None
    if "mcot" in methods:
        aveg = AVEG(model, AVEGConfig(
            alpha=args.activation_alpha, threshold=args.activation_threshold,
            info_layers=args.aveg_info_layers, candidate_k=args.aveg_candidate_k,
            soft_gate=args.aveg_soft_gate, gate_temperature=args.aveg_gate_temperature,
            think_alpha_scale=args.aveg_think_scale, answer_alpha_scale=args.aveg_answer_scale,
            disable_gate=args.disable_activation_gate,
        ))
    ott = None
    if "ott" in methods:
        ott = QwenOTTController(model, OTTConfig(
            use_crc=True, use_svc=True, crc_lambda=args.crc_lambda, svc_ratio=args.svc_ratio,
            max_visual_tokens=args.max_visual_tokens, crc_layers=args.ott_crc_layers,
            svc_layers=args.ott_svc_layers, decode_only=args.ott_decode_only,
            visual_select=args.ott_visual_select, svc_ref_space=args.ott_svc_ref_space,
            relevance_gate=args.ott_relevance_gate,
            default_svc_gate=0.0 if args.router == "risk" else 1.0,
        ))
        ott.install()
    locore = None
    if "locore" in methods:
        locore = LocoREController(model, LocoREConfig(
            beta=args.locore_beta, window=args.locore_window, layers=args.locore_layers,
            strict_mask=not args.locore_non_strict, adaptive=args.locore_adaptive,
            beta_max=args.locore_beta_max, visual_beta=args.locore_visual_beta,
            visual_anchor_k=args.locore_visual_anchor_k,
        ))
        locore.install()
    sgrs_cfg = SaliencyConfig(
        top_k=args.sgrs_top_k, max_resample=args.sgrs_max_resample, alpha=args.sgrs_alpha,
        history_window=args.sgrs_history_window, target_layers=args.sgrs_target_layers,
        query_mode=args.saliency_query_mode, visual_lambda=args.sgrs_visual_lambda,
        entropy_kappa=args.sgrs_entropy_kappa, entropy_gamma=args.activation_threshold,
        fallback_eta=args.sgrs_fallback_eta, exclude_neutral_history=args.sgrs_exclude_neutral,
    )
    router = RiskRouter(RouterConfig(
        mode=args.router, w_entropy=args.risk_w_entropy, w_saliency=args.risk_w_saliency,
        w_conf=args.risk_w_conf, risk_threshold=args.risk_threshold, sgrs_alpha=args.sgrs_alpha,
        confident_skip=args.risk_confident_skip, retrace=args.risk_retrace,
    ))
    return aveg, ott, locore, sgrs_cfg, router


def main():
    args = resolve_args(build_parser().parse_args())
    methods = parse_methods(args.methods)
    if ("sgrs" in methods or "locore" in methods or args.ott_visual_select == "attention") \
            and args.attn_implementation != "eager":
        raise ValueError("SGRS/LocoRE/attention-prioritised OTT require --attn_implementation eager")

    model, processor = mcot.load_model_and_processor(
        model_id=args.model_id, attn_implementation=args.attn_implementation, device=args.device)
    aveg, ott, locore, sgrs_cfg, router = build_components(model, args, methods)

    try:
        records = mcot.load_records(args.input)
        if args.shuffle:
            records = list(records)
            random.Random(args.seed).shuffle(records)
        records = mcot.select_chunk(records=records, num_chunks=args.num_chunks, chunk_index=args.chunk_index,
                                    num_samples=args.num_samples if args.num_samples > 0 else None)
        if not records:
            raise ValueError("No records selected")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results: List[dict] = []
        bar = tqdm(records, total=len(records), desc=f"[{args.preset}] " + (" + ".join(sorted(methods)) or "baseline"))
        for sample_idx, record in enumerate(bar):
            image_id = record.get("image_id")
            if not image_id:
                raise KeyError("each record must contain image_id")
            question = record.get("instruction", "")
            image_path = mcot.resolve_image_path(args.image_root, image_id)
            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)
            base_inputs = mcot.prepare_inputs(processor=processor, image_path=image_path,
                                              question=question, device=str(model.device))
            text, gate_trace, sal_trace, cost, _ = generate_one(
                model=model, processor=processor, base_inputs=base_inputs, methods=methods,
                max_new_tokens=args.max_new_tokens, aveg=aveg, ott=ott, locore=locore,
                sgrs_cfg=sgrs_cfg, router=router, seed=args.seed + sample_idx,
            )
            out = dict(record)
            out.update({
                "image_src": image_path, "question": question, "model_answer": text,
                "methods": sorted(methods), "preset": args.preset,
                "activation_gate_trace": gate_trace, "saliency_trace": sal_trace, "cost": cost,
            })
            results.append(out)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            bar.set_postfix_str(f"{image_id} fw={cost['forwards']} sgrs={cost['sgrs_steps']}")

        n = max(len(results), 1)
        summary = {k: sum(r["cost"][k] for r in results) / n for k in ("forwards", "retrace_steps", "sgrs_steps", "steps", "seconds")}
        print(f"Saved {len(results)} records to {args.output}")
        print("Mean cost per caption:", json.dumps(summary))
        with open(os.path.splitext(args.output)[0] + "_cost.json", "w") as f:
            json.dump({"preset": args.preset, "methods": sorted(methods), "mean_cost": summary, "args": vars(args)}, f, indent=2)
        if locore is not None:
            print(f"LocoRE mask applications: {locore.applied_calls}; skipped-mask calls: {locore.skipped_mask_calls}")
    finally:
        if ott is not None:
            ott.remove()
        if locore is not None:
            locore.remove()


if __name__ == "__main__":
    main()
