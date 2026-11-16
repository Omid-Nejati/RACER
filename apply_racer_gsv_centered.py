#!/usr/bin/env python3
"""Patch RACER's GSV with softmax-centered, sign-aware attention sensitivity.

Run at the repository root: python apply_racer_gsv_centered.py --dry-run
                               python apply_racer_gsv_centered.py

Use a new git branch. This tool patches only matching public-main anchors;
unknown revisions produce an error before *any* file is written.
The change is research code and requires model-level validation.
"""
import argparse
import ast
import difflib
from pathlib import Path


def change(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"Cannot locate exactly one {label}; found {count}. Source may have changed.")
    return src.replace(old, new, 1)


def saliency_patch(src: str) -> str:
    src = change(src,
'''    exclude_neutral_history: bool = False  # N3.d
    strict_grounding: bool = False''',
'''    exclude_neutral_history: bool = False  # N3.d
    signed_saliency: bool = False   # proposed: preserve port behavior; enabled for ours
    direction_min_ratio: float = 0.5  # require support > opposition
    strict_grounding: bool = False''', 'SaliencyConfig options')
    src = change(src,
'''    measure_first_visual: bool = False,
) -> Dict[str, object]:''',
'''    measure_first_visual: bool = False,
    signed_saliency: bool = False,
) -> Dict[str, object]:''', 'saliency signature')
    src = change(src,
'''    Gradient-aware attention saliency |A * dL/dA| (head-mean, row-L2-normalised)
    of one candidate.  Returns the text-history part (original method) and,
    if visual_positions is given, the visual-grounding part (N3.a).
''',
'''    Candidate-conditioned attention sensitivity (head-mean, row-L2-normalised).
    Legacy mode uses |A * dL/dA|. Signed mode separates supportive and opposing
    dL/dz where z are pre-softmax attention logits. Neither implies causality.
''', 'saliency function docstring')
    src = change(src,
'''        txt_scores: List[torch.Tensor] = []
        vis_scores: List[torch.Tensor] = []''',
'''        txt_scores: List[torch.Tensor] = []
        vis_scores: List[torch.Tensor] = []
        txt_opposing: List[torch.Tensor] = []
        vis_opposing: List[torch.Tensor] = []''', 'saliency accumulator')
    src = change(src,
'''            sal = (attn.float() * grad.float()).abs()
            tq, tk = sal.shape[-2], sal.shape[-1]
            tri = torch.tril(torch.ones((tq, tk), device=sal.device, dtype=sal.dtype))
            sal = sal * tri.view(1, 1, tq, tk)
            sal = sal.mean(dim=1)
            sal = sal / sal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)
            q_idx = tq - 2 if query_mode == "predictor" else tq - 1''',
'''            a, g = attn.float(), grad.float()
            tq, tk = a.shape[-2], a.shape[-1]
            tri = torch.tril(torch.ones((tq, tk), device=a.device,
                                        dtype=a.dtype)).view(1, 1, tq, tk)
            if signed_saliency:
                # Softmax-consistent derivative w.r.t pre-softmax attention logits:
                # dL/dz_j = A_j * (dL/dA_j - sum_k A_k * dL/dA_k)
                # Unlike A*g, this accounts for redistribution between keys.
                weighted = a * (g - (a * g).sum(dim=-1, keepdim=True))
                # Split BEFORE head pooling to avoid cancellation across heads.
                supp = (-weighted).clamp_min(0.0).mul(tri).mean(dim=1)
                opp = weighted.clamp_min(0.0).mul(tri).mean(dim=1)
                # Common denominator retains relative support/opposition strength.
                denom = (supp + opp).norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)
                sal, anti = supp / denom, opp / denom
            else:
                # EXACT original magnitude-only LVLMs-Saliency behavior.
                sal = (a * g).abs().mul(tri).mean(dim=1)
                sal = sal / sal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)
                anti = None
            q_idx = tq - 2 if query_mode == "predictor" else tq - 1''', 'signed saliency computation')
    src = change(src,
'''            else:
                txt_scores.append(sal[0, q_idx, hist_start:hist_end].mean())
            if visual_positions is not None and visual_positions.numel() > 0:
                vis_scores.append(sal[0, q_idx].index_select(0, visual_positions.to(sal.device)).mean())''',
'''            else:
                txt_scores.append(sal[0, q_idx, hist_start:hist_end].mean())
                if signed_saliency:
                    txt_opposing.append(anti[0, q_idx, hist_start:hist_end].mean())
            if visual_positions is not None and visual_positions.numel() > 0:
                pos = visual_positions.to(device=sal.device, dtype=torch.long)
                vis_scores.append(sal[0, q_idx].index_select(0, pos).mean())
                if signed_saliency:
                    vis_opposing.append(anti[0, q_idx].index_select(0, pos).mean())''', 'directional score collection')
    src = change(src,
'''        vis = float(torch.stack(vis_scores).mean().detach().cpu().item()) if vis_scores else None

    model.zero_grad(set_to_none=True)
    return {"txt": txt, "vis": vis, "neutral": neutral}''',
'''        vis = float(torch.stack(vis_scores).mean().detach().cpu().item()) if vis_scores else None
        txt_opp = (float(torch.stack(txt_opposing).mean().detach().cpu().item())
                   if txt_opposing else (0.0 if signed_saliency and neutral else None))
        vis_opp = (float(torch.stack(vis_opposing).mean().detach().cpu().item())
                   if vis_opposing else None)

    model.zero_grad(set_to_none=True)
    return {"txt": txt, "vis": vis, "txt_opp": txt_opp,
            "vis_opp": vis_opp, "neutral": neutral}''', 'directional return values')
    src = change(src,
'''def _support_pass(parts: Dict[str, object], state: SaliencyState, cfg: SaliencyConfig) -> bool:''',
'''def _directionally_supportive(positive, opposing, floor: float = 0.5) -> bool:
    """Reject missing, zero, or opposition-dominated evidence."""
    if positive is None or opposing is None:
        return False
    pos, neg = float(positive), float(opposing)
    if not (math.isfinite(pos) and math.isfinite(neg)) or pos <= 0 or neg < 0:
        return False
    return pos / max(pos + neg, 1e-12) > float(floor)


def _support_pass(parts: Dict[str, object], state: SaliencyState, cfg: SaliencyConfig) -> bool:''', 'directionality helper')
    src = change(src,
'''    if v is None or v < cfg.tau_visual:
        return False
    if not parts.get("neutral", False):
        t = _relative_support(parts.get("txt"), state.mean_txt(cfg.history_window))
        if t is None or t < cfg.tau_text:
            return False
    return True''',
'''    if v is None or v < cfg.tau_visual:
        return False
    if cfg.signed_saliency and not _directionally_supportive(
        parts.get("vis"), parts.get("vis_opp"), cfg.direction_min_ratio
    ):
        return False
    if not parts.get("neutral", False):
        t = _relative_support(parts.get("txt"), state.mean_txt(cfg.history_window))
        if t is None or t < cfg.tau_text:
            return False
        if cfg.signed_saliency and not _directionally_supportive(
            parts.get("txt"), parts.get("txt_opp"), cfg.direction_min_ratio
        ):
            return False
    return True''', 'polarity-aware support gate')
    src = change(src,
'''    if config.strict_grounding:
        if config.visual_lambda <= 0:''',
'''    if config.signed_saliency and not config.strict_grounding:
        raise ValueError("signed GSV requires strict_grounding=True")
    if config.signed_saliency and not (0.0 <= config.direction_min_ratio < 1.0):
        raise ValueError("direction_min_ratio must lie in [0, 1)")
    if config.strict_grounding:
        if config.visual_lambda <= 0:''', 'GSV validation')
    src = change(src,
'''            config.query_mode, vis_pos, config.strict_grounding
        )''',
'''            config.query_mode, vis_pos, config.strict_grounding, config.signed_saliency
        )''', 'signed saliency invocation')
    src = change(src,
'''        "chosen_visual_saliency": p.get("vis"),''',
'''        "chosen_visual_saliency": p.get("vis"),
        "chosen_visual_opposing_saliency": p.get("vis_opp"),
        "chosen_text_opposing_saliency": p.get("txt_opp"),''', 'chosen diagnostics')
    src = change(src,
'''                "visual_saliency": parts[cid].get("vis") if cid in parts else None,''',
'''                "visual_saliency": parts[cid].get("vis") if cid in parts else None,
                "visual_opposing_saliency": parts[cid].get("vis_opp") if cid in parts else None,
                "text_opposing_saliency": parts[cid].get("txt_opp") if cid in parts else None,''', 'candidate diagnostics')
    return src


def generator_patch(src: str) -> str:
    src = change(src,
'''        sgrs_strict_grounding=False, sgrs_tau_visual=0.25, sgrs_tau_text=0.25, sgrs_no_grounding_action="eos",''',
'''        sgrs_strict_grounding=False, sgrs_tau_visual=0.25, sgrs_tau_text=0.25, sgrs_no_grounding_action="eos",
        sgrs_signed_saliency=False, sgrs_direction_min_ratio=0.5,''', 'port preset')
    src = change(src,
'''        sgrs_strict_grounding=True, sgrs_tau_visual=0.25, sgrs_tau_text=0.25, sgrs_no_grounding_action="eos",''',
'''        sgrs_strict_grounding=True, sgrs_tau_visual=0.25, sgrs_tau_text=0.25, sgrs_no_grounding_action="eos",
        sgrs_signed_saliency=True, sgrs_direction_min_ratio=0.5,''', 'ours preset')
    src = change(src,
'''    p.add_argument("--sgrs_strict_grounding", type=str2bool, default=None)''',
'''    p.add_argument("--sgrs_strict_grounding", type=str2bool, default=None)
    p.add_argument("--sgrs_signed_saliency", type=str2bool, default=None)
    p.add_argument("--sgrs_direction_min_ratio", type=float, default=None)''', 'command line args')
    src = change(src,
'''        strict_grounding=args.sgrs_strict_grounding, tau_visual=args.sgrs_tau_visual,''',
'''        strict_grounding=args.sgrs_strict_grounding,
        signed_saliency=args.sgrs_signed_saliency,
        direction_min_ratio=args.sgrs_direction_min_ratio,
        tau_visual=args.sgrs_tau_visual,''', 'GSV configuration wiring')
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='Print proposed diffs; write no files')
    args = ap.parse_args()
    patches = {
        'saliency_qwen.py': saliency_patch,
        'src/saliency_qwen.py': saliency_patch,
        'generate_chair_all.py': generator_patch,
    }
    # Intentionally exclude src/generate_chair_all.py: older entrypoint with
    # no strict-GSV options. Legacy behavior there remains unchanged.
    staged = {}
    for name, updater in patches.items():
        target = Path(name)
        if not target.is_file():
            raise FileNotFoundError(f"Missing {name}; run from the RACER repository root")
        before = target.read_text(encoding='utf-8')
        if ('signed_saliency: bool = False' in before or
                'sgrs_signed_saliency=True' in before):
            raise RuntimeError(f"{name} appears already patched; refusing to reapply")
        after = updater(before)
        ast.parse(after, filename=name)   # fail before any writing
        staged[target] = (before, after)
    for target, (before, after) in staged.items():
        if args.dry_run:
            print(''.join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f'a/{target}', tofile=f'b/{target}'
            )))
        else:
            target.write_text(after, encoding='utf-8')
            print(f"Updated {target}")
    if args.dry_run:
        print('Dry run succeeded: all source anchors matched; no files changed.')
    else:
        print('Patch applied. Run tests, review git diff, and benchmark on the model.')

if __name__ == '__main__':
    main()
