#!/usr/bin/env python3
"""
Generate publication-ready saliency figures for RACER / Qwen2.5-VL.

Outputs:
  1) saliency_baseline.pdf / .png
  2) saliency_racer.pdf / .png
  3) saliency_trajectory.pdf / .png   (when per-step saliency files are provided)

The heatmaps use ONE shared scale across baseline and RACER.

Typical usage
-------------
A) Two heatmaps from single saliency matrices:

python generate_saliency_figures.py \
  --baseline-matrix results/saliency/baseline/sample_001/step_17.npy \
  --racer-matrix results/saliency/racer/sample_001/step_17.npy \
  --baseline-tokens results/saliency/baseline/sample_001/step_17_tokens.txt \
  --racer-tokens results/saliency/racer/sample_001/step_17_tokens.txt \
  --output-dir figures/saliency

B) Heatmaps + trajectory from per-step matrices:

python generate_saliency_figures.py \
  --baseline-matrix results/saliency/baseline/sample_001/step_17.npy \
  --racer-matrix results/saliency/racer/sample_001/step_17.npy \
  --baseline-steps-dir results/saliency/baseline/sample_001/full_steps \
  --racer-steps-dir results/saliency/racer/sample_001/full_steps \
  --evidence-indices "10:45" \
  --intervention-step 17 \
  --output-dir figures/saliency

Per-step files should be named so the step index can be extracted, e.g.
  step_00.npy
  step_01.npy
  step_02.npy
  ...

For the trajectory, each per-step matrix is assumed to be a causal
token-to-token saliency matrix. The evidence score at a step is the sum
of saliency from the FINAL query row to the selected evidence/key columns.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_tokens(path):
    """Load token labels from actual_qwen_saliency.py *_tokens.txt output."""
    if path is None:
        return None

    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            labels.append(parts[1] if len(parts) == 2 else parts[0])
    return labels


def parse_indices(spec):
    """
    Parse evidence indices.

    Supported:
      "1,2,3,8"
      "10:45"      -> 10..44
      "10:45,60:70,90"
    """
    if not spec:
        return None

    indices = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if ":" in chunk:
            start_s, end_s = chunk.split(":", 1)
            start = int(start_s)
            end = int(end_s)
            indices.extend(range(start, end))
        else:
            indices.append(int(chunk))

    return sorted(set(indices))


def validate_matrix(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {matrix.shape}")
    return matrix


def common_normalize(a, b):
    """Normalize two matrices with the SAME min/max."""
    a = validate_matrix(a, "baseline")
    b = validate_matrix(b, "RACER")

    lo = min(float(np.nanmin(a)), float(np.nanmin(b)))
    hi = max(float(np.nanmax(a)), float(np.nanmax(b)))

    if hi <= lo:
        return np.zeros_like(a), np.zeros_like(b)

    return (a - lo) / (hi - lo), (b - lo) / (hi - lo)


def choose_tick_indices(n, max_ticks=45):
    if n <= max_ticks:
        return np.arange(n)
    step = max(1, int(np.ceil(n / max_ticks)))
    return np.arange(0, n, step)


def plot_heatmap(matrix, output_stem, title, labels=None):
    """Plot a causal/saliency heatmap."""
    rows, cols = matrix.shape

    fig_w = min(max(6.0, cols * 0.09), 14.0)
    fig_h = min(max(5.0, rows * 0.09), 12.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Key / Context Token", fontsize=11)
    ax.set_ylabel("Query Token", fontsize=11)

    if labels is not None and len(labels) == cols:
        xidx = choose_tick_indices(cols)
        ax.set_xticks(xidx)
        ax.set_xticklabels(
            [labels[i] for i in xidx],
            rotation=90,
            fontsize=5,
        )

    if labels is not None and len(labels) == rows:
        yidx = choose_tick_indices(rows)
        ax.set_yticks(yidx)
        ax.set_yticklabels(
            [labels[i] for i in yidx],
            fontsize=5,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized causal saliency", fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def extract_step_number(path):
    """Extract the last integer from a filename stem."""
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else None


def collect_step_files(directory):
    files = sorted(Path(directory).glob("*.npy"))
    pairs = []

    for p in files:
        step = extract_step_number(p)
        if step is not None:
            pairs.append((step, p))

    pairs.sort(key=lambda x: x[0])
    return pairs


def evidence_score_from_matrix(matrix, evidence_indices):
    """
    Evidence score for one target step.

    Uses saliency from the final query row to the selected evidence/key
    columns. This is appropriate when each .npy corresponds to a target
    generation step produced by actual_qwen_saliency.py.
    """
    matrix = validate_matrix(matrix, "step saliency")

    valid = [i for i in evidence_indices if 0 <= i < matrix.shape[1]]
    if not valid:
        raise ValueError(
            f"No evidence indices are valid for matrix width {matrix.shape[1]}"
        )

    final_query = matrix[-1, valid]
    return float(np.sum(final_query))


def build_trajectory(directory, evidence_indices):
    items = collect_step_files(directory)
    if not items:
        raise FileNotFoundError(f"No step_*.npy-style files found in: {directory}")

    steps = []
    scores = []

    for step, path in items:
        matrix = np.load(path)
        score = evidence_score_from_matrix(matrix, evidence_indices)
        steps.append(step)
        scores.append(score)

    return np.asarray(steps), np.asarray(scores, dtype=np.float32)


def normalize_trajectories(base_scores, racer_scores):
    """Normalize two trajectories with one shared max."""
    hi = max(
        float(np.max(base_scores)) if len(base_scores) else 0.0,
        float(np.max(racer_scores)) if len(racer_scores) else 0.0,
    )

    if hi <= 0:
        return base_scores, racer_scores

    return base_scores / hi, racer_scores / hi


def plot_trajectory(
    baseline_steps,
    baseline_scores,
    racer_steps,
    racer_scores,
    output_stem,
    intervention_step=None,
):
    baseline_scores, racer_scores = normalize_trajectories(
        baseline_scores,
        racer_scores,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(
        baseline_steps,
        baseline_scores,
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="Baseline",
    )

    ax.plot(
        racer_steps,
        racer_scores,
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="RACER",
    )

    if intervention_step is not None:
        ax.axvline(
            intervention_step,
            linestyle="--",
            linewidth=1.7,
        )

        ymax = ax.get_ylim()[1]
        ax.text(
            intervention_step + 0.35,
            ymax * 0.95,
            "RACER intervention",
            rotation=90,
            va="top",
            fontsize=9,
        )

    ax.set_xlabel("Reasoning / Generation Step", fontsize=11)
    ax.set_ylabel("Normalized Evidence Saliency", fontsize=11)
    ax.set_title(
        "Evidence Grounding During Reasoning",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-matrix",
        required=True,
        help="Baseline .npy saliency matrix.",
    )
    parser.add_argument(
        "--racer-matrix",
        required=True,
        help="RACER .npy saliency matrix.",
    )
    parser.add_argument(
        "--baseline-tokens",
        default=None,
        help="Optional baseline *_tokens.txt file.",
    )
    parser.add_argument(
        "--racer-tokens",
        default=None,
        help="Optional RACER *_tokens.txt file.",
    )
    parser.add_argument(
        "--baseline-steps-dir",
        default=None,
        help="Optional directory of per-step baseline .npy matrices.",
    )
    parser.add_argument(
        "--racer-steps-dir",
        default=None,
        help="Optional directory of per-step RACER .npy matrices.",
    )
    parser.add_argument(
        "--evidence-indices",
        default=None,
        help='Evidence/key token columns, e.g. "10:45" or "10:45,60:70".',
    )
    parser.add_argument(
        "--intervention-step",
        type=int,
        default=None,
        help="Reasoning step where RACER intervention occurs.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/saliency",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_raw = np.load(args.baseline_matrix)
    racer_raw = np.load(args.racer_matrix)

    baseline_norm, racer_norm = common_normalize(
        baseline_raw,
        racer_raw,
    )

    baseline_tokens = load_tokens(args.baseline_tokens)
    racer_tokens = load_tokens(args.racer_tokens)

    plot_heatmap(
        baseline_norm,
        os.path.join(args.output_dir, "saliency_baseline"),
        "Baseline",
        labels=baseline_tokens,
    )

    plot_heatmap(
        racer_norm,
        os.path.join(args.output_dir, "saliency_racer"),
        "RACER",
        labels=racer_tokens,
    )

    print(f"Saved: {os.path.join(args.output_dir, 'saliency_baseline.pdf')}")
    print(f"Saved: {os.path.join(args.output_dir, 'saliency_racer.pdf')}")

    trajectory_requested = (
        args.baseline_steps_dir is not None
        or args.racer_steps_dir is not None
    )

    if trajectory_requested:
        if args.baseline_steps_dir is None or args.racer_steps_dir is None:
            raise ValueError(
                "For trajectory generation, provide BOTH "
                "--baseline-steps-dir and --racer-steps-dir."
            )

        evidence_indices = parse_indices(args.evidence_indices)
        if not evidence_indices:
            raise ValueError(
                "Trajectory generation requires --evidence-indices."
            )

        b_steps, b_scores = build_trajectory(
            args.baseline_steps_dir,
            evidence_indices,
        )

        r_steps, r_scores = build_trajectory(
            args.racer_steps_dir,
            evidence_indices,
        )

        plot_trajectory(
            b_steps,
            b_scores,
            r_steps,
            r_scores,
            os.path.join(args.output_dir, "saliency_trajectory"),
            intervention_step=args.intervention_step,
        )

        print(f"Saved: {os.path.join(args.output_dir, 'saliency_trajectory.pdf')}")
    else:
        print(
            "Trajectory not generated. To create it, also provide "
            "--baseline-steps-dir, --racer-steps-dir, and --evidence-indices."
        )


if __name__ == "__main__":
    main()
