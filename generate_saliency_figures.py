import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Replace these with your actual saved saliency arrays.
#
# Expected shape:
#   [num_generated_tokens, num_context_tokens]
#
# Example:
#   baseline_saliency = np.load("baseline_saliency.npy")
#   racer_saliency    = np.load("racer_saliency.npy")
# ------------------------------------------------------------

baseline_saliency = np.load("baseline_saliency.npy")
racer_saliency = np.load("racer_saliency.npy")

# Optional token labels
# Replace these with your real tokens.
#
# context_tokens = [...]
# reasoning_tokens = [...]

context_tokens = [
    "<image>",
    "Question",
    "What",
    "is",
    "shown",
    "?",
]

reasoning_tokens = [
    "The",
    "image",
    "shows",
    "a",
    "finding",
    "...",
]

# ------------------------------------------------------------
# Evidence token indices
#
# These are columns corresponding to image/evidence-relevant
# context tokens.
#
# IMPORTANT:
# Replace these with the actual indices in your token sequence.
# ------------------------------------------------------------

evidence_token_indices = [0, 1, 2]

# Reasoning step at which RACER intervention starts
intervention_step = 12


# ============================================================
# Utility functions
# ============================================================

def normalize_matrix(matrix):
    """
    Normalize saliency matrix into [0, 1].
    """
    matrix = np.asarray(matrix, dtype=np.float32)

    matrix = matrix - matrix.min()

    max_val = matrix.max()

    if max_val > 0:
        matrix = matrix / max_val

    return matrix


def smooth_curve(values, window=3):
    """
    Simple moving-average smoothing.
    """
    if window <= 1:
        return values

    kernel = np.ones(window) / window

    padded = np.pad(
        values,
        (window // 2, window - 1 - window // 2),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def compute_evidence_saliency(
    saliency_matrix,
    evidence_indices,
):
    """
    Aggregate saliency assigned to evidence-relevant tokens
    at every generated reasoning step.

    Input:
        saliency_matrix:
            [reasoning_tokens, context_tokens]

    Returns:
        [reasoning_tokens]
    """

    evidence_saliency = saliency_matrix[:, evidence_indices]

    # Mean evidence saliency for each generated token
    trajectory = evidence_saliency.mean(axis=1)

    return trajectory


# ============================================================
# Normalize with common scale
# ============================================================

baseline_saliency = np.asarray(
    baseline_saliency,
    dtype=np.float32,
)

racer_saliency = np.asarray(
    racer_saliency,
    dtype=np.float32,
)

# ------------------------------------------------------------
# IMPORTANT:
# Use the SAME scale for both heatmaps.
# Otherwise visual comparisons are misleading.
# ------------------------------------------------------------

global_min = min(
    baseline_saliency.min(),
    racer_saliency.min(),
)

global_max = max(
    baseline_saliency.max(),
    racer_saliency.max(),
)

baseline_plot = (
    baseline_saliency - global_min
) / (global_max - global_min + 1e-8)

racer_plot = (
    racer_saliency - global_min
) / (global_max - global_min + 1e-8)


# ============================================================
# Heatmap plotting function
# ============================================================

def plot_saliency_heatmap(
    saliency,
    output_file,
    title,
    context_tokens=None,
    reasoning_tokens=None,
):
    fig, ax = plt.subplots(
        figsize=(6.2, 5.2)
    )

    im = ax.imshow(
        saliency,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="magma",
        norm=Normalize(
            vmin=0,
            vmax=1,
        ),
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    ax.set_xlabel(
        "Context / Evidence Tokens",
        fontsize=12,
    )

    ax.set_ylabel(
        "Generated Reasoning Tokens",
        fontsize=12,
    )

    # --------------------------------------------------------
    # X-axis token labels
    # --------------------------------------------------------

    if (
        context_tokens is not None
        and len(context_tokens) == saliency.shape[1]
    ):
        ax.set_xticks(
            np.arange(len(context_tokens))
        )

        ax.set_xticklabels(
            context_tokens,
            rotation=60,
            ha="right",
            fontsize=7,
        )

    # --------------------------------------------------------
    # Y-axis token labels
    # --------------------------------------------------------

    if (
        reasoning_tokens is not None
        and len(reasoning_tokens) == saliency.shape[0]
    ):
        ax.set_yticks(
            np.arange(len(reasoning_tokens))
        )

        ax.set_yticklabels(
            reasoning_tokens,
            fontsize=7,
        )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Causal Saliency",
        fontsize=11,
    )

    # --------------------------------------------------------
    # Highlight evidence-related context tokens
    # --------------------------------------------------------

    for idx in evidence_token_indices:

        ax.axvspan(
            idx - 0.5,
            idx + 0.5,
            alpha=0.08,
        )

    ax.tick_params(
        axis="both",
        length=2,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# Figure (a): Baseline saliency
# ============================================================

plot_saliency_heatmap(
    saliency=baseline_plot,
    output_file=os.path.join(
        OUTPUT_DIR,
        "saliency_baseline.pdf",
    ),
    title="Baseline",
    context_tokens=context_tokens
    if len(context_tokens) == baseline_plot.shape[1]
    else None,
    reasoning_tokens=reasoning_tokens
    if len(reasoning_tokens) == baseline_plot.shape[0]
    else None,
)


# ============================================================
# Figure (b): RACER saliency
# ============================================================

plot_saliency_heatmap(
    saliency=racer_plot,
    output_file=os.path.join(
        OUTPUT_DIR,
        "saliency_racer.pdf",
    ),
    title="RACER",
    context_tokens=context_tokens
    if len(context_tokens) == racer_plot.shape[1]
    else None,
    reasoning_tokens=reasoning_tokens
    if len(reasoning_tokens) == racer_plot.shape[0]
    else None,
)


# ============================================================
# Figure (c): Evidence-saliency trajectory
# ============================================================

baseline_trajectory = compute_evidence_saliency(
    baseline_plot,
    evidence_token_indices,
)

racer_trajectory = compute_evidence_saliency(
    racer_plot,
    evidence_token_indices,
)

# Optional smoothing for visualization
baseline_trajectory = smooth_curve(
    baseline_trajectory,
    window=3,
)

racer_trajectory = smooth_curve(
    racer_trajectory,
    window=3,
)

steps = np.arange(
    1,
    len(baseline_trajectory) + 1,
)

fig, ax = plt.subplots(
    figsize=(7.2, 4.6)
)

ax.plot(
    steps,
    baseline_trajectory,
    linewidth=2.3,
    label="Baseline",
)

ax.plot(
    steps,
    racer_trajectory,
    linewidth=2.3,
    label="RACER",
)

# ------------------------------------------------------------
# RACER intervention location
# ------------------------------------------------------------

ax.axvline(
    intervention_step,
    linestyle="--",
    linewidth=1.8,
)

ax.text(
    intervention_step + 0.5,
    ax.get_ylim()[1] * 0.90,
    "RACER intervention",
    rotation=90,
    va="top",
    fontsize=10,
)

ax.set_xlabel(
    "Reasoning Step",
    fontsize=12,
)

ax.set_ylabel(
    "Evidence Saliency",
    fontsize=12,
)

ax.set_title(
    "Evidence Grounding During Reasoning",
    fontsize=14,
    fontweight="bold",
)

ax.legend(
    frameon=False,
    fontsize=11,
)

ax.grid(
    alpha=0.2,
    linestyle="--",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

trajectory_path = os.path.join(
    OUTPUT_DIR,
    "saliency_trajectory.pdf",
)

plt.savefig(
    trajectory_path,
    dpi=600,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {trajectory_path}")