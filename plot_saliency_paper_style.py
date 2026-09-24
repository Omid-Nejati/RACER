
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _normalize_lower(mat):
    """Min-max normalize only the valid causal (lower-triangular) entries."""
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("saliency must be a square [N, N] matrix.")
    n = mat.shape[0]
    valid = np.tril(np.ones((n, n), dtype=bool))
    vals = mat[valid]
    lo, hi = vals.min(), vals.max()
    out = np.zeros_like(mat, dtype=float)
    if hi > lo:
        out[valid] = (vals - lo) / (hi - lo)
    return out


def _causal_mask(mat):
    """Mask future-token locations (upper triangle)."""
    n = mat.shape[0]
    return np.ma.array(mat, mask=np.triu(np.ones((n, n), dtype=bool), k=1))


def _group_box(ax, x0, x1, y0, y1, edgecolor, lw=2.3):
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=edgecolor,
            linewidth=lw,
        )
    )


def plot_block_panel(ax, saliency, system_end, prompt_end, title="Saliency map"):
    """
    Paper-style schematic panel.

    Matrix convention:
        row = Query token
        col = Key token

    Boundaries:
        [0, system_end)        = system
        [system_end,prompt_end)= prompt/user
        [prompt_end,N)         = generated output
    """
    s = _normalize_lower(saliency)
    sm = _causal_mask(s)
    n = s.shape[0]

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#fffaf7")

    ax.imshow(sm, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=14)

    # Main causal diagonal.
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="#2f6bd1", lw=1.8)

    # Region separators.
    for b in (system_end, prompt_end):
        ax.axvline(b - 0.5, color="#666666", lw=2.0)
        ax.axhline(b - 0.5, color="#666666", lw=2.0)

    # Block outlines.
    _group_box(ax, 0, system_end, 0, system_end, "#2f6bd1")
    _group_box(ax, 0, system_end, system_end, prompt_end, "#2f6bd1")
    _group_box(ax, system_end, prompt_end, system_end, prompt_end, "#21a657")
    _group_box(ax, 0, system_end, prompt_end, n, "#2f6bd1")
    _group_box(ax, system_end, prompt_end, prompt_end, n, "#21a657")
    _group_box(ax, prompt_end, n, prompt_end, n, "#b03030")

    # Labels.
    def put(x, y, txt, color, fs=12):
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color)

    put(system_end * 0.32, system_end * 0.55, "System to\nsystem", "#2f6bd1")
    put(system_end * 0.36, (system_end + prompt_end) / 2,
        "System to\nprompt", "#2f6bd1")
    put((system_end + prompt_end) / 2,
        (system_end + prompt_end) / 2,
        "Prompt to\nprompt", "#21a657")
    put(system_end * 0.34, (prompt_end + n) / 2,
        "System\nto\noutput", "#2f6bd1")
    put((system_end + prompt_end) / 2, (prompt_end + n) / 2,
        "Prompt\nto\noutput", "#21a657")
    put((prompt_end + n) / 2, (prompt_end + n) / 2,
        "Output\nto\noutput", "#111111")

    # Query / Key arrows, positioned in upper-right masked region.
    ax.annotate("Query",
                xy=(n * 0.66, n * 0.10), xytext=(n * 0.80, n * 0.03),
                fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"),
                ha="center", va="center")
    ax.annotate("Key",
                xy=(n * 0.92, n * 0.34), xytext=(n * 0.92, n * 0.18),
                fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"),
                ha="center", va="center", rotation=90)

    # Hide dense token ticks for the schematic.
    ax.set_xticks([])
    ax.set_yticks([])

    # Bottom strip = saliency from all previous tokens to NEXT prediction.
    strip = s[-1, :]
    strip_ax = ax.inset_axes([0.00, -0.055, 1.00, 0.035])
    strip_ax.imshow(strip[None, :], cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])
    for spine in strip_ax.spines.values():
        spine.set_linewidth(1.2)

    ax.text(0.02, 0.02,
            "System/User tokens to next prediction token",
            transform=ax.transAxes, fontsize=11.5, fontweight="bold",
            color="#138d3c",
            bbox=dict(facecolor="#fff200", edgecolor="none", pad=1.5))

    # Group labels under the strip.
    y = n + 1.75
    ax.add_patch(Rectangle((-0.5, y), system_end, 2.2, fill=False, lw=1.2, clip_on=False))
    ax.add_patch(Rectangle((system_end - 0.5, y), prompt_end-system_end, 2.2, fill=False, lw=1.2, clip_on=False))
    ax.add_patch(Rectangle((prompt_end - 0.5, y), n-prompt_end, 2.2, fill=False, lw=1.2, clip_on=False))
    ax.text((system_end-1)/2, y+1.1, "System", ha="center", va="center",
            fontsize=11, fontweight="bold", clip_on=False)
    ax.text((system_end+prompt_end-1)/2, y+1.1, "P", ha="center", va="center",
            fontsize=11, fontweight="bold", clip_on=False)
    ax.text((prompt_end+n-1)/2, y+1.1, "Output", ha="center", va="center",
            fontsize=11, fontweight="bold", clip_on=False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n + 4.3, -0.5)


def plot_token_panel(
    ax,
    saliency,
    tokens,
    sentence_spans=None,
    high_saliency_text="high saliency",
):
    """
    Token-level triangular saliency map.

    sentence_spans example:
        [(0, 5, "sentence1"), (5, 12, "sentence2"), (12, 20, "sentence3")]
    """
    s = _normalize_lower(saliency)
    sm = _causal_mask(s)
    n = len(tokens)
    if s.shape != (n, n):
        raise ValueError("len(tokens) must equal saliency.shape[0].")

    cmap = plt.get_cmap("RdPu").copy()
    cmap.set_bad("#fffaf7")

    ax.imshow(sm, cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", origin="upper", aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(tokens, fontsize=7)

    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Bottom green rectangle, like the example figure.
    ax.add_patch(Rectangle(
        (-0.5, n - 1.55), n, 1.05,
        fill=False, edgecolor="#24ad4f", linewidth=2.5
    ))

    if sentence_spans:
        for start, end, label in sentence_spans:
            cx = (start + end - 1) / 2
            ax.text(
                cx, n - 2.0, label,
                ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="#d8d8d8", edgecolor="none", pad=2.0)
            )

    # Put annotation in a relatively empty part of the causal triangle.
    ax.text(
        n * 0.47, n * 0.78, high_saliency_text,
        color="#146fc4", fontsize=19, fontstyle="italic",
        fontfamily="serif"
    )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)


def make_demo_saliency(n, seed=7, late_boost=True):
    """
    Synthetic causal matrix only for demonstrating the plotting style.
    Replace this with your real saliency matrix.
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((n, n), dtype=float)

    for q in range(n):
        k = np.arange(q + 1)
        # locality bias + low random background
        m[q, :q+1] = (
            0.12 * rng.random(q + 1)
            + 0.25 * np.exp(-(q - k) / 4.5)
        )
        # self/near-diagonal saliency
        m[q, q] += 0.35

    if late_boost and n > 8:
        # Stronger structured saliency near the end, for a visual pattern
        for q in range(int(n * 0.72), n):
            start = max(0, q - 5)
            m[q, start:q+1] += np.linspace(0.10, 0.60, q-start+1)

    return m


def plot_paper_style_figure(
    saliency_full,
    system_end,
    prompt_end,
    output_tokens,
    output_saliency,
    sentence_spans=None,
    save_prefix="saliency_paper_style",
):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(15.5, 7.2),
        gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    plot_block_panel(
        axes[0], saliency_full,
        system_end=system_end,
        prompt_end=prompt_end,
        title="Saliency map"
    )

    plot_token_panel(
        axes[1],
        output_saliency,
        output_tokens,
        sentence_spans=sentence_spans,
        high_saliency_text="high saliency"
    )

    fig.tight_layout(w_pad=2.2)
    fig.savefig(f"{save_prefix}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # ---------------------------
    # 1) DEMO full sequence map
    # ---------------------------
    N = 42
    SYSTEM_END = 14
    PROMPT_END = 20
    full_saliency = make_demo_saliency(N, seed=11, late_boost=True)

    # ---------------------------
    # 2) DEMO output-only map
    # ---------------------------
    output_tokens = [
        "The", "image", "shows", "a", "cozy", "vintage", "room",
        "with", "a", "mix", "of", "traditional", "and", "rustic",
        "elements", ".", "The", "room", "features", "a", "sloped",
        "ceiling", "with", "striped", "detail"
    ]

    output_saliency = make_demo_saliency(
        len(output_tokens), seed=23, late_boost=True
    )

    sentence_spans = [
        (0, 5, "sentence1"),
        (5, 15, "sentence2"),
        (15, len(output_tokens), "sentence3"),
    ]

    plot_paper_style_figure(
        saliency_full=full_saliency,
        system_end=SYSTEM_END,
        prompt_end=PROMPT_END,
        output_tokens=output_tokens,
        output_saliency=output_saliency,
        sentence_spans=sentence_spans,
        save_prefix="saliency_paper_style",
    )
