"""
Render the cnn_ctc line-recognizer architecture diagram as a PNG.

Replaces the TikZ block in ch3.tex. Layout uses explicit horizontal
lanes per element so labels never collide with figures, and arrows
never cross any text:

    y =  9.0  : title
    y =  7.7  : stage names (Input, Conv 1, ..., Output)
    y = 4.5--7.0 : visual band (input image, feature maps, neurons,
                  softmax bars)
    y =  4.0  : tensor-shape labels (1x64xW, 64ch 32xW/2, ...)
    y =  3.4  : descriptive labels (Conv+ReLU+MaxPool, ...)
    y =  2.0  : CTC decode block
    y =  1.0  : final transcription text
    y =  0.0  : section braces
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


OUT = "report/DUK SoCSE Thesis Template/figures/cnn_ctc_arch.png"


# Lane y-coordinates. Centralised so the layout is easy to tune.
Y_TITLE   = 9.20
Y_STAGE   = 7.95          # stage name row
Y_VIS_TOP = 7.30          # top of the visual band
Y_VIS_BOT = 4.60          # bottom of the visual band
Y_VIS_MID = (Y_VIS_TOP + Y_VIS_BOT) / 2
Y_DIM     = 4.10          # tensor-shape labels
Y_DESC    = 3.45          # description labels
Y_CTC     = 2.10          # CTC decode block
Y_OUT     = 1.10          # transcription text
Y_BRACE   = 0.30          # section-brace lane
Y_BRACE_T = 0.05          # section-brace text


def stack(ax, x, y_top, w, h, color_base="#dde8f5", n=3, off=0.10):
    """Three stacked rectangles to suggest a feature-map tensor.
    `(x, y_top)` is the top-left of the front face."""
    base = np.array([0.86, 0.91, 0.96])
    for i in range(n - 1, -1, -1):
        c = base - 0.13 * i
        c = np.clip(c, 0, 1)
        ax.add_patch(mpatches.Rectangle(
            (x + i * off, y_top - h - i * off), w, h,
            facecolor=tuple(c), edgecolor="black", linewidth=0.7))


def main():
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 17.0)
    ax.set_ylim(-0.7, 9.6)
    ax.axis("off")
    ax.set_aspect("auto")

    # ---- Title ----
    ax.text(8.5, Y_TITLE, "Convolutional Line Recognizer (cnn_ctc)",
            ha="center", va="center", fontsize=15, weight="bold")

    # =================================================================
    # Lane definitions: x-centre of each stage column, x-extent of its
    # visual element, and stage label.
    # =================================================================
    cols = {
        "input":   {"x": 1.0, "label": "Input"},
        "conv1":   {"x": 3.7, "label": "Conv stage 1"},
        "conv2":   {"x": 6.0, "label": "Conv stage 2"},
        "conv3":   {"x": 8.1, "label": "Conv stage 3"},
        "flatten": {"x": 9.9, "label": "Flatten"},
        "fc":      {"x": 12.0, "label": "Fully connected"},
        "output":  {"x": 14.5, "label": "Output"},
    }
    for k, c in cols.items():
        ax.text(c["x"], Y_STAGE, c["label"], ha="center", va="center",
                fontsize=10.5, weight="bold")

    # =================================================================
    # Input image with kernel highlight (stage column 0)
    # =================================================================
    ix = cols["input"]["x"]
    img_w, img_h = 1.7, 2.3
    ax.add_patch(mpatches.Rectangle(
        (ix - img_w/2, Y_VIS_BOT + 0.15), img_w, img_h,
        facecolor="#efefef", edgecolor="black", linewidth=0.8))
    # Sample stroke pattern
    for x in np.linspace(ix - img_w/2 + 0.2, ix + img_w/2 - 0.05, 8):
        for y in np.linspace(Y_VIS_BOT + 0.45, Y_VIS_BOT + 2.20, 5):
            ax.plot(x, y, "k.", markersize=2.5)
    # Kernel highlight (orange box on the input)
    kx, ky = ix - 0.65, Y_VIS_BOT + 1.3
    ax.add_patch(mpatches.Rectangle(
        (kx, ky), 0.28, 0.28,
        facecolor="none", edgecolor="#b85a16", linewidth=1.6))
    # Tensor shape + description for input
    ax.text(ix, Y_DIM, r"$1\times64\times W$", ha="center",
            fontsize=10)
    ax.text(ix, Y_DESC, "raw line image",
            ha="center", fontsize=9.5, style="italic")

    # =================================================================
    # Conv stages 1, 2, 3 — feature-map stacks
    # =================================================================
    conv_specs = [
        ("conv1", 1.30, 1.85, "64ch  $32 \\times W/2$",
                              "Conv + ReLU + MaxPool"),
        ("conv2", 1.10, 1.55, "256ch  $8 \\times W/8$",
                              "Conv + ReLU + MaxPool"),
        ("conv3", 0.90, 1.25, "512ch  $1 \\times W/8$",
                              "Conv + ReLU + MaxPool"),
    ]
    for key, w, h, dim, desc in conv_specs:
        cx = cols[key]["x"]
        # Top of stack ramps lower as channels deepen visually
        y_top = Y_VIS_TOP - 0.4 if key == "conv1" else \
                Y_VIS_TOP - 0.7 if key == "conv2" else \
                Y_VIS_TOP - 1.0
        stack(ax, cx - w/2 - 0.10, y_top, w, h)
        ax.text(cx, Y_DIM, dim, ha="center", fontsize=10)
        ax.text(cx, Y_DESC, desc, ha="center", fontsize=9.5,
                style="italic")

    # =================================================================
    # Arrows BETWEEN stages (in the visual band, NOT through any text)
    # =================================================================
    arrow_y = Y_VIS_MID + 0.2
    arrow_pairs = [
        (cols["input"]["x"]   + 0.95, cols["conv1"]["x"]   - 0.85),
        (cols["conv1"]["x"]   + 0.85, cols["conv2"]["x"]   - 0.75),
        (cols["conv2"]["x"]   + 0.75, cols["conv3"]["x"]   - 0.65),
        (cols["conv3"]["x"]   + 0.65, cols["flatten"]["x"] - 0.65),
        (cols["flatten"]["x"] + 0.65, cols["fc"]["x"]      - 0.95),
        (cols["fc"]["x"]      + 0.95, cols["output"]["x"]  - 0.95),
    ]
    for x0, x1 in arrow_pairs:
        ax.annotate("", xy=(x1, arrow_y), xytext=(x0, arrow_y),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # =================================================================
    # Flatten / time-series strip
    # =================================================================
    fx = cols["flatten"]["x"]
    n_strips = 9
    strip_w = 0.10
    strip_gap = 0.03
    strip_h = 1.6
    total_w = n_strips * strip_w + (n_strips - 1) * strip_gap
    x_start = fx - total_w / 2
    for i in range(n_strips):
        sx = x_start + i * (strip_w + strip_gap)
        ax.add_patch(mpatches.Rectangle(
            (sx, Y_VIS_TOP - 1.7 - strip_h + 1.7),
            strip_w, strip_h,
            facecolor="#cfe5f3", edgecolor="black", linewidth=0.5))
    ax.text(fx, Y_DIM, r"$T \times 512$", ha="center", fontsize=10)
    ax.text(fx, Y_DESC, "time series",
            ha="center", fontsize=9.5, style="italic")

    # =================================================================
    # Fully connected layer (two columns of neurons + cross-connections)
    # =================================================================
    fcx = cols["fc"]["x"]
    a_x = fcx - 0.6
    b_x = fcx + 0.6
    a_pts = [(a_x, Y_VIS_TOP - 0.30 - i * 0.28) for i in range(7)]
    b_pts = [(b_x, Y_VIS_TOP - 0.45 - i * 0.32) for i in range(5)]
    for p, q in [(p, q) for p in a_pts for q in b_pts]:
        ax.plot([p[0], q[0]], [p[1], q[1]],
                color="#999999", linewidth=0.3, zorder=1)
    for p in a_pts:
        ax.plot(*p, "o", color="#9bd0e3", markersize=8,
                markeredgecolor="black", zorder=2)
    for p in b_pts:
        ax.plot(*p, "o", color="#5a7fbf", markersize=8,
                markeredgecolor="black", zorder=2)
    ax.text(fcx, Y_DIM, r"Linear $\mathbb{R}^{512}\!\to\!\mathbb{R}^{|V|+1}$",
            ha="center", fontsize=9.5)
    ax.text(fcx, Y_DESC, "per-step softmax",
            ha="center", fontsize=9.5, style="italic")

    # =================================================================
    # Output: per-timestep softmax bars
    # =================================================================
    ox = cols["output"]["x"]
    bar_w = 0.20
    bar_gap = 0.13
    bar_specs = [(0.85, "ka"), (0.70, "li"), (0.55, "la"),
                 (0.45, "pe"), (0.40, "ri")]
    n_bars = len(bar_specs)
    total_bar_w = n_bars * bar_w + (n_bars - 1) * bar_gap
    bx_start = ox - total_bar_w / 2
    bar_base = Y_VIS_BOT + 0.30
    for i, (h, lab) in enumerate(bar_specs):
        bx = bx_start + i * (bar_w + bar_gap)
        ax.add_patch(mpatches.Rectangle(
            (bx, bar_base), bar_w, h,
            facecolor="#3d75b5", edgecolor="black", linewidth=0.6))
        ax.text(bx + bar_w/2, bar_base - 0.18, lab,
                ha="center", fontsize=8.5)
    ax.text(ox, Y_DIM, "softmax / step", ha="center", fontsize=10)
    ax.text(ox, Y_DESC, "per-token probabilities",
            ha="center", fontsize=9.5, style="italic")

    # =================================================================
    # CTC decode block — well below the visual band
    # =================================================================
    ctc_w = 3.0
    ctc_h = 0.6
    ax.add_patch(mpatches.FancyBboxPatch(
        (ox - ctc_w/2, Y_CTC - ctc_h/2), ctc_w, ctc_h,
        boxstyle="round,pad=0.06,rounding_size=0.05",
        facecolor="#f7c8b8", edgecolor="black", linewidth=0.8))
    ax.text(ox, Y_CTC, "CTC: collapse blanks",
            ha="center", va="center", fontsize=10)
    # Arrow from output bars down to CTC block (vertical, no overlap)
    ax.annotate("", xy=(ox, Y_CTC + ctc_h/2 + 0.05),
                xytext=(ox, bar_base - 0.45),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # =================================================================
    # Final transcription text
    # =================================================================
    ax.text(8.5, Y_OUT, r'$\quad ka / li / la / pe / ri / \ldots \quad$',
            ha="center", va="center", fontsize=11, style="italic",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#fff5cc", edgecolor="black", linewidth=0.6))
    # Arrow from CTC block to transcription
    ax.annotate("", xy=(8.5 + 1.8, Y_OUT), xytext=(ox - ctc_w/2, Y_CTC),
        arrowprops=dict(arrowstyle="->", color="black",
                        lw=1.0, connectionstyle="arc3,rad=0.15"))

    # =================================================================
    # Section braces (drawn as horizontal bars with vertical end caps)
    # =================================================================
    spans = [
        (cols["input"]["x"]   - 0.95, cols["flatten"]["x"] + 0.65,
         "Feature Extraction"),
        (cols["fc"]["x"]      - 0.95, cols["output"]["x"]  + 1.0,
         "Per-step Classification"),
    ]
    for x0, x1, lab in spans:
        ax.plot([x0, x1], [Y_BRACE, Y_BRACE], "k-", lw=1.4)
        ax.plot([x0, x0], [Y_BRACE, Y_BRACE + 0.10], "k-", lw=1.4)
        ax.plot([x1, x1], [Y_BRACE, Y_BRACE + 0.10], "k-", lw=1.4)
        ax.text((x0 + x1) / 2, Y_BRACE_T, lab,
                ha="center", va="top", fontsize=10.5, weight="bold")

    plt.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    import os
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
