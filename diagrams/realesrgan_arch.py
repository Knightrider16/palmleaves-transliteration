"""
Render the Real-ESRGAN super-resolution architecture diagram as PNG.

Replaces the TikZ block in ch3.tex (Stage 2 of the preprocessing
pipeline). Mirrors the TikZ layout but rendered as an image so
Overleaf does not have to lay out the diagram on every compile.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


OUT = "report/DUK SoCSE Thesis Template/figures/realesrgan_arch.png"


def box(ax, x, y, w, h, text, color, fontsize=9, weight="normal"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        facecolor=color, edgecolor="black", linewidth=0.8))
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def arrow(ax, p1, p2, **kw):
    ax.annotate("", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0, **kw))


def feature_stack(ax, x, y, w, h):
    """Stacked feature-map style rectangle."""
    for i in range(2, -1, -1):
        off = i * 0.06
        ax.add_patch(mpatches.Rectangle(
            (x - w/2 + off, y - h/2 + off), w, h,
            facecolor=("#dbeaf8", "#bcd6ed", "#9bbedf")[i],
            edgecolor="black", linewidth=0.6))


def main():
    fig, ax = plt.subplots(figsize=(14.5, 7))
    ax.set_xlim(0, 16); ax.set_ylim(-1, 7.5); ax.axis("off")
    ax.set_aspect("auto")

    # title
    ax.text(8, 7.0, r"Real-ESRGAN 2× Super-Resolution Generator",
            ha="center", va="center", fontsize=14, weight="bold")

    # main row
    box(ax, 0.9, 5.5, 1.7, 1.0, "Low-res\ninput\nH×W×3", "#fff5cc",
        fontsize=9)
    feature_stack(ax, 2.8, 5.5, 0.7, 1.0)
    ax.text(2.8, 5.5, "Conv\n3×3\n64ch", ha="center", va="center",
            fontsize=8)
    box(ax, 5.0, 5.5, 2.2, 1.0,
        "23 stacked\nRRDB blocks\n(β = 0.2)", "#cdddef")
    feature_stack(ax, 7.2, 5.5, 0.7, 1.0)
    ax.text(7.2, 5.5, "Conv\n3×3\n64ch", ha="center", va="center",
            fontsize=8)

    # add node
    ax.add_patch(mpatches.Circle((8.4, 5.5), 0.18,
        facecolor="white", edgecolor="black", linewidth=1.0))
    ax.text(8.4, 5.5, "+", ha="center", va="center",
            fontsize=12, weight="bold")

    box(ax, 9.85, 5.5, 1.6, 1.0, "Pixel-Shuffle\n↑ 2", "#ffd7a8")
    box(ax, 11.65, 5.5, 1.4, 1.0, "Conv 3×3\n+ LReLU", "#cde6cd")
    box(ax, 13.10, 5.5, 1.1, 1.0, "Conv\n3×3", "#cde6cd")
    box(ax, 14.85, 5.5, 1.7, 1.0, "SR output\n2H×2W×3", "#fff5cc")

    # main arrows
    arrow(ax, (1.75, 5.5), (2.40, 5.5))
    arrow(ax, (3.20, 5.5), (3.85, 5.5))
    arrow(ax, (6.10, 5.5), (6.80, 5.5))
    arrow(ax, (7.60, 5.5), (8.20, 5.5))
    arrow(ax, (8.60, 5.5), (9.00, 5.5))
    arrow(ax, (10.65, 5.5), (10.90, 5.5))
    arrow(ax, (12.35, 5.5), (12.55, 5.5))
    arrow(ax, (13.65, 5.5), (14.00, 5.5))

    # long skip connection
    ax.plot([2.80, 2.80, 8.40], [6.0, 6.4, 6.4], "k-", lw=1.0)
    arrow(ax, (8.40, 6.4), (8.40, 5.68))
    ax.text(5.5, 6.50, "long skip connection",
            ha="center", fontsize=8.5)

    # ---- RRDB internal blow-up ----
    ax.text(7.0, 3.55, "RRDB internal: dense + dense + dense, scaled residual",
            ha="center", fontsize=10.5, weight="bold")
    for i, label in enumerate(["DDB$_1$", "DDB$_2$", "DDB$_3$"]):
        x_c = 2.5 + i * 2.0
        ax.add_patch(mpatches.Rectangle(
            (x_c, 2.2), 1.7, 0.85,
            facecolor="#b8cfeb", edgecolor="black", linewidth=0.7))
        ax.text(x_c + 0.85, 2.75, label, ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(x_c + 0.85, 2.4, "5 conv + LReLU", ha="center",
                va="center", fontsize=7.5)
        if i < 2:
            arrow(ax, (x_c + 1.7, 2.62), (x_c + 2.0, 2.62))
    ax.add_patch(mpatches.Circle((9.4, 2.62), 0.18,
        facecolor="white", edgecolor="black", linewidth=1.0))
    ax.text(9.4, 2.62, "+", ha="center", va="center",
            fontsize=12, weight="bold")
    ax.text(9.4, 2.95, "β", ha="center", fontsize=8, style="italic")
    arrow(ax, (8.20, 2.62), (9.22, 2.62))
    arrow(ax, (9.58, 2.62), (10.20, 2.62))
    ax.text(10.40, 2.62, "output", ha="left", va="center", fontsize=9)

    # inner residual loop
    ax.plot([2.5, 2.5, 9.4], [2.20, 1.80, 1.80], "k-", lw=1.0)
    arrow(ax, (9.4, 1.80), (9.4, 2.44))
    ax.text(6.0, 1.55, "residual skip", ha="center", fontsize=8.5)

    # bottom braces (just labels under main row)
    spans = [
        (0.05, 2.40, "Encoder"),
        (2.45, 8.55, "Trunk (RRDBs + skip)"),
        (8.65, 11.30, "Upsampler"),
        (11.40, 15.70, "Decoder + output"),
    ]
    for x0, x1, lab in spans:
        ax.plot([x0, x1], [4.85, 4.85], "k-", lw=1.4)
        ax.plot([x0, x0], [4.85, 4.95], "k-", lw=1.4)
        ax.plot([x1, x1], [4.85, 4.95], "k-", lw=1.4)
        ax.text((x0 + x1) / 2, 4.55, lab, ha="center", va="top",
                fontsize=10, weight="bold")

    plt.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    import os
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
