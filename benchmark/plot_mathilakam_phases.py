"""
Generate a Mathilakam phase-progression chart for the report.

Shows val TER (real Mathilakam holdout) on the y-axis and Phase 1..4
on the x-axis, with one line per architecture and the project-best
trajectory highlighted.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("report/DUK SoCSE Thesis Template/figures/mathilakam_phases.png")

# Data captured directly from benchmark/results/mathilakam/phaseN_results.csv.
# Phase 4 is the cnn_ctc + LM rescore best at alpha=0.3. The other archs
# in Phase 4 reuse their Phase 3 score because we only LM-rescored cnn_ctc.
PHASES   = [1, 2, 3, 4]
SERIES   = {
    "cnn_ctc":  [91.5, 89.34, 87.29, 82.47],
    "crnn_ctc": [93.0, 91.80, 88.32, 88.32],
    "conformer":[92.6, 89.34, 89.69, 89.69],
}
COLOURS  = {
    "cnn_ctc":  "#1f77b4",
    "crnn_ctc": "#2ca02c",
    "conformer":"#d62728",
}
PHASE_LABEL = {
    1: "P1\nreal-only,\n50ep+aug",
    2: "P2\nsynth+real,\nmode-collapse",
    3: "P3\nsynth+stitched\n+real x50",
    4: "P4\n+ 5-gram LM\nrescoring",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arch, ys in SERIES.items():
        ax.plot(PHASES, ys, "o-", color=COLOURS[arch], linewidth=2.0,
                 markersize=8, label=arch)
        for x, y in zip(PHASES, ys):
            ax.annotate(f"{y:.1f}%", (x, y),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8.5,
                         color=COLOURS[arch], weight="bold")

    # Annotate the best-of-project trajectory.
    best_x = PHASES
    best_y = [min(s[i] for s in SERIES.values()) for i in range(len(PHASES))]
    ax.plot(best_x, best_y, "k--", alpha=0.4, linewidth=1.0,
             label="best of project")

    ax.set_xticks(PHASES)
    ax.set_xticklabels([PHASE_LABEL[p] for p in PHASES], fontsize=9)
    ax.set_ylabel("Validation TER (%)  — lower is better", fontsize=10)
    ax.set_title("Mathilakam phase progression (val on 15-line real holdout)",
                  fontsize=11, weight="bold")
    ax.set_ylim(80, 95)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
