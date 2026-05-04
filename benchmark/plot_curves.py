"""
Generate training-curve plots from the per-model log.csv files.

Reads either a single CharClassifier log:
    benchmark/results/amadi_charrec/log.csv  (epoch, train_loss, val_acc)

or one or more line-model logs:
    benchmark/results/amadi_lineocr/<arch>/log.csv (epoch, train_loss, val_ter_pct, ...)

Saves a PNG into the same folder, plus a combined "all models" plot at
benchmark/results/amadi_lineocr/curves.png if multiple logs exist.

Usage:
    python -m benchmark.plot_curves
"""
from __future__ import annotations
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(log: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if not log.exists():
        return out
    with open(log, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                out.setdefault(k, []).append(float(v) if v else 0.0)
    return out


def plot_charrec(log: Path, out: Path):
    d = _load(log)
    if not d:
        print(f"no data at {log}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(d["epoch"], d["train_loss"], "o-", color="C0")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train cross-entropy loss")
    axes[0].set_title("AMADI CharClassifier — training loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(d["epoch"], d["val_acc"], "o-", color="C1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val accuracy (%)")
    axes[1].set_title("AMADI CharClassifier — validation accuracy")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"wrote {out}")


def plot_line_models(line_root: Path):
    if not line_root.exists():
        return
    models: dict[str, dict[str, list[float]]] = {}
    for arch_dir in sorted(line_root.iterdir()):
        log = arch_dir / "log.csv"
        if log.exists():
            d = _load(log)
            if d:
                models[arch_dir.name] = d

    if not models:
        return

    # Figure out title suffix
    name_label = line_root.name.replace("_", " ").upper()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cm = plt.get_cmap("tab10")
    for i, (name, d) in enumerate(models.items()):
        c = cm(i)
        if "train_loss" in d:
            axes[0].plot(d["epoch"], d["train_loss"], "o-",
                          color=c, label=name)
        # Choose the second-axis metric: CER > TER > word_acc > val_acc
        for key in ("val_cer_pct", "val_ter_pct", "val_word_acc", "val_acc"):
            if key in d:
                axes[1].plot(d["epoch"], d[key], "o-",
                              color=c, label=name)
                metric_name = key
                break

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train CTC / CE loss")
    axes[0].set_title(f"{name_label} — training loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation metric (%)")
    axes[1].set_title(f"{name_label} — validation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = line_root / "curves.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"wrote {out}")


def main():
    char_log = Path("benchmark/results/amadi_charrec/log.csv")
    if char_log.exists():
        plot_charrec(char_log,
                     Path("benchmark/results/amadi_charrec/curves.png"))

    plot_line_models(Path("benchmark/results/amadi_lineocr"))
    plot_line_models(Path("benchmark/results/icfhr_d_balinese"))


if __name__ == "__main__":
    main()
