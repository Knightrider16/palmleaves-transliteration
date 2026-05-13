"""
Mathilakam (Malayanma) phase runner.

Trains every registered LineRecognizer on the in-house labelled corpus
(`data/real_lines/index.csv`), one phase at a time, with per-phase output
isolated under `benchmark/results/mathilakam/<phase>/`.

Each call invokes `crnn.train_v2` as a subprocess so resume + per-epoch
checkpointing already-baked-in keep working unchanged.

After the whole sweep finishes, the best val TER for each arch is
collected into `benchmark/results/mathilakam/<phase>_results.csv` so
phases can be compared at a glance.

Usage:
    python -m benchmark.run_mathilakam --phase 1
    python -m benchmark.run_mathilakam --phase 1 --models conformer,cnn_ctc
    python -m benchmark.run_mathilakam --phase 1 --resume
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from crnn.models import REGISTRY


PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "benchmark" / "results" / "mathilakam"

DEFAULT_MODELS = ["conformer", "cnn_ctc", "crnn_ctc",
                  "crnn_attn", "vit_ctc", "trocr"]

# Phase-specific defaults. Phase 1 = recipe transfer (50 ep + augment_word).
# Phase 2 = synth+real combined training with a stable real-only val.
PHASE_DEFAULTS = {
    1: {
        "epochs":    50,
        "batch":     8,
        "lr":        0.01,
        "train_csv": "data/real_lines/index.csv",
        "train_dir": "data/real_lines",
        "vocab":     "data/real_lines/vocab.txt",
        "cnn_init":  "models/cnn_backbone.pth",
        "val_csv":   "",
        "val_dir":   "",
    },
    2: {
        "epochs":    50,
        "batch":     16,
        "lr":        0.01,
        # Stack synthetic lines + the real Mathilakam train slice.
        "train_csv": "data/synthetic/lines/index.csv,data/real_lines/index_train.csv",
        "train_dir": "data/synthetic/lines,data/real_lines",
        "vocab":     "data/real_lines/vocab_combined.txt",
        "cnn_init":  "models/cnn_backbone.pth",
        # Pin val to the held-out real Mathilakam slice for direct
        # cross-phase comparison.
        "val_csv":   "data/real_lines/index_val.csv",
        "val_dir":   "data/real_lines",
    },
    3: {
        "epochs":    50,
        "batch":     16,
        "lr":        0.01,
        # Phase 3: synth + stitched (3000+3000) + real_train upsampled
        # 50x. Real becomes ~36% of the training pool to break the
        # synth-distribution mode collapse seen in Phase 2.
        "train_csv": ("data/synthetic/lines/index.csv,"
                       "data/synthetic/real_stitched_lines/index.csv,"
                       "data/real_lines/index_train_upsampled.csv"),
        "train_dir": ("data/synthetic/lines,"
                       "data/synthetic/real_stitched_lines,"
                       "data/real_lines"),
        "vocab":     "data/real_lines/vocab_v3.txt",
        "cnn_init":  "models/cnn_backbone.pth",
        "val_csv":   "data/real_lines/index_val.csv",
        "val_dir":   "data/real_lines",
    },
}


def _last_best_ter(log_csv: Path) -> float | None:
    if not log_csv.is_file():
        return None
    best = None
    with open(log_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row.get("val_ter_pct") or row.get("val_ter") or "")
            except ValueError:
                continue
            if best is None or t < best:
                best = t
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",  type=int, choices=PHASE_DEFAULTS, required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated arch names")
    ap.add_argument("--resume", action="store_true",
                    help="resume any in-progress arch from its last.pth")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override default epoch count")
    args = ap.parse_args()

    cfg = PHASE_DEFAULTS[args.phase].copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    out_dir = RESULTS / f"phase{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)

    archs = [m.strip() for m in args.models.split(",") if m.strip()]
    archs = [a for a in archs if a in REGISTRY]

    summaries: list[dict] = []

    for arch in archs:
        arch_dir = out_dir / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        log_csv = arch_dir / "log.csv"

        cmd = [
            sys.executable, "-m", "crnn.train_v2",
            "--model",     arch,
            "--vocab",     cfg["vocab"],
            "--train-csv", cfg["train_csv"],
            "--train-dir", cfg["train_dir"],
            "--epochs",    str(cfg["epochs"]),
            "--batch",     str(cfg["batch"]),
            "--lr",        str(cfg["lr"]),
            "--out-dir",   str(out_dir),
        ]
        if cfg.get("val_csv") and cfg.get("val_dir"):
            cmd += ["--val-csv", cfg["val_csv"], "--val-dir", cfg["val_dir"]]
        if cfg["cnn_init"] and Path(cfg["cnn_init"]).is_file():
            cmd += ["--cnn-init", cfg["cnn_init"]]
        if args.resume:
            cmd += ["--resume"]

        print("\n" + "=" * 70)
        print(f"  Phase {args.phase}  arch={arch}")
        print(f"  $ {' '.join(cmd)}")
        print("=" * 70)
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        rc = subprocess.call(cmd, env=env, cwd=str(PROJECT))
        if rc != 0:
            print(f"  [warn] {arch} returned exit code {rc}")

        best = _last_best_ter(log_csv)
        summaries.append({
            "phase":    args.phase,
            "arch":     arch,
            "epochs":   cfg["epochs"],
            "best_val_ter_pct": best if best is not None else "",
            "exit_code": rc,
        })

    # Snapshot results
    out_csv = RESULTS / f"phase{args.phase}_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "phase", "arch", "epochs", "best_val_ter_pct", "exit_code"])
        w.writeheader()
        w.writerows(summaries)
    print(f"\nWrote {out_csv}")

    # Print a quick comparison table
    print("\nPhase {} results:".format(args.phase))
    print(f"{'arch':<14} {'epochs':>8} {'best TER':>10}")
    print("-" * 36)
    for s in sorted(summaries,
                    key=lambda x: (float(x["best_val_ter_pct"])
                                    if x["best_val_ter_pct"] != "" else 1e9)):
        ter = s["best_val_ter_pct"]
        ter_s = f"{ter:.1f}%" if ter != "" else "-"
        print(f"{s['arch']:<14} {s['epochs']:>8} {ter_s:>10}")


if __name__ == "__main__":
    main()
