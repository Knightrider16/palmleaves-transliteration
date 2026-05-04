"""
Train every registered model sequentially, with full resume support.

Design:
    A simple state file (RUN_STATE.json) tracks which models are
    "pending" / "in_progress" / "done".  Each training call saves
    per-epoch checkpoints, so killing the process and re-running
    `python run_all.py` continues:
      - the in-progress model resumes from its last checkpoint
      - then any pending models follow
      - already-done models are skipped

Usage:
    python run_all.py                    # train all
    python run_all.py --models crnn_ctc,cnn_ctc
    python run_all.py --epochs 8         # override per-model epochs
    python run_all.py --reset            # forget all state, start over
    python run_all.py --status           # just print state, no training
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "RUN_STATE.json"
PY = sys.executable


DEFAULT_MODELS = [
    "cnn_ctc",
    "crnn_ctc",
    "vit_ctc",
    "conformer",
    "crnn_attn",
    "trocr",
]


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def train_one(model_name: str, args, resume: bool) -> int:
    cmd = [
        PY, "-m", "crnn.train_v2",
        "--model", model_name,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
    ]
    if args.cnn_init and Path(args.cnn_init).is_file():
        cmd += ["--cnn-init", args.cnn_init]
    if args.train_csv:
        cmd += ["--train-csv", args.train_csv]
    if args.train_dir:
        cmd += ["--train-dir", args.train_dir]
    if resume:
        cmd += ["--resume"]
    print("\n" + "=" * 70)
    print(f"  Training: {model_name}  resume={resume}")
    print("  $ " + " ".join(cmd))
    print("=" * 70)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.call(cmd, env=env)


def print_status(state: dict):
    print("\nRUN STATE")
    print("-" * 50)
    if not state:
        print("  (no runs yet)")
        return
    for name, info in state.items():
        status = info.get("status", "?")
        last_ep = info.get("last_epoch", "-")
        ter = info.get("best_ter", None)
        ter_s = f"{ter * 100:.1f}%" if isinstance(ter, (int, float)) else "-"
        print(f"  {name:12s}  status={status:11s}  "
              f"last_epoch={last_ep}  best_TER={ter_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model names to run")
    ap.add_argument("--epochs",   type=int,   default=12)
    ap.add_argument("--batch",    type=int,   default=16)
    ap.add_argument("--lr",       type=float, default=0.01)
    ap.add_argument("--cnn-init", default="models/cnn_backbone.pth")
    ap.add_argument("--train-csv", default="")
    ap.add_argument("--train-dir", default="")
    ap.add_argument("--reset",    action="store_true",
                    help="forget any state and re-train from scratch")
    ap.add_argument("--status",   action="store_true",
                    help="print state and exit")
    args = ap.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("Reset: cleared run state")

    state = load_state()

    if args.status:
        print_status(state)
        return

    # Pull in best_ter and last_epoch from per-arch logs if available
    for name in DEFAULT_MODELS:
        log = Path("models") / name / "log.csv"
        if log.exists():
            with open(log, "r", encoding="utf-8") as f:
                rows = list(f.readlines())[1:]
            if rows:
                ter_vals = []
                for r in rows:
                    parts = r.strip().split(",")
                    if len(parts) >= 3:
                        try:
                            ter_vals.append(float(parts[2]))
                        except ValueError:
                            pass
                if ter_vals:
                    state.setdefault(name, {})
                    state[name]["last_epoch"] = len(rows)
                    state[name]["best_ter"]   = min(ter_vals) / 100.0

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    print_status(state)
    print(f"\nRequested: {requested}")

    for name in requested:
        info = state.get(name, {})
        if info.get("status") == "done":
            print(f"\n[skip] {name} already done")
            continue
        info["status"] = "in_progress"
        state[name] = info
        save_state(state)

        last_ckpt = Path("models") / name / "last.pth"
        rc = train_one(name, args, resume=last_ckpt.exists())

        if rc == 0:
            info["status"] = "done"
        else:
            info["status"] = "failed"
            info["last_rc"] = rc
            print(f"\n  ✗ {name} exited with code {rc}")
        state[name] = info
        save_state(state)

    print_status(state)
    print("\nrun_all complete")


if __name__ == "__main__":
    main()
