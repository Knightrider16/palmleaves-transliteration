"""
Shared resume / checkpoint helpers for benchmark training.

Single-model script (e.g. amadi_charrec):
    last_path = arch_dir / "last.pth"
    best_path = arch_dir / "best.pth"
    state = load_last(last_path, model, optim, sched)
    start_epoch = state["epoch"] + 1 if state else 1
    best_metric = state["best_metric"] if state else float("-inf")
    ...
    for ep in range(start_epoch, args.epochs + 1):
        ...
        save_last(last_path, model, optim, sched, ep, best_metric, extra)

Multi-arch script (e.g. amadi_lineocr, icfhr_d_balinese):
    if is_arch_done(arch_dir): continue
    # else train, with resume from last.pth if it exists

Resume is by default ON (i.e. if `last.pth` exists, we pick up from it).
Pass `--reset` to clear an arch's directory before retraining.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch


def is_arch_done(arch_dir: Path) -> bool:
    """An arch is 'done' iff its summary.json exists (test eval ran)."""
    return (arch_dir / "summary.json").is_file()


def reset_arch(arch_dir: Path) -> None:
    """Wipe an arch's results so the next run starts fresh."""
    if arch_dir.exists():
        shutil.rmtree(arch_dir)


def save_last(path: Path,
              model: torch.nn.Module,
              optim: torch.optim.Optimizer,
              sched: Any,
              epoch: int,
              best_metric: float,
              extra: dict | None = None) -> None:
    """Atomic save of full training state. Use after every epoch."""
    payload = {
        "model_state":  model.state_dict(),
        "optim_state":  optim.state_dict(),
        "sched_state":  sched.state_dict() if sched is not None else None,
        "epoch":        epoch,
        "best_metric":  best_metric,
        "extra":        extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_last(path: Path,
              model: torch.nn.Module,
              optim: torch.optim.Optimizer | None = None,
              sched: Any = None,
              device: torch.device | str = "cpu") -> dict | None:
    """Load full training state into model/optim/sched. Returns the
    dict with epoch/best_metric/extra, or None if no checkpoint exists."""
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optim is not None and ckpt.get("optim_state") is not None:
        optim.load_state_dict(ckpt["optim_state"])
    if sched is not None and ckpt.get("sched_state") is not None:
        sched.load_state_dict(ckpt["sched_state"])
    return {
        "epoch":       ckpt["epoch"],
        "best_metric": ckpt["best_metric"],
        "extra":       ckpt.get("extra", {}),
    }


def save_best(path: Path,
              model: torch.nn.Module,
              extra: dict | None = None) -> None:
    """Save model-only checkpoint for inference. Use when val improves."""
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def write_summary(arch_dir: Path, summary: dict) -> None:
    """Write summary.json atomically. Presence of this file marks 'done'."""
    path = arch_dir / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp.replace(path)


def append_log(log_csv: Path, header: list[str], row: list) -> None:
    """Append a row to log.csv, writing the header if the file is new."""
    import csv
    new = not log_csv.is_file()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def trim_log_to(log_csv: Path, max_epoch: int) -> None:
    """Keep only rows where epoch <= max_epoch. Used after resume to drop
    any partial-epoch row that may have been written before a crash."""
    import csv
    if not log_csv.is_file():
        return
    with open(log_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, data = rows[0], rows[1:]
    kept = [r for r in data if r and r[0].isdigit() and int(r[0]) <= max_epoch]
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
