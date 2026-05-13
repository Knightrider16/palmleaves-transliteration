"""
Unified, resumable trainer for any registered LineRecognizer.

Per-epoch checkpoint format:
    models/<arch>/last.pth     # restartable
    models/<arch>/best.pth     # best val TER
    models/<arch>/log.csv      # epoch / loss / TER

Each checkpoint is a single dict containing model + optimizer + scheduler
+ RNG state + epoch counter, so you can ctrl+C any time, restart, and
pick up from the next epoch.

Examples:
    # CTC pretrain a CRNN model on the synthetic line dataset
    python -m crnn.train_v2 \\
        --model crnn_ctc \\
        --train-csv data/synthetic/lines/index.csv \\
        --train-dir data/synthetic/lines \\
        --epochs 12 --batch 16 --lr 0.01

    # Resume the same run later
    python -m crnn.train_v2 --model crnn_ctc --resume

    # Add a second dataset by stacking two index CSVs (comma-separated)
    python -m crnn.train_v2 \\
        --model crnn_ctc \\
        --train-csv data/synthetic/lines/index.csv,data/synthetic/real_stitched_lines/index.csv \\
        --train-dir data/synthetic/lines,data/synthetic/real_stitched_lines
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import time
from pathlib import Path

import editdistance
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

from .dataset import LineDataset, collate
from .models  import REGISTRY, build
from .vocab   import Vocab


# ─────────────────────────────────────────────────────────────────────
# AR-aware collate that builds padded targets for AR models
# ─────────────────────────────────────────────────────────────────────

def collate_ar(batch):
    imgs, targets, in_lens, tgt_lens, names = collate(batch)
    # Build (B, T_max+1) — fed into embedding tables, so every value
    # must be a valid index.  Pad positions with 0 (the CTC blank, a
    # valid no-op token in the AR models' embedding).  AR models' own
    # compute_loss builds the proper -100-padded label tensor.
    B = imgs.size(0)
    T_max = int(tgt_lens.max().item()) + 1   # +1 for <eos> slot
    padded = torch.zeros((B, T_max), dtype=torch.long)
    offset = 0
    for b in range(B):
        L = int(tgt_lens[b])
        padded[b, :L] = targets[offset:offset + L]
        offset += L
    return imgs, targets, in_lens, tgt_lens, names, padded


# ─────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────

def _ter(model, loader, device, max_batches: int | None = None):
    model.eval()
    total_dist  = 0
    total_chars = 0
    seen        = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            imgs, targets, in_lens, tgt_lens, _names, *rest = batch
            imgs = imgs.to(device)
            out  = model(imgs)
            preds = model.decode(out)
            offset = 0
            for b in range(len(preds)):
                gold = targets[offset:offset + tgt_lens[b]].tolist()
                offset += int(tgt_lens[b])
                gold_toks = [model.vocab.itos[i] for i in gold]
                d = editdistance.eval(preds[b], gold_toks)
                total_dist  += d
                total_chars += max(1, len(gold_toks))
                seen        += 1
    return total_dist / max(1, total_chars), seen


# ─────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────

def _save_ckpt(path: Path, model, optim, sched, epoch: int,
               best_ter: float, args_dict: dict):
    state = {
        "model_state":     model.state_dict(),
        "optim_state":     optim.state_dict(),
        "scheduler_state": sched.state_dict() if sched else None,
        "epoch":           epoch,
        "best_ter":        best_ter,
        "args":            args_dict,
        "vocab":           model.vocab.itos,
        "rng_torch":       torch.get_rng_state(),
        "rng_cuda":        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy":       np.random.get_state(),
        "rng_python":      random.getstate(),
    }
    tmp = path.with_suffix(".pth.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_ckpt(path: Path, model, optim, sched, device):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optim.load_state_dict(ckpt["optim_state"])
    if sched and ckpt.get("scheduler_state"):
        sched.load_state_dict(ckpt["scheduler_state"])
    if ckpt.get("rng_torch") is not None:
        torch.set_rng_state(ckpt["rng_torch"])
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    if ckpt.get("rng_numpy") is not None:
        np.random.set_state(ckpt["rng_numpy"])
    if ckpt.get("rng_python") is not None:
        random.setstate(ckpt["rng_python"])
    return ckpt["epoch"], ckpt.get("best_ter", float("inf"))


# ─────────────────────────────────────────────────────────────────────
# Main train loop
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True, choices=sorted(REGISTRY))
    p.add_argument("--vocab",     default="data/synthetic/chars/vocab.txt")
    p.add_argument("--train-csv", default="data/synthetic/lines/index.csv,data/synthetic/real_stitched_lines/index.csv",
                    help="comma-separated list of index CSVs to concatenate")
    p.add_argument("--train-dir", default="data/synthetic/lines,data/synthetic/real_stitched_lines",
                    help="comma-separated dirs (must align with --train-csv)")
    p.add_argument("--val-csv",   default="",
                    help="optional explicit validation CSV; if absent we "
                         "carve a slice off --train-csv with random_split")
    p.add_argument("--val-dir",   default="",
                    help="image directory paired with --val-csv")
    p.add_argument("--height",    type=int,   default=64)
    p.add_argument("--epochs",    type=int,   default=12)
    p.add_argument("--batch",     type=int,   default=16)
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--cnn-init",  default="",
                    help="optional CNN backbone .pth (only used by "
                         "models with the 8-block CNN)")
    p.add_argument("--resume",    action="store_true",
                    help="resume from models/<model>/last.pth")
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--out-dir",   default="models")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    arch_dir = Path(args.out_dir) / args.model
    arch_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Vocab ----
    vocab = Vocab.from_vocab_file(args.vocab)
    print(f"Vocab: {len(vocab)}")

    # ---- Datasets (possibly multiple) ----
    csv_paths = [s.strip() for s in args.train_csv.split(",") if s.strip()]
    dir_paths = [s.strip() for s in args.train_dir.split(",") if s.strip()]
    assert len(csv_paths) == len(dir_paths), \
        "--train-csv and --train-dir must have the same number of entries"
    parts: list[LineDataset] = []
    for cp, dp in zip(csv_paths, dir_paths):
        if not os.path.isfile(cp):
            print(f"  [skip] {cp} not found")
            continue
        parts.append(LineDataset(cp, dp, vocab=vocab,
                                  height=args.height, augment=True))
        print(f"  {cp}: {len(parts[-1])} lines")
    full = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    n_total = len(full)
    print(f"Total training lines: {n_total}")

    if args.val_csv and args.val_dir and os.path.isfile(args.val_csv):
        # Explicit, stable validation set.
        val_ds = LineDataset(args.val_csv, args.val_dir, vocab=vocab,
                              height=args.height, augment=False)
        train = full
        val   = val_ds
        print(f"Using explicit val set: {args.val_csv} ({len(val)} lines)")
    else:
        n_val = max(1, min(n_total - 1, max(50, n_total // 20)))
        train, val = random_split(
            full, [n_total - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed))
        print(f"Auto val split: {len(val)} lines from training pool")

    # ---- Build model ----
    model = build(args.model, vocab=vocab).to(device)
    is_ar = (model.TYPE == "ar")
    coll = collate_ar if is_ar else collate
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model}  params={n_params:.2f} M  type={model.TYPE}")

    if args.cnn_init and os.path.isfile(args.cnn_init):
        cnn_state = torch.load(args.cnn_init, map_location=device)
        # Some models don't have a `cnn` attribute -- load_state_dict
        # with strict=False simply skips missing keys.
        msg = model.load_state_dict(cnn_state, strict=False)
        n_loaded = sum(1 for k in cnn_state if k.startswith("cnn."))
        print(f"  Loaded {n_loaded} CNN backbone weights")

    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                              collate_fn=coll, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=args.batch, shuffle=False,
                              collate_fn=coll, num_workers=0)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                             nesterov=True, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=[max(1, args.epochs // 2),
                    max(2, args.epochs * 3 // 4)], gamma=0.1)

    start_ep = 1
    best_ter = float("inf")
    last_ckpt = arch_dir / "last.pth"
    log_csv   = arch_dir / "log.csv"

    if args.resume and last_ckpt.exists():
        start_ep, best_ter = _load_ckpt(last_ckpt, model, optim, sched, device)
        start_ep += 1
        print(f"Resuming at epoch {start_ep}, best_ter={best_ter*100:.1f}%")

    if not log_csv.exists() or start_ep == 1:
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_ter_pct", "epoch_seconds"])

    # ---- Loop ----
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"{args.model} ep{ep}/{args.epochs}")
        for batch in pbar:
            if is_ar:
                imgs, targets, in_lens, tgt_lens, _names, padded = batch
                padded = padded.to(device)
            else:
                imgs, targets, in_lens, tgt_lens, _names = batch
                padded = None

            imgs    = imgs.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            if is_ar:
                out = model(imgs, padded)
            else:
                out = model(imgs)
            loss = model.compute_loss(out, targets, in_lens, tgt_lens)
            if not torch.isfinite(loss):
                continue
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            run_loss += loss.item(); n += 1
            pbar.set_postfix(loss=f"{run_loss/max(1,n):.3f}")
        sched.step()

        ter, n_val_seen = _ter(model, val_loader, device, max_batches=50)
        dt = time.time() - t0
        avg_loss = run_loss / max(1, n)
        print(f"  ep {ep}: loss={avg_loss:.3f}  val_TER={ter*100:.1f}%  "
              f"({n_val_seen} val lines)  {dt:.0f}s")

        # Log
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, avg_loss, ter * 100, dt])

        # Always save last checkpoint (resume point)
        _save_ckpt(last_ckpt, model, optim, sched, ep, best_ter,
                   {"model": args.model, "lr": args.lr,
                    "batch": args.batch, "epochs": args.epochs})

        if ter < best_ter:
            best_ter = ter
            _save_ckpt(arch_dir / "best.pth", model, optim, sched, ep,
                       best_ter,
                       {"model": args.model, "lr": args.lr,
                        "batch": args.batch, "epochs": args.epochs})
            print(f"    ✓ best (TER={ter*100:.1f}%)")

    print(f"\n{args.model} done.  Best val TER: {best_ter*100:.1f}%")


if __name__ == "__main__":
    main()
