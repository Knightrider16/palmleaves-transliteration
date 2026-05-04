"""
Train each of the 6 registered line-recognizer architectures on AMADI
Challenge 3, treating every isolated character as a "line" with a
target sequence of length 1.

This is an apples-to-apples cross-architecture comparison on a public
benchmark with real ground truth.  Each model uses the same:
    - 133-class Balinese vocabulary (+ <blank> = 134 total)
    - 11710 train images / 7673 test images (with stratified val split)
    - SGD lr=0.01, momentum=0.9, MultiStep schedule
    - Same input pipeline (height=64, aspect-preserving resize)

Per-model output:
    benchmark/results/amadi_lineocr/<arch>/
        last.pth, best.pth, log.csv, predictions.csv, summary.json

A unified comparison.csv is written at the end.

Usage:
    python -m benchmark.amadi_lineocr --epochs 12 --batch 32 --lr 0.01
    python -m benchmark.amadi_lineocr --models conformer,vit_ctc
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from crnn.models import REGISTRY, build
from crnn.train_v2 import collate_ar
from crnn.dataset import collate
from crnn.vocab import Vocab

from benchmark._augment import augment_char
from benchmark._checkpoint import (
    is_arch_done, reset_arch,
    save_last, load_last, save_best, write_summary,
    append_log, trim_log_to,
)

import re

AMADI_ROOT = Path("benchmark/amadi/c3")
TRAIN_DIR  = AMADI_ROOT / "train" / "Challenge-3-ForTrain" / "train_image"
TEST_DIR   = AMADI_ROOT / "test"  / "Challenge-3-ForTest"  / "test_image_random"
EVAL_DIR   = AMADI_ROOT / "eval"  / "Challenge-3-ForEvaluation"
OUT_ROOT   = Path("benchmark/results/amadi_lineocr")
MODEL_ROOT = Path("models/amadi_lineocr")

NAME_PAT = re.compile(r"^(.+)_(\d+)\.(jpg|png|jpeg|bmp)$", re.IGNORECASE)


def _load_classes() -> list[str]:
    with open(EVAL_DIR / "list_class_name.txt", "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_test_labels() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(EVAL_DIR / "GT_test_image_random.txt", "r",
              encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split(";")
            if len(parts) >= 2 and parts[0]:
                out[parts[0]] = parts[1]
    return out


def _read_normalised(path: Path, size: int = 64) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((size, size), dtype=np.uint8)
    if np.mean(img) > 127:
        img = 255 - img
    ys, xs = np.where(img > 64)
    if len(xs) > 4 and len(ys) > 4:
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape
    side = max(h, w)
    pad = np.zeros((side, side), dtype=np.uint8)
    pad[(side - h) // 2:(side - h) // 2 + h,
        (side - w) // 2:(side - w) // 2 + w] = img
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)


class AmadiLineDataset(Dataset):
    """Returns each char as a "line" of width=64 with a single-token target."""
    def __init__(self, samples: list[tuple[Path, int]], vocab: Vocab,
                 size: int = 64, augment: bool = False):
        self.samples = samples
        self.size    = size
        self.vocab   = vocab
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cls_idx = self.samples[idx]
        img = _read_normalised(path, self.size)
        if self.augment:
            img = augment_char(img)
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0    # (1, H, W)
        # Target: single class index (offset by 1 because vocab[0] = <blank>)
        y = torch.tensor([cls_idx + 1], dtype=torch.long)
        return x, y, str(path.name)


def _build_samples(label_map: dict[str, int]
                   ) -> list[tuple[Path, int]]:
    out = []
    for f in sorted(TRAIN_DIR.iterdir()):
        m = NAME_PAT.match(f.name)
        if not m:
            continue
        cls = label_map.get(m.group(1))
        if cls is not None:
            out.append((f, cls))
    return out


def _build_test_samples(label_map: dict[str, int],
                        gt: dict[str, str]) -> list[tuple[Path, int]]:
    out = []
    for f in sorted(TEST_DIR.iterdir()):
        cls = gt.get(f.name)
        if cls is None or cls not in label_map:
            continue
        out.append((f, label_map[cls]))
    return out


def _ter(model, loader, device, vocab):
    model.eval()
    correct, total = 0, 0
    edit_dist, gold_chars = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs, targets, in_lens, tgt_lens, _names, *rest = batch
            imgs = imgs.to(device)
            out = model(imgs)
            preds = model.decode(out)
            offset = 0
            for b in range(len(preds)):
                gold_idx = int(targets[offset:offset + tgt_lens[b]].tolist()[0])
                offset += int(tgt_lens[b])
                gold_tok = vocab.itos[gold_idx]
                pred_first = preds[b][0] if preds[b] else ""
                if pred_first == gold_tok:
                    correct += 1
                total += 1
                edit_dist  += editdistance.eval(preds[b], [gold_tok])
                gold_chars += 1
    acc = correct / max(1, total) * 100
    ter = edit_dist / max(1, gold_chars) * 100
    return acc, ter


def _train_one_model(name: str, vocab: Vocab, train_loader, val_loader,
                      test_loader, device, args, classes: list[str]):
    arch_dir  = OUT_ROOT / name
    arch_dir.mkdir(parents=True, exist_ok=True)
    log_csv   = arch_dir / "log.csv"
    last_path = arch_dir / "last.pth"
    best_path = arch_dir / "best.pth"

    model = build(name, vocab=vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    is_ar = (model.TYPE == "ar")
    print(f"\n=== {name} === ({n_params:.2f} M, type={model.TYPE})")

    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, nesterov=True, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=[max(1, args.epochs // 2),
                    max(2, args.epochs * 3 // 4)], gamma=0.1)

    state = load_last(last_path, model, optim, sched, device)
    if state is not None:
        start_ep = state["epoch"] + 1
        best_acc = state["best_metric"]
        trim_log_to(log_csv, state["epoch"])
        print(f"  resumed from epoch {state['epoch']}  "
              f"(best_val_acc={best_acc:.2f}%)")
    else:
        start_ep = 1
        best_acc = -1.0

    if start_ep > args.epochs:
        print(f"  already at target epochs ({args.epochs}), skipping training")
    else:
        for ep in range(start_ep, args.epochs + 1):
            model.train()
            t0 = time.time()
            run_loss, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"{name} ep{ep}/{args.epochs}")
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
                out = model(imgs, padded) if is_ar else model(imgs)
                loss = model.compute_loss(out, targets, in_lens, tgt_lens)
                if not torch.isfinite(loss):
                    continue
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
                run_loss += loss.item(); n += 1
                pbar.set_postfix(loss=f"{run_loss/max(1,n):.3f}")
            sched.step()

            val_acc, val_ter = _ter(model, val_loader, device, vocab)
            dt = time.time() - t0
            avg_loss = run_loss / max(1, n)
            print(f"  ep {ep}: loss={avg_loss:.3f}  val_acc={val_acc:.2f}%  "
                  f"val_TER={val_ter:.2f}%  {dt:.0f}s")
            append_log(
                log_csv,
                ["epoch", "train_loss", "val_acc", "val_ter_pct",
                 "epoch_seconds"],
                [ep, avg_loss, val_acc, val_ter, dt],
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_best(best_path, model, extra={
                    "vocab":   vocab.itos,
                    "classes": classes,
                    "config":  {"name": name, "lr": args.lr,
                                "batch": args.batch,
                                "epochs": args.epochs},
                })
            save_last(last_path, model, optim, sched, ep, best_acc,
                      extra={"name": name})

    # Final test eval
    print(f"  ── test eval ──")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_acc, test_ter = _ter(model, test_loader, device, vocab)
    print(f"  test_acc={test_acc:.2f}%  test_TER={test_ter:.2f}%")

    summary = {
        "model":        name,
        "params_M":     n_params,
        "type":         model.TYPE,
        "best_val_acc": best_acc,
        "test_acc":     test_acc,
        "test_ter":     test_ter,
        "epochs":       args.epochs,
        "batch":        args.batch,
    }
    write_summary(arch_dir, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",  default=",".join(sorted(REGISTRY)),
                    help="comma-separated model names")
    ap.add_argument("--epochs",  type=int,   default=12)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=0.01)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--reset",   action="store_true",
                    help="wipe per-arch results before training (no resume)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = _load_classes()
    label_map = {n: i for i, n in enumerate(classes)}
    n_classes = len(classes)
    print(f"Classes: {n_classes}")

    # Vocab: <blank> at 0, then class names
    vocab = Vocab(classes)
    print(f"Vocab (incl. blank): {len(vocab)}")

    train_samples = _build_samples(label_map)
    test_samples  = _build_test_samples(label_map, _load_test_labels())
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    full_train = AmadiLineDataset(train_samples, vocab, augment=True)
    test_ds    = AmadiLineDataset(test_samples,  vocab, augment=False)

    n_val = max(200, len(full_train) // 20)
    train, val = random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    summaries = []
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in requested:
        if name not in REGISTRY:
            print(f"[skip] unknown model {name}")
            continue

        arch_dir = OUT_ROOT / name
        if args.reset:
            reset_arch(arch_dir)
        if is_arch_done(arch_dir):
            print(f"[skip] {name} already done (summary.json present)")
            with open(arch_dir / "summary.json", "r", encoding="utf-8") as f:
                summaries.append(json.load(f))
            continue

        is_ar = (REGISTRY[name].TYPE == "ar")
        coll = collate_ar if is_ar else collate
        train_loader = DataLoader(train, batch_size=args.batch,
                                   shuffle=True, num_workers=0,
                                   collate_fn=coll)
        val_loader   = DataLoader(val,   batch_size=args.batch,
                                   shuffle=False, num_workers=0,
                                   collate_fn=coll)
        test_loader  = DataLoader(test_ds, batch_size=args.batch,
                                   shuffle=False, num_workers=0,
                                   collate_fn=coll)
        try:
            s = _train_one_model(name, vocab, train_loader, val_loader,
                                  test_loader, device, args, classes)
            summaries.append(s)
        except KeyboardInterrupt:
            print(f"\n  Interrupted during {name}. "
                  f"Re-run to resume from last.pth.")
            raise

    if summaries:
        cmp_csv = OUT_ROOT / "comparison.csv"
        with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "type", "params_M", "best_val_acc",
                "test_acc", "test_ter", "epochs", "batch"])
            w.writeheader(); w.writerows(summaries)
        print(f"\nWrote: {cmp_csv}")


if __name__ == "__main__":
    main()
