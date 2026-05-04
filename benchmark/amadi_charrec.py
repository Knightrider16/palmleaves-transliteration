"""
AMADI Challenge 3 — Isolated Balinese Character Recognition benchmark.

Trains the same GlyphClassifier architecture used for Malayalam (in
crnn/pretrain_cnn.py) on the AMADI Balinese training set, evaluates on
the official test set, and reports top-1 + top-5 accuracy.

This is a script-agnostic test of the CNN backbone — same architecture,
same training recipe, different alphabet.  A high accuracy here would
mean the framework's character-recognition stage transfers cleanly to
other palm-leaf scripts; a low accuracy would mean we have an
architecture/recipe issue that the Malayalam-only test couldn't surface.

Dataset layout (after the download in benchmark/amadi/):
    c3/train/Challenge-3-ForTrain/train_image/<class_id>_<idx>.jpg
    c3/test/Challenge-3-ForTest/test_image_random/<id>_test.jpg
    c3/eval/Challenge-3-ForEvaluation/list_class_name.txt
    c3/eval/Challenge-3-ForEvaluation/GT_test_image_random.txt

Usage:
    python -m benchmark.amadi_charrec --epochs 12 --batch 128 --lr 0.05

Outputs:
    models/amadi_balinese/glyph_classifier.pth
    benchmark/results/amadi_charrec/log.csv
    benchmark/results/amadi_charrec/predictions.csv
    benchmark/results/amadi_charrec/REPORT.md
"""
from __future__ import annotations
import argparse
import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from crnn.pretrain_cnn import GlyphClassifier

from benchmark._augment import augment_char
from benchmark._checkpoint import (
    save_last, load_last, save_best, append_log, trim_log_to,
)


AMADI_ROOT = Path("benchmark/amadi/c3")
TRAIN_DIR  = AMADI_ROOT / "train" / "Challenge-3-ForTrain" / "train_image"
TEST_DIR   = AMADI_ROOT / "test"  / "Challenge-3-ForTest"  / "test_image_random"
EVAL_DIR   = AMADI_ROOT / "eval"  / "Challenge-3-ForEvaluation"
OUT_DIR    = Path("benchmark/results/amadi_charrec")
MODEL_DIR  = Path("models/amadi_balinese")


def _load_classes() -> list[str]:
    with open(EVAL_DIR / "list_class_name.txt", "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_test_labels() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(EVAL_DIR / "GT_test_image_random.txt", "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split(";")
            if len(parts) >= 2 and parts[0]:
                out[parts[0]] = parts[1]
    return out


class TrainDataset(Dataset):
    """
    Train images are named like <CLASS_NAME>_<idx>.jpg where CLASS_NAME
    is one of the strings in list_class_name.txt (e.g. "KA", "GANTUNGAN-MA").
    A handful of numeric-prefixed files exist that don't match any class;
    those are skipped.
    """
    PAT = re.compile(r"^(.+)_(\d+)\.(jpg|png|jpeg|bmp)$", re.IGNORECASE)

    def __init__(self, root: Path, label_map: dict[str, int],
                 size: int = 64, augment: bool = False):
        self.root = root
        self.size = size
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []
        skipped: dict[str, int] = {}
        for f in sorted(root.iterdir()):
            m = self.PAT.match(f.name)
            if not m:
                continue
            name = m.group(1)
            cls = label_map.get(name)
            if cls is None:
                skipped[name] = skipped.get(name, 0) + 1
                continue
            self.samples.append((f, cls))
        if skipped:
            top = sorted(skipped.items(), key=lambda kv: -kv[1])[:5]
            print(f"  Skipped {sum(skipped.values())} files with unknown "
                  f"prefixes (top: {top})")

    def __len__(self):
        return len(self.samples)

    def _read(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((self.size, self.size), dtype=np.uint8)
        # Some Balinese crops are dark on light, others light on dark.
        # Normalise to white-on-black to match our Malayalam pipeline.
        if np.mean(img) > 127:
            img = 255 - img
        # Tight crop on foreground
        ys, xs = np.where(img > 64)
        if len(xs) > 4 and len(ys) > 4:
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape
        side = max(h, w)
        pad = np.zeros((side, side), dtype=np.uint8)
        pad[(side - h) // 2:(side - h) // 2 + h,
            (side - w) // 2:(side - w) // 2 + w] = img
        return cv2.resize(pad, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def __getitem__(self, idx: int):
        path, cls = self.samples[idx]
        img = self._read(path)
        if self.augment:
            img = augment_char(img)
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0, cls


class TestDataset(Dataset):
    """
    Test images are named like <id>_test.jpg.  Labels come from
    GT_test_image_random.txt (filename;class_name).
    """
    def __init__(self, root: Path, label_map: dict[str, int],
                 gt: dict[str, str], size: int = 64):
        self.root = root
        self.size = size
        self.samples: list[tuple[Path, int, str]] = []
        for f in sorted(root.iterdir()):
            cls = gt.get(f.name)
            if cls is None or cls not in label_map:
                continue
            self.samples.append((f, label_map[cls], f.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cls, name = self.samples[idx]
        # Reuse the train-time normalisation
        ds = TrainDataset.__new__(TrainDataset)
        ds.size = self.size
        img = ds._read(path)
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0, cls, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int,   default=12)
    ap.add_argument("--batch",  type=int,   default=128)
    ap.add_argument("--lr",     type=float, default=0.05)
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--reset",  action="store_true",
                    help="ignore last.pth and start training from epoch 1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = _load_classes()
    n_classes = len(classes)
    print(f"Classes: {n_classes}  (e.g. {classes[:10]})")

    label_map = {name: i for i, name in enumerate(classes)}
    gt = _load_test_labels()

    full = TrainDataset(TRAIN_DIR, label_map, augment=False)
    print(f"Train images: {len(full)}")
    n_val = max(200, len(full) // 20)
    train, val = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train.dataset.augment = True
    train_loader = DataLoader(train, batch_size=args.batch,
                               shuffle=True, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=args.batch,
                               shuffle=False, num_workers=0)

    test_ds = TestDataset(TEST_DIR, label_map, gt)
    print(f"Test images: {len(test_ds)}")
    test_loader = DataLoader(test_ds, batch_size=args.batch,
                              shuffle=False, num_workers=0)

    model = GlyphClassifier(num_classes=n_classes).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[args.epochs // 2, args.epochs * 3 // 4],
        gamma=0.1)
    crit = nn.CrossEntropyLoss()

    log_csv   = OUT_DIR / "log.csv"
    last_path = MODEL_DIR / "last.pth"
    best_path = MODEL_DIR / "glyph_classifier.pth"

    if args.reset and last_path.exists():
        last_path.unlink()
        if log_csv.exists():
            log_csv.unlink()

    state = load_last(last_path, model, optim, sched, device)
    if state is not None:
        start_ep = state["epoch"] + 1
        best_val = state["best_metric"]
        trim_log_to(log_csv, state["epoch"])
        print(f"Resumed from epoch {state['epoch']}  "
              f"(best_val={best_val:.2f}%)")
    else:
        start_ep = 1
        best_val = -1.0

    if start_ep > args.epochs:
        print(f"Already at target epochs ({args.epochs}), skipping training")
    else:
        for ep in range(start_ep, args.epochs + 1):
            model.train()
            run_loss, n = 0.0, 0
            for x, y in tqdm(train_loader, desc=f"ep {ep}/{args.epochs}"):
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                optim.zero_grad(); loss.backward(); optim.step()
                run_loss += loss.item(); n += 1
            sched.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device); y = y.to(device)
                    pred = model(x).argmax(-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            val_acc = correct / max(1, total) * 100
            avg = run_loss / max(1, n)
            append_log(log_csv,
                       ["epoch", "train_loss", "val_acc"],
                       [ep, avg, val_acc])
            print(f"  ep {ep}: train_loss={avg:.3f}  val_acc={val_acc:.2f}%")
            if val_acc > best_val:
                best_val = val_acc
                save_best(best_path, model, extra={"classes": classes})
            save_last(last_path, model, optim, sched, ep, best_val,
                      extra={"classes": classes})

    # ----- Final test eval -----
    print("\n=== Test set evaluation ===")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    correct1, correct5, total = 0, 0, 0
    pred_rows = []
    with torch.no_grad():
        for x, y, names in test_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            top5  = logits.topk(5, dim=-1).indices
            top1  = top5[:, 0]
            for i in range(x.size(0)):
                gold_idx  = int(y[i])
                pred_idx  = int(top1[i])
                top5_idx  = top5[i].tolist()
                pred_rows.append({
                    "filename":   names[i],
                    "gold_class": classes[gold_idx],
                    "pred_top1":  classes[pred_idx],
                    "pred_top5":  ",".join(classes[k] for k in top5_idx),
                    "correct":    int(pred_idx == gold_idx),
                })
            correct1 += (top1 == y).sum().item()
            correct5 += (top5 == y.unsqueeze(-1)).any(-1).sum().item()
            total    += y.numel()

    test_top1 = correct1 / max(1, total) * 100
    test_top5 = correct5 / max(1, total) * 100
    print(f"Test top-1: {test_top1:.2f}%   ({correct1}/{total})")
    print(f"Test top-5: {test_top5:.2f}%   ({correct5}/{total})")

    with open(OUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "filename", "gold_class", "pred_top1", "pred_top5", "correct"])
        w.writeheader(); w.writerows(pred_rows)

    md = [
        "# AMADI Challenge 3 — Balinese Character Recognition",
        "",
        "Benchmark of the project's `GlyphClassifier` architecture on the "
        "official ICFHR 2016 isolated-character recognition dataset.",
        "",
        "## Setup",
        "",
        f"- Architecture: `crnn.pretrain_cnn.GlyphClassifier` (8-block CNN + "
        f"GlobalAvgPool + Linear, ~9.5 M params)",
        f"- Train images: **{len(full):,}**",
        f"- Validation split: {len(val):,}",
        f"- Test images:  **{len(test_ds):,}**",
        f"- Classes:      **{n_classes}** Balinese characters",
        f"- Optimizer: SGD lr={args.lr}, momentum=0.9, MultiStep schedule",
        f"- Augmentation: shared `benchmark._augment.augment_char` (affine, "
        f"elastic, stroke jitter, erasing)",
        f"- Epochs: {args.epochs}, batch {args.batch}",
        "",
        "## Headline numbers",
        "",
        f"- **Test top-1 accuracy: {test_top1:.2f}%**  ({correct1:,} / {total:,})",
        f"- **Test top-5 accuracy: {test_top5:.2f}%**  ({correct5:,} / {total:,})",
        f"- Best val accuracy during training: {best_val:.2f}%",
        "",
        "## Comparison with Malayanma project's same architecture",
        "",
        "Same `GlyphClassifier` architecture trained on synthetic Malayalam "
        "glyphs reaches ~98% val accuracy on its own synthetic test "
        "split (see `crnn/pretrain_cnn.py`).  This benchmark uses the "
        "*same architecture and recipe* but on real palm-leaf data with a "
        "different script — so the gap between the two numbers reflects "
        "the synthetic→real domain gap and the script difference.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m benchmark.amadi_charrec --epochs 12 --batch 128 --lr 0.05",
        "```",
        "",
        "Outputs:",
        f"- `{MODEL_DIR / 'glyph_classifier.pth'}` (best model on val)",
        f"- `{log_csv}` per-epoch loss/val-accuracy",
        f"- `{OUT_DIR / 'predictions.csv'}` per-test-image predictions",
        "",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote: {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
