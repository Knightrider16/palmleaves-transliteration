"""
Pretrain the CRNN's CNN backbone as a character classifier.

Why this exists:
    Training a CRNN+CTC from scratch on a large vocabulary (~471 tokens)
    with only a few thousand lines reliably collapses to predicting
    blank at every timestep.  Giving the convolutional backbone a
    strong shape prior — by training it as a closed-vocabulary glyph
    classifier first — makes the subsequent CTC training tractable.

Pipeline:
    1. Read every char crop in data/synthetic/chars/<token>/*.png
    2. Train a tiny classifier head on top of the same CNN architecture
       used by crnn.model.CRNN.cnn
    3. Save the CNN-only state dict so crnn.train can pick it up

Usage:
    python -m crnn.pretrain_cnn --epochs 8 --batch 128

Output:
    models/cnn_backbone.pth   (just the CNN feature extractor)
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from .model import _conv_block
from .vocab import Vocab


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# Same CNN as crnn.model.CRNN.cnn -- duplicated here so we can train it
# in isolation and save just the backbone weights.
def build_cnn() -> nn.Sequential:
    return nn.Sequential(
        _conv_block(1,   64,  pool=(2, 2)),
        _conv_block(64,  128, pool=(2, 2)),
        _conv_block(128, 256, bn=True),
        _conv_block(256, 256, pool=(2, 2)),
        _conv_block(256, 512, bn=True),
        _conv_block(512, 512, pool=(2, 1)),
        _conv_block(512, 512, pool=(2, 1)),
        _conv_block(512, 512, pool=(2, 1)),
    )


class CharDataset(Dataset):
    def __init__(self, root: str, vocab: Vocab,
                 size: int = 64, augment: bool = False):
        self.root  = root
        self.size  = size
        self.vocab = vocab
        self.augment = augment
        self.samples: list[tuple[str, int]] = []
        idx_path = os.path.join(root, "index.csv")
        with open(idx_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tok = r["token"]
                if tok in vocab.stoi and tok != "<blank>":
                    self.samples.append(
                        (os.path.join(root, r["filename"]),
                         vocab.stoi[tok]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        if img.shape != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size),
                             interpolation=cv2.INTER_AREA)
        if self.augment and np.random.random() < 0.4:
            sigma = np.random.uniform(5, 15)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return x, label


class GlyphClassifier(nn.Module):
    """CNN backbone + global-pool + linear head."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn  = build_cnn()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        f = self.cnn(x)                  # (B, 512, 1, ~1)  for 64-input
        f = self.pool(f).flatten(1)      # (B, 512)
        return self.head(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab",   default="data/synthetic/chars/vocab.txt")
    ap.add_argument("--root",    default="data/synthetic/chars")
    ap.add_argument("--epochs",  type=int,   default=12)
    ap.add_argument("--batch",   type=int,   default=128)
    ap.add_argument("--lr",      type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocab.from_vocab_file(args.vocab)
    print(f"Vocab: {len(vocab)} (incl. blank)")

    full = CharDataset(args.root, vocab)
    print(f"Char samples: {len(full)}")
    n_val   = max(100, len(full) // 20)
    n_train = len(full) - n_val
    train, val = random_split(full, [n_train, n_val],
                              generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train, batch_size=args.batch,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    model = GlyphClassifier(num_classes=len(vocab)).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[max(1, args.epochs // 2),
                           max(2, args.epochs * 3 // 4)], gamma=0.1)
    crit  = nn.CrossEntropyLoss()

    best_acc = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"epoch {ep}/{args.epochs}"):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss   = crit(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            run_loss += loss.item(); n += 1
        sched.step()

        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                pred = model(x).argmax(-1)
                correct += (pred == y).sum().item()
                total   += y.numel()
        acc = correct / max(1, total) * 100
        print(f"  epoch {ep}: train_loss={run_loss/max(1,n):.3f}  val_acc={acc:.1f}%")
        if acc > best_acc:
            best_acc = acc
            cnn_state = {k: v for k, v in model.state_dict().items()
                         if k.startswith("cnn.")}
            torch.save(cnn_state, MODEL_DIR / "cnn_backbone.pth")
            torch.save({"model_state": model.state_dict(),
                        "vocab": vocab.itos},
                       MODEL_DIR / "glyph_classifier.pth")
            print(f"    ✓ saved CNN backbone + classifier (val_acc={acc:.1f}%)")

    print(f"\nDone.  Best val accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
