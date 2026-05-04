"""
Embed every char crop in data/characters_named/ using the CNN backbone
that was pretrained on synthetic glyphs.

The CNN's penultimate feature (global-pooled 512-D) becomes the visual
embedding used for re-clustering and pseudo-labeling.

Usage:
    python -m pseudolabel.embed_chars
    python -m pseudolabel.embed_chars --backbone models/cnn_backbone.pth

Outputs:
    data/embeddings/char_embeddings.npy   (N, 512) float32
    data/embeddings/char_filenames.txt    (N filenames, same order)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from crnn.pretrain_cnn import build_cnn


CHAR_DIR = "data/characters_named"
OUT_DIR  = "data/embeddings"


class CharCropDataset(Dataset):
    def __init__(self, root: str, size: int = 64):
        self.root  = root
        self.size  = size
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and f != "character_index.csv"
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.files[idx])
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        if img.shape != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size),
                             interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return x, idx


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = build_cnn()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.cnn(x)
        return self.pool(f).flatten(1)        # (B, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="models/cnn_backbone.pth")
    ap.add_argument("--char-dir", default=CHAR_DIR)
    ap.add_argument("--out-dir",  default=OUT_DIR)
    ap.add_argument("--batch",    type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNEncoder().to(device)
    state = torch.load(args.backbone, map_location=device)
    # State has keys like cnn.0.0.weight; strip nothing, fits as-is
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded backbone from {args.backbone}")

    ds = CharCropDataset(args.char_dir)
    print(f"Encoding {len(ds)} char crops...")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=True)

    embs = np.zeros((len(ds), 512), dtype=np.float32)
    with torch.no_grad():
        for x, idx in tqdm(loader):
            x = x.to(device, non_blocking=True)
            f = model(x).cpu().numpy()
            embs[idx.numpy()] = f

    np.save(os.path.join(args.out_dir, "char_embeddings.npy"), embs)
    with open(os.path.join(args.out_dir, "char_filenames.txt"),
              "w", encoding="utf-8") as f:
        for name in ds.files:
            f.write(name + "\n")
    print(f"Saved: {args.out_dir}/char_embeddings.npy  shape={embs.shape}")
    print(f"Saved: {args.out_dir}/char_filenames.txt   ({len(ds.files)} files)")


if __name__ == "__main__":
    main()
