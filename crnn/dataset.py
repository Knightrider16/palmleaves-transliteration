"""
Dataset + collate for the CRNN line recognizer.

Two CSV-driven datasets:

    LineDataset
      Reads (filename, transcript) pairs from an index CSV plus an
      image directory.  Used for both synthetic and real data.

The collate function pads variable-width images to the batch max width
and concatenates token-id targets in CTC's flat layout.
"""
from __future__ import annotations
import csv
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab
from benchmark._augment import augment_word


class LineDataset(Dataset):
    def __init__(self,
                 index_csv: str,
                 img_dir: str,
                 vocab: Vocab,
                 height: int = 64,
                 max_width: int = 2400,
                 augment: bool = False,
                 image_col: str = "filename",
                 transcript_col: str = "transcript"):
        self.img_dir       = img_dir
        self.height        = height
        self.max_width     = max_width
        self.vocab         = vocab
        self.augment       = augment
        self.image_col     = image_col
        self.transcript_col = transcript_col

        self.rows: list[dict] = []
        with open(index_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get(transcript_col):
                    continue
                self.rows.append(r)

        # Filter rows whose transcripts contain only known tokens
        kept = []
        for r in self.rows:
            toks = self._tokens(r[transcript_col])
            if not toks:
                continue
            if all(t in vocab.stoi for t in toks):
                kept.append(r)
        self.rows = kept

    @staticmethod
    def _tokens(transcript: str) -> List[str]:
        out = []
        for raw in transcript.split("/"):
            t = raw.strip().lower()
            if not t:
                continue
            if t == "[unk]":
                # CTC can't emit unknowns reliably -- drop them at training
                # time.  Inference is unaffected.
                continue
            out.append(t)
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def _load_img(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape
        scale = self.height / h
        new_w = max(8, min(self.max_width, int(w * scale)))
        img = cv2.resize(img, (new_w, self.height), interpolation=cv2.INTER_AREA)
        return img

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Strong augmentation pipeline ported from benchmark/_augment.py.
        # Same transforms (affine + elastic + stroke jitter + horizontal
        # stretch + erasing) that lifted ICFHR D Balinese from 77% → 24%
        # CER. Light gaussian-noise / blur are kept as a final
        # low-probability layer for photometric robustness.
        rng = np.random.default_rng()
        img = augment_word(img, rng=rng)
        if rng.random() < 0.2:
            ksize = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        if rng.random() < 0.2:
            sigma = rng.uniform(3, 10)
            noise = rng.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = self._load_img(os.path.join(self.img_dir, row[self.image_col]))
        if self.augment:
            img = self._augment(img)
        # Normalize to [0, 1]
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, H, W)

        toks = self._tokens(row[self.transcript_col])
        y = torch.tensor(self.vocab.encode(toks), dtype=torch.long)
        return x, y, row[self.image_col]


def collate(batch):
    """
    Pad variable-width images and concatenate flat targets.

    Returns:
        imgs       : (B, 1, H, W_max)
        targets    : (sum_target_lengths,) flat
        in_lengths : (B,) time-steps after backbone (W_i // 8)
        tgt_lens   : (B,)
        names      : list[str]
    """
    xs, ys, names = zip(*batch)
    H = xs[0].shape[1]
    Ws = [t.shape[2] for t in xs]
    W_max = max(Ws)

    imgs = torch.zeros(len(xs), 1, H, W_max)
    for i, t in enumerate(xs):
        imgs[i, :, :, :Ws[i]] = t

    in_lengths = torch.tensor([max(1, w // 8) for w in Ws], dtype=torch.long)
    tgt_lens   = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    targets    = torch.cat(ys, dim=0).long() if ys else torch.zeros(0, dtype=torch.long)

    return imgs, targets, in_lengths, tgt_lens, list(names)
