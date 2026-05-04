"""
Build "real-glyph synthetic" line strips by stitching together
pseudo-labeled real char crops.

The output is a synthetic dataset that looks like the real palm-leaf
domain (since every glyph is sourced from a real char crop) but has
perfect labels (since we control the token sequence).

Pipeline:
    1. Read CNN predictions (data/pseudo_labeled_v3/cnn_predictions.csv)
    2. Keep only high-confidence predictions  (default ≥ 0.8)
    3. Group chars by predicted token
    4. Render N synthetic lines:
        - sample token sequences from the empirical distribution of the
          gold labels CSV
        - for each token, sample a random char crop from that token's
          pool
        - resize each crop so heights match
        - hstack with small random gaps
        - apply line-level degradation
    5. Save (image, transcript) index

Usage:
    python -m pseudolabel.stitch_lines --n-lines 3000

Outputs:
    data/synthetic/real_stitched_lines/img_*.png
    data/synthetic/real_stitched_lines/index.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from synthetic.degradation import degrade_line
from synthetic.tokens import load_vocab_from_csv


def _load_predictions(pred_csv: str, min_conf: float
                      ) -> dict[str, list[str]]:
    """Group filenames by predicted token, keep high-conf only."""
    by_token: dict[str, list[str]] = defaultdict(list)
    with open(pred_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["confidence"]) < min_conf:
                continue
            by_token[r["predicted_token"]].append(r["filename"])
    return by_token


def _render_line(token_seq: list[str],
                 token_to_files: dict[str, list[str]],
                 char_dir: str,
                 height: int,
                 gap_range: tuple[int, int],
                 rng: np.random.Generator) -> tuple[np.ndarray, list[str]] | None:
    """
    Render one line by stitching real char crops in token order.
    Returns (img, kept_tokens) or None if no token has a crop.
    """
    pieces: list[np.ndarray] = []
    kept: list[str] = []
    for tok in token_seq:
        files = token_to_files.get(tok, [])
        if not files:
            continue
        f = rng.choice(files)
        img = cv2.imread(os.path.join(char_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Resize to (height, height * orig_w / orig_h)
        h, w = img.shape
        new_h = height
        new_w = max(8, int(w * new_h / max(h, 1)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pieces.append(img)
        kept.append(tok)
        # Small variable gap
        gap_w = int(rng.integers(gap_range[0], gap_range[1] + 1))
        if gap_w > 0:
            pieces.append(np.zeros((height, gap_w), dtype=np.uint8))

    if not kept:
        return None
    line = np.hstack(pieces)
    return line, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv",  default="data/pseudo_labeled_v3/cnn_predictions.csv")
    ap.add_argument("--char-dir",  default="data/characters_named")
    ap.add_argument("--labels",    default="data/labels/labels.csv")
    ap.add_argument("--out-dir",   default="data/synthetic/real_stitched_lines")
    ap.add_argument("--n-lines",   type=int, default=3000)
    ap.add_argument("--min-conf",  type=float, default=0.8)
    ap.add_argument("--height",    type=int, default=64)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=28)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--no-degrade", action="store_true",
                    help="skip the line-level degradation pipeline "
                         "(real chars already carry their own degradation)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # Build token pool from CNN predictions
    print(f"Loading CNN predictions (min conf {args.min_conf})...")
    token_to_files = _load_predictions(args.pred_csv, args.min_conf)
    print(f"  Tokens with ≥1 high-conf real crop: {len(token_to_files)}")

    # Token frequency from real labels CSV
    real_counts = load_vocab_from_csv(args.labels)
    print(f"  Real tokens: {len(real_counts)}")

    # Sampling distribution: weighted by real frequency
    available_tokens = [t for t in token_to_files.keys() if t in real_counts]
    if not available_tokens:
        # Fall back: any token with ≥1 high-conf crop, even if not in labels
        print("  WARNING: no overlap between CNN-confident tokens and "
              "real labels; using all CNN-confident tokens uniformly")
        available_tokens = list(token_to_files.keys())
        weights = np.ones(len(available_tokens), dtype=np.float64)
    else:
        weights = np.array(
            [np.sqrt(real_counts.get(t, 1)) + 0.5
             for t in available_tokens], dtype=np.float64)
    weights /= weights.sum()
    print(f"  Sampling pool: {len(available_tokens)} tokens")

    rows = []
    for i in tqdm(range(args.n_lines), desc="lines"):
        n = int(rng.integers(args.min_tokens, args.max_tokens + 1))
        seq = list(rng.choice(available_tokens, size=n, p=weights))
        out = _render_line(seq, token_to_files, args.char_dir,
                            args.height, (1, 6), rng)
        if out is None:
            continue
        img, kept = out
        if not args.no_degrade:
            img = degrade_line(img, rng)

        fname = f"img_{i:06d}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname), img)
        rows.append({"filename": fname,
                     "transcript": "/".join(kept)})

    out_csv = os.path.join(args.out_dir, "index.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "transcript"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} stitched lines")
    print(f"Index: {out_csv}")


if __name__ == "__main__":
    main()
