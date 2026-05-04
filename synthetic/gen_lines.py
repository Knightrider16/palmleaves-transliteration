"""
Generate synthetic line strips for CRNN/CTC pretraining.

A line consists of N tokens drawn from the empirical token distribution
of the real labels CSV (favoring frequent tokens to match the real
manuscript distribution).  Each line is rendered as one wide horizontal
binary strip with degradation applied.

Usage:
    python -m synthetic.gen_lines                       # default 5000 lines
    python -m synthetic.gen_lines --n-lines 20000

Outputs:
    data/synthetic/lines/img_<i>.png
    data/synthetic/lines/index.csv   (filename, transcript)
"""
from __future__ import annotations
import os
import csv
import argparse
import numpy as np
from tqdm import tqdm
import cv2

from .tokens import (
    load_vocab_from_csv, renderable_tokens, build_synthetic_vocab,
    romanize_to_malayalam,
)
from .glyph_renderer import render_line, DEFAULT_FONT
from .degradation import degrade_line


def _build_token_pool(real_csv: str) -> tuple[list[str], np.ndarray]:
    """
    Build a sampling distribution over renderable tokens.
    Real-label frequency is up-weighted; everything else gets a flat tail.
    """
    real_counts = load_vocab_from_csv(real_csv)
    vocab       = build_synthetic_vocab(real_csv)
    rendlist    = [t for t, _ in renderable_tokens(vocab)]

    weights = []
    for t in rendlist:
        # Real tokens weighted by sqrt(count)+5; unseen tokens get 1.
        weights.append(np.sqrt(real_counts.get(t, 0)) * 3 + 1.0)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    return rendlist, w


def _make_transcript(tokens_pool: list[str], probs: np.ndarray,
                     n_tokens: int, rng: np.random.Generator) -> list[str]:
    return list(rng.choice(tokens_pool, size=n_tokens, p=probs))


def _to_malayalam(tokens: list[str]) -> str:
    parts = [romanize_to_malayalam(t) or "" for t in tokens]
    # Light spacing keeps the renderer from mashing conjuncts together
    return " ".join(p for p in parts if p)


def generate(real_csv: str,
             out_dir: str,
             n_lines: int = 5000,
             min_tokens: int = 8,
             max_tokens: int = 28,
             height: int = 64,
             seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    pool, probs = _build_token_pool(real_csv)
    print(f"Sampling pool: {len(pool)} tokens")

    rows = []
    for i in tqdm(range(n_lines), desc="lines"):
        n = int(rng.integers(min_tokens, max_tokens + 1))
        toks = _make_transcript(pool, probs, n, rng)
        text = _to_malayalam(toks)
        if not text:
            continue
        try:
            img = render_line(text, font_path=DEFAULT_FONT,
                              height=height, font_size=int(height * 0.75),
                              rng=rng)
        except Exception:
            continue
        img = degrade_line(img, rng)

        fname = f"img_{i:06d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        rows.append({"filename": fname,
                     "transcript": "/".join(toks)})

    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "transcript"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nGenerated {len(rows)} synthetic lines")
    print(f"Index : {index_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv",   default="data/labels/labels.csv")
    ap.add_argument("--out-dir",    default="data/synthetic/lines")
    ap.add_argument("--n-lines",    type=int, default=5000)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=28)
    ap.add_argument("--height",     type=int, default=64)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()
    generate(args.real_csv, args.out_dir,
             args.n_lines, args.min_tokens, args.max_tokens,
             args.height, args.seed)
