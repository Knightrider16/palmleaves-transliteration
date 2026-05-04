"""
Generate a synthetic character dataset for pretraining a glyph classifier.

Usage:
    python -m synthetic.gen_chars                # default settings
    python -m synthetic.gen_chars --per-token 50

Outputs:
    data/synthetic/chars/<token>/<token>_<i>.png
    data/synthetic/chars/index.csv      (filename, token)
    data/synthetic/chars/vocab.txt      (one token per line)
"""
from __future__ import annotations
import os
import csv
import argparse
import numpy as np
from tqdm import tqdm
import cv2

from .tokens import build_synthetic_vocab, renderable_tokens
from .glyph_renderer import GlyphRenderer
from .degradation import degrade_char


def generate(real_csv: str,
             out_dir: str,
             per_token: int = 60,
             out_size: int = 64,
             seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    vocab = build_synthetic_vocab(real_csv)
    pairs = renderable_tokens(vocab)
    print(f"Vocab: {len(vocab)} tokens, renderable: {len(pairs)}")

    # Save vocab so the CRNN training script can build its char map
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for t in vocab:
            f.write(t + "\n")

    renderer = GlyphRenderer(out_size=out_size)
    index_rows = []

    for tok, glyph in tqdm(pairs, desc="rendering"):
        tok_dir = os.path.join(out_dir, tok)
        os.makedirs(tok_dir, exist_ok=True)
        for i in range(per_token):
            base = renderer.render(glyph, jitter=True, rng=rng)
            img  = degrade_char(base, rng)
            fname = f"{tok}_{i:05d}.png"
            cv2.imwrite(os.path.join(tok_dir, fname), img)
            index_rows.append({"filename": f"{tok}/{fname}", "token": tok})

    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "token"])
        w.writeheader()
        w.writerows(index_rows)

    print(f"\nGenerated {len(index_rows)} synthetic char images")
    print(f"Index : {index_path}")
    print(f"Dir   : {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv",   default="data/labels/labels.csv")
    ap.add_argument("--out-dir",    default="data/synthetic/chars")
    ap.add_argument("--per-token",  type=int, default=60)
    ap.add_argument("--out-size",   type=int, default=64)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()
    generate(args.real_csv, args.out_dir,
             args.per_token, args.out_size, args.seed)
