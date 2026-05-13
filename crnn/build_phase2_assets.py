"""
Build the combined synth+real vocab and the real train/val split for
Phase 2 of the Mathilakam runs.

Outputs:
    data/real_lines/vocab_combined.txt      union of synth + real tokens
    data/real_lines/index_train.csv         ~85% of real, stratified by image
    data/real_lines/index_val.csv           ~15% of real, stratified by image

The stratified split makes sure every source image contributes lines
to both train and val so the val metric reflects cross-image
generalisation rather than per-image overfitting.
"""
from __future__ import annotations
import csv
import math
import random
from pathlib import Path

REAL_INDEX  = Path("data/real_lines/index.csv")
SYNTH_VOCAB = Path("data/synthetic/chars/vocab.txt")
OUT_DIR     = Path("data/real_lines")
VAL_FRAC    = 0.15
SEED        = 0


def main() -> None:
    rng = random.Random(SEED)

    # ---- combined vocab ----
    real_tokens: set[str] = set()
    rows: list[dict] = []
    with open(REAL_INDEX, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            for raw in r["transcript"].split("/"):
                t = raw.strip().lower()
                if t and t != "[unk]":
                    real_tokens.add(t)

    synth_tokens: list[str] = []
    if SYNTH_VOCAB.is_file():
        with open(SYNTH_VOCAB, "r", encoding="utf-8") as f:
            synth_tokens = [ln.strip() for ln in f if ln.strip()]

    union = sorted(set(synth_tokens) | real_tokens)
    out_vocab = OUT_DIR / "vocab_combined.txt"
    with open(out_vocab, "w", encoding="utf-8") as f:
        for t in union:
            f.write(t + "\n")
    print(f"vocab_combined: {len(union)} tokens "
          f"(synth={len(synth_tokens)}, real={len(real_tokens)}, "
          f"intersection={len(set(synth_tokens) & real_tokens)})")
    print(f"  wrote {out_vocab}")

    # ---- stratified train/val split by image ----
    by_img: dict[str, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image_id"], []).append(r)

    train_rows: list[dict] = []
    val_rows:   list[dict] = []
    for img, rs in sorted(by_img.items()):
        rs = rs[:]
        rng.shuffle(rs)
        # At least 1 val per image when image has >= 2 rows
        n_val = max(1, math.ceil(len(rs) * VAL_FRAC)) if len(rs) >= 2 else 0
        val_rows.extend(rs[:n_val])
        train_rows.extend(rs[n_val:])
        print(f"  {img}: total={len(rs)} -> train={len(rs)-n_val}, val={n_val}")

    fieldnames = list(rows[0].keys()) if rows else []
    for path, data in [(OUT_DIR / "index_train.csv", train_rows),
                        (OUT_DIR / "index_val.csv",   val_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)
        print(f"  wrote {path}: {len(data)} rows")


if __name__ == "__main__":
    main()
