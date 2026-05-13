"""
Build Phase 3 training assets.

Phase 3 mixes three data sources:
    A. data/synthetic/lines/                3000 font-rendered lines
    B. data/synthetic/real_stitched_lines/  3000 real-char crops in synth layouts
    C. data/real_lines/index_train.csv      68  real Mathilakam lines, upsampled 50x

The 50x upsample on real lines makes them ~36% of the training pool so
gradients become real-dominated and mode collapse to the synth token
distribution becomes much harder.

Outputs:
    data/real_lines/index_train_upsampled.csv   real_train repeated 50x
    data/real_lines/vocab_v3.txt                union of synth + stitched + real
"""
from __future__ import annotations
import csv
from pathlib import Path

REAL_TRAIN     = Path("data/real_lines/index_train.csv")
SYNTH_INDEX    = Path("data/synthetic/lines/index.csv")
STITCHED_INDEX = Path("data/synthetic/real_stitched_lines/index.csv")
SYNTH_VOCAB    = Path("data/synthetic/chars/vocab.txt")

OUT_TRAIN_UP   = Path("data/real_lines/index_train_upsampled.csv")
OUT_VOCAB      = Path("data/real_lines/vocab_v3.txt")

UPSAMPLE_X     = 50


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    real_rows = _read_csv(REAL_TRAIN)
    synth_rows    = _read_csv(SYNTH_INDEX)
    stitched_rows = _read_csv(STITCHED_INDEX)
    print(f"real_train:   {len(real_rows)} lines")
    print(f"synth:        {len(synth_rows)} lines")
    print(f"stitched:     {len(stitched_rows)} lines")

    # 1) Upsampled real_train
    if real_rows:
        with open(OUT_TRAIN_UP, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(real_rows[0].keys()))
            w.writeheader()
            for _ in range(UPSAMPLE_X):
                w.writerows(real_rows)
        print(f"wrote {OUT_TRAIN_UP}: {len(real_rows) * UPSAMPLE_X} rows "
              f"(real_train x {UPSAMPLE_X})")

    # 2) Combined vocab
    tokens: set[str] = set()
    for src in [synth_rows, stitched_rows, real_rows]:
        for r in src:
            for raw in (r.get("transcript") or "").split("/"):
                t = raw.strip().lower()
                if t and t != "[unk]":
                    tokens.add(t)
    if SYNTH_VOCAB.is_file():
        with open(SYNTH_VOCAB, "r", encoding="utf-8") as f:
            for ln in f:
                t = ln.strip()
                if t:
                    tokens.add(t)

    out = sorted(tokens)
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        for t in out:
            f.write(t + "\n")
    print(f"wrote {OUT_VOCAB}: {len(out)} tokens")


if __name__ == "__main__":
    main()
