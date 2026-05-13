"""
Train a 5-gram LM on the Mathilakam transcripts and save it.

Source: data/labels/labels.csv (full 112 lines including the val
holdout; using everything is fine since this is a separate model
fit on text only, no image leakage).

Output: data/labels/mathilakam_5gram.pkl
"""
from __future__ import annotations
import csv
from pathlib import Path

from .lm import NGramLM


SOURCE = Path("data/labels/labels.csv")
OUT    = Path("data/labels/mathilakam_5gram.pkl")
N      = 5


def main() -> None:
    transcripts: list[str] = []
    with open(SOURCE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r.get("transcript") or "").strip()
            if t:
                transcripts.append(t)
    print(f"transcripts: {len(transcripts)}")

    lm = NGramLM.from_transcripts(transcripts, n=N)
    print(f"vocab: {len(lm.vocab)} tokens")
    print(f"unigram total: {lm.total_unigrams}")
    print(f"5-gram contexts seen: {len(lm.counts[N - 1])}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lm.save(str(OUT))
    print(f"wrote {OUT}")

    # Sanity scores
    for sample in [["na", "ka"], ["zha", "na", "na", "ne"],
                    ["xyz", "abc"]]:
        s = lm.score_seq(sample)
        print(f"  log P({'/'.join(sample)}) = {s:.2f}")


if __name__ == "__main__":
    main()
