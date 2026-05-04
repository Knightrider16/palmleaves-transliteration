"""
Token-level vocabulary for the CTC line recognizer.

Index 0 is reserved for the CTC blank symbol.  Real tokens occupy 1..N.
'[unk]' is included as a regular token so the model can predict it.
"""
from __future__ import annotations
import os
from typing import List


BLANK = "<blank>"


class Vocab:
    def __init__(self, tokens: List[str]):
        # Force blank at index 0; remove duplicates, keep order otherwise.
        seen = {BLANK}
        ordered = [BLANK]
        for t in tokens:
            t = t.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            ordered.append(t)
        self.itos = ordered
        self.stoi = {t: i for i, t in enumerate(ordered)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def blank_idx(self) -> int:
        return 0

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids
                if 0 <= i < len(self.itos) and i != 0]

    def ctc_decode(self, ids: List[int]) -> List[str]:
        """Greedy CTC collapse: drop blanks and consecutive repeats."""
        out: List[str] = []
        prev = -1
        for i in ids:
            if i != prev and i != 0:
                if 0 < i < len(self.itos):
                    out.append(self.itos[i])
            prev = i
        return out

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for t in self.itos:
                f.write(t + "\n")

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            tokens = [ln.strip() for ln in f.readlines()]
        # Skip the blank if it was saved (we always re-insert it).
        tokens = [t for t in tokens if t != BLANK]
        return cls(tokens)

    @classmethod
    def from_vocab_file(cls, path: str) -> "Vocab":
        """Load from synthetic/.../vocab.txt produced by gen_chars."""
        with open(path, "r", encoding="utf-8") as f:
            toks = [ln.strip() for ln in f if ln.strip()]
        return cls(toks)


if __name__ == "__main__":
    import sys
    v = Vocab.from_vocab_file(sys.argv[1])
    print(f"Vocab size (incl. blank): {len(v)}")
    print(f"First 10: {v.itos[:10]}")
    print(f"Encoded 'ka/li/la': {v.encode(['ka','li','la'])}")
    print(f"CTC decode test: {v.ctc_decode([0,2,2,0,3,3,3,4])}")
