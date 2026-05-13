"""
Token-level n-gram language model with stupid-backoff smoothing.

Built for the Mathilakam corpus where "tokens" are syllables
(`na`, `ka`, `tha`, ...) rather than individual Unicode characters.
Mathematically identical to a character n-gram LM; we just chose a
larger atomic unit because that's how the labels are tokenised.

Usage:
    from crnn.lm import NGramLM
    lm = NGramLM.from_transcripts(["na/ka/li", "ru/ma/ne"], n=5)
    score = lm.score_token("ka", history=("na",))   # log P(ka | na)
    score = lm.score_seq(["na", "ka", "li"])         # sum of log probs
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Iterable, Sequence


class NGramLM:
    BOS = "<s>"
    EOS = "</s>"
    BACKOFF_ALPHA = 0.4    # Stupid-backoff multiplier per missing-context level

    def __init__(self, n: int = 5):
        self.n = n
        self.counts: list[Counter] = [Counter() for _ in range(n)]
        self.context_counts: list[Counter] = [Counter() for _ in range(n)]
        self.vocab: set[str] = set()
        self.total_unigrams: int = 0

    # ---- training ---------------------------------------------------

    def fit(self, sequences: Iterable[Sequence[str]]) -> None:
        """Build n-gram counts from token sequences."""
        for toks in sequences:
            toks = [self.BOS] * (self.n - 1) + list(toks) + [self.EOS]
            self.vocab.update(t for t in toks
                               if t != self.BOS and t != self.EOS)
            for k in range(1, self.n + 1):
                for i in range(len(toks) - k + 1):
                    ctx = tuple(toks[i:i + k - 1])
                    word = toks[i + k - 1]
                    self.counts[k - 1][ctx + (word,)] += 1
                    self.context_counts[k - 1][ctx]   += 1
        self.total_unigrams = sum(self.counts[0].values())

    # ---- scoring ----------------------------------------------------

    def score_token(self, token: str, history: tuple[str, ...] = ()) -> float:
        """log P(token | history) under stupid-backoff."""
        # Truncate history to (n-1) most recent tokens.
        h = history[-(self.n - 1):] if self.n > 1 else ()
        # Pad short history with BOS so the n-gram math stays consistent.
        if len(h) < self.n - 1:
            h = (self.BOS,) * (self.n - 1 - len(h)) + h

        for level in range(self.n, 0, -1):
            ctx = h[self.n - level:] if level > 1 else ()
            ngram = ctx + (token,)
            num   = self.counts[level - 1].get(ngram, 0)
            den   = self.context_counts[level - 1].get(ctx, 0)
            if num > 0 and den > 0:
                p = num / den
                # Each backoff step costs a factor of BACKOFF_ALPHA.
                penalty = (self.n - level) * math.log(self.BACKOFF_ALPHA)
                return math.log(p) + penalty
        # Fully unseen: uniform fallback over the (training) vocabulary.
        v = max(1, len(self.vocab))
        # Stack one more backoff penalty for going off the vocabulary.
        penalty = self.n * math.log(self.BACKOFF_ALPHA)
        return -math.log(v) + penalty

    def score_seq(self, tokens: Sequence[str]) -> float:
        """log P(sequence) summed across positions, including EOS."""
        s = 0.0
        h: tuple[str, ...] = ()
        for tok in list(tokens) + [self.EOS]:
            s += self.score_token(tok, h)
            h = (h + (tok,))[-(self.n - 1):]
        return s

    # ---- IO ---------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "n": self.n,
                "counts": [dict(c) for c in self.counts],
                "context_counts": [dict(c) for c in self.context_counts],
                "vocab": sorted(self.vocab),
                "total_unigrams": self.total_unigrams,
            }, f)

    @classmethod
    def load(cls, path: str) -> "NGramLM":
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(n=d["n"])
        m.counts = [Counter(c) for c in d["counts"]]
        m.context_counts = [Counter(c) for c in d["context_counts"]]
        m.vocab = set(d["vocab"])
        m.total_unigrams = d["total_unigrams"]
        return m

    @classmethod
    def from_transcripts(cls, transcripts: Iterable[str],
                          n: int = 5,
                          token_split: str = "/") -> "NGramLM":
        """Convenience: parse slash-delimited transcripts and fit."""
        seqs: list[list[str]] = []
        for line in transcripts:
            toks = []
            for raw in line.split(token_split):
                t = raw.strip().lower()
                if t and t != "[unk]":
                    toks.append(t)
            if toks:
                seqs.append(toks)
        m = cls(n=n)
        m.fit(seqs)
        return m
