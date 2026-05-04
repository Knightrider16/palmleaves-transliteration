"""
Romanization → Malayalam Unicode mapping for synthetic data generation.

The user's transcription scheme uses /-separated romanized akshara tokens
(e.g. "ka/li/la/pe/ri").  Each token is one Malayalam syllable: an optional
consonant cluster + an optional vowel matra.

This module:
  * Reads the real label CSV to get the empirical token vocabulary.
  * Maps each romanized token to a Malayalam Unicode string when possible.
  * Exposes helpers to tokenize / detokenize transcripts.

The mapping is best-effort.  Tokens that cannot be parsed (numerals,
foreign words, [unk]) are kept as-is in the vocabulary but skipped during
synthetic glyph rendering.
"""
from __future__ import annotations
import csv
import os
import re
from collections import Counter
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────
# Romanization scheme used in this project
#
# Consonants are written with optional digraphs.  Each row maps a
# roman prefix → Malayalam consonant.  Longer prefixes are tried first.
# ─────────────────────────────────────────────────────────────────────

CONSONANTS: dict[str, str] = {
    # velars
    "k":   "ക",  "kh":  "ഖ",  "g":  "ഗ",  "gh":  "ഘ",  "ng":  "ങ",
    # palatals
    "ch":  "ച",  "chh": "ഛ",  "j":  "ജ",  "jh":  "ഝ",  "nj":  "ഞ",
    # retroflex
    "t":   "ട",  "tt":  "ഠ",  "d":  "ഡ",  "dd":  "ഢ",  "nn":  "ണ",
    # dentals
    "th":  "ത",  "thh": "ഥ",  "dh": "ദ",  "dhh": "ധ",  "n":   "ന",
    # labials
    "p":   "പ",  "ph":  "ഫ",  "b":  "ബ",  "bh":  "ഭ",  "m":   "മ",
    # approximants / fricatives
    "y":   "യ",  "r":   "ര",  "l":  "ല",  "v":   "വ",
    "sh":  "ശ",  "ssh": "ഷ",  "s":  "സ",  "h":   "ഹ",
    # Malayalam-specific
    "ll":  "ള",  "zh":  "ഴ",  "rr": "റ",  "nnn": "ഩ",
}

VOWEL_MATRAS: dict[str, str] = {
    "a":   "",            # inherent
    "aa":  "ാ",      # ാ
    "i":   "ി",      # ി
    "ii":  "ീ",      # ീ
    "u":   "ു",      # ു
    "uu":  "ൂ",      # ൂ
    "e":   "െ",      # െ
    "ee":  "േ",      # േ
    "ai":  "ൈ",      # ൈ
    "o":   "ൊ",      # ൊ
    "oo":  "ോ",      # ോ
    "au":  "ൌ",      # ൌ
}

# The user's CSV uses some alternative spellings; map to canonical above.
VOWEL_ALIASES: dict[str, str] = {
    "lii": "lii",  # caught by general parser
}

INDEPENDENT_VOWELS: dict[str, str] = {
    "a":  "അ",  "aa": "ആ",  "i":  "ഇ",  "ii": "ഈ",
    "u":  "ഉ",  "uu": "ഊ",  "e":  "എ",  "ee": "ഏ",
    "ai": "ഐ",  "o":  "ഒ",  "oo": "ഓ",  "au": "ഔ",
}

# Special multi-syllable tokens we want to render as fixed compounds
# (these appear in the labels CSV as single tokens despite being conjuncts).
SPECIAL_COMPOUNDS: dict[str, str] = {
    "shree": "ശ്രീ",
    "ndra":  "ന്ദ്ര",
    "ncha":  "ഞ്ച",
    "ntha":  "ന്ത",
    "nda":   "ന്ദ",
    "nthha": "ന്ഥ",
    "thhaa": "ഥാ",
    "thhu":  "ഥു",
    "thhi":  "ഥി",
    "ezhu":  "എഴു",
    "athi":  "അതി",
    "anu":   "അനു",
    "atha":  "അത",
    "iru":   "ഇരു",
    "moo":   "മൂ",
    "vee":   "വീ",
    "rri":   "റി",
    "nnaa":  "ന്നാ",
    "rraa":  "റാ",
}

# Sort consonant keys longest-first for greedy matching.
_CONS_SORTED = sorted(CONSONANTS.keys(), key=len, reverse=True)
_VOW_SORTED  = sorted(VOWEL_MATRAS.keys(), key=len, reverse=True)
_INDV_SORTED = sorted(INDEPENDENT_VOWELS.keys(), key=len, reverse=True)


def romanize_to_malayalam(token: str) -> Optional[str]:
    """
    Convert one romanized token (e.g. 'kaa', 'shree', 'thi') to Malayalam.
    Returns None if the token cannot be parsed.
    """
    t = token.strip().lower()
    if not t:
        return None

    if t in SPECIAL_COMPOUNDS:
        return SPECIAL_COMPOUNDS[t]

    # Pure independent vowel (must match whole token)
    if t in INDEPENDENT_VOWELS:
        return INDEPENDENT_VOWELS[t]

    # Consonant + optional vowel matra
    for c in _CONS_SORTED:
        if t.startswith(c):
            tail = t[len(c):]
            if tail == "":
                # bare consonant -- render with virama (chillu-like)
                return CONSONANTS[c] + "്"
            for v in _VOW_SORTED:
                if tail == v:
                    return CONSONANTS[c] + VOWEL_MATRAS[v]
            # consonant + unknown tail -- still try inherent
            return CONSONANTS[c]

    # Lone numeral / unparseable -- return None
    return None


def tokenize(transcript: str) -> List[str]:
    """Split a /-separated transcript into clean tokens."""
    if not transcript or not isinstance(transcript, str):
        return []
    return [t.strip() for t in transcript.split("/") if t.strip()]


def load_vocab_from_csv(csv_path: str) -> Counter:
    """
    Read every transcript in the labels CSV and return a token frequency
    Counter (excluding [unk]).
    """
    counts: Counter = Counter()
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for tok in tokenize(row.get("transcript", "")):
                if tok != "[unk]":
                    counts[tok] += 1
    return counts


def build_synthetic_vocab(real_csv_path: str,
                          extra_consonants: bool = True) -> List[str]:
    """
    Build the union of:
      - tokens observed in the real labels CSV
      - the full Malayalam syllabary (every consonant × every vowel matra)
        when extra_consonants=True
    Returns a sorted, de-duplicated list of unique tokens.

    Tokens that can't be mapped to Malayalam are still kept so the
    classifier's output vocabulary covers them, but synthetic rendering
    will skip them.
    """
    vocab: set[str] = set()
    real = load_vocab_from_csv(real_csv_path)
    vocab.update(real.keys())

    if extra_consonants:
        for c in CONSONANTS:
            vocab.add(c + "a")        # inherent
            for v in VOWEL_MATRAS:
                if v == "a":
                    continue
                vocab.add(c + v)
        for v in INDEPENDENT_VOWELS:
            vocab.add(v)

    # Always include unk sentinel
    vocab.add("[unk]")
    return sorted(vocab)


def renderable_tokens(vocab: List[str]) -> List[Tuple[str, str]]:
    """Return [(token, malayalam_glyph), ...] for tokens we can render."""
    pairs = []
    for tok in vocab:
        if tok == "[unk]":
            continue
        if not re.match(r"^[a-z\[\]]+$", tok):
            continue
        m = romanize_to_malayalam(tok)
        if m is not None:
            pairs.append((tok, m))
    return pairs


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/labels/labels.csv"
    real_counts = load_vocab_from_csv(csv_path)
    vocab = build_synthetic_vocab(csv_path)
    pairs = renderable_tokens(vocab)

    print(f"Real tokens (unique): {len(real_counts)}")
    print(f"Synthetic vocab     : {len(vocab)}")
    print(f"Renderable          : {len(pairs)}")
    print("\nSample mappings:")
    for tok, ml in pairs[:30]:
        print(f"  {tok:8s} -> {ml}")
    print("\nReal tokens NOT renderable:")
    rs = set(t for t, _ in pairs)
    not_rend = sorted(set(real_counts) - rs - {"[unk]"})
    for t in not_rend[:30]:
        print(f"  {t}  (count={real_counts[t]})")
