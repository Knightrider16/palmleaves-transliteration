"""
Sentence-embedding-based semantic search over archive records.

Used by store.search_with_related as the "related entries" stage.
Direct (lexical) hits stay on the dictionary path; semantic search
fills in fuzzy / topical matches that lexical alone would miss.

Model: sentence-transformers/all-MiniLM-L6-v2  (384 dims, ~80 MB)

Behaviour:
    - Model is loaded lazily on first call (a few seconds).
    - Per-archive embeddings are cached in memory; recomputed only
      when the underlying row count changes.
    - nearest(query, k, exclude_ids) returns [(archive_id, score), ...]
      sorted by descending cosine similarity.
"""
from __future__ import annotations
import os

# Quiet HF cache warning on Windows where symlinks are restricted.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None  # lazy: SentenceTransformer instance once loaded
_cache = {
    "ids":        None,  # list[int]
    "embeddings": None,  # np.ndarray (N, 384), L2-normalised
    "row_sig":    None,  # tuple of (id, title, description, tags) for invalidation
}


def _archive_text(row: dict) -> str:
    """Concatenated text used as the embedding input for one archive."""
    parts = [
        row.get("title") or "",
        row.get("description") or "",
        " ".join(row.get("tags") or []),
    ]
    return ". ".join(p for p in parts if p)


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def warm() -> None:
    """Eagerly load the model. Call once at app startup to avoid
    paying the 2–4 second model-load cost inside the first user request."""
    _load_model()


def index(rows: list[dict]) -> None:
    """
    (Re)build the in-memory embedding cache from `rows` (full archive dicts).
    Idempotent: if the rows haven't changed since the last call, this is a no-op.
    """
    sig = tuple(
        (r["id"], r.get("title"), r.get("description"),
         tuple(r.get("tags") or []))
        for r in rows
    )
    if sig == _cache["row_sig"]:
        return
    if not rows:
        _cache["ids"] = []
        _cache["embeddings"] = np.zeros((0, 384), dtype=np.float32)
        _cache["row_sig"] = sig
        return

    model = _load_model()
    texts = [_archive_text(r) for r in rows]
    embs  = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs  = _normalise(embs.astype(np.float32))
    _cache["ids"]        = [r["id"] for r in rows]
    _cache["embeddings"] = embs
    _cache["row_sig"]    = sig


def nearest(query: str, k: int = 5,
            exclude_ids: set[int] | None = None,
            min_score: float = 0.20) -> list[tuple[int, float]]:
    """
    Return the top-`k` archives most similar to `query`, as
    [(archive_id, cosine_score), ...]. Scores are in [-1, 1]; in
    practice for sane queries they sit in [0.2, 0.8].

    `exclude_ids` lets callers drop already-shown direct hits.
    `min_score` filters out near-orthogonal matches (noise floor).
    """
    if _cache["embeddings"] is None or len(_cache["ids"]) == 0:
        return []
    model = _load_model()
    q = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = _normalise(q.astype(np.float32))[0]
    sims = _cache["embeddings"] @ q  # (N,)

    pairs = list(zip(_cache["ids"], sims.tolist()))
    if exclude_ids:
        pairs = [(i, s) for (i, s) in pairs if i not in exclude_ids]
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = [(i, s) for (i, s) in pairs if s >= min_score]
    return pairs[:k]
