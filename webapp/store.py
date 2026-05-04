"""SQLite-backed persistence: users, archive records, contact submissions."""
from __future__ import annotations
import hashlib
import os
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "archives.db"


# Domain synonyms used to expand a search query before LIKE-matching.
# Lower-case keys; values are the alternative tokens worth probing.
# Keep this small and curated — broad expansion drowns the signal.
SEARCH_SYNONYMS: dict[str, list[str]] = {
    "temple":      ["religious", "shrine", "endowment", "deity"],
    "religious":   ["temple", "endowment", "shrine"],
    "court":       ["legal", "judicial", "munsiff", "civil"],
    "munsiff":     ["court", "judicial", "civil"],
    "land":        ["revenue", "settlement", "property", "estate"],
    "revenue":     ["land", "settlement", "tax"],
    "manuscript":  ["palm-leaf", "palmleaf", "document", "leaf"],
    "palmleaf":    ["palm-leaf", "manuscript", "leaf"],
    "palm":        ["palm-leaf", "palmleaf", "manuscript"],
    "malayanma":   ["palm-leaf", "manuscript", "script"],
    "inscription": ["epigraph", "stone", "engraved"],
    "administration": ["governance", "office", "register"],
    "register":    ["administration", "record"],
    "record":      ["register", "document", "archive"],
    "archive":     ["record", "collection", "register"],
    "kerala":      ["malabar", "travancore", "cochin"],
    "travancore":  ["kerala", "malabar"],
    "malabar":     ["kerala", "travancore"],
}


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def init_db():
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt     TEXT NOT NULL,
                pw_hash  TEXT NOT NULL,
                role     TEXT DEFAULT 'researcher'
            );

            CREATE TABLE IF NOT EXISTS archives (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                description TEXT,
                year        INTEGER,
                tags        TEXT,
                available   INTEGER DEFAULT 1,
                cover       TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS contacts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                email      TEXT NOT NULL,
                message    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        cur = c.execute("SELECT COUNT(*) AS n FROM users")
        if cur.fetchone()["n"] == 0:
            _seed_default_users(c)
        cur = c.execute("SELECT COUNT(*) AS n FROM archives")
        if cur.fetchone()["n"] == 0:
            _seed_default_archives(c)


def _seed_default_users(c: sqlite3.Connection):
    defaults = [
        ("researcher", "archives2026"),
        ("admin", "admin"),
    ]
    for u, p in defaults:
        salt = secrets.token_hex(8)
        c.execute(
            "INSERT INTO users(username, salt, pw_hash, role) VALUES (?,?,?,?)",
            (u, salt, _hash_pw(p, salt), "researcher" if u != "admin" else "admin"),
        )


def _seed_default_archives(c: sqlite3.Connection):
    now = datetime.utcnow().isoformat(timespec="seconds")
    rows = [
        ("Records of Temple Administration",
         "Manuscript contains details about temple lands, revenue and annual accounts.",
         1882, "Temple,Administration,1882", 1),
        ("Temple Inscriptions — Collection I",
         "Compilation of inscriptional copies from various temples in the Malabar region.",
         1875, "Inscriptions,Temple,Malabar", 1),
        ("Religious Endowments Register",
         "Register mentioning endowments and donations to temples and religious bodies.",
         1890, "Religious,Endowment,1890", 1),
        ("Land Revenue Settlement Notes",
         "Field-level notes from the 1893 land revenue settlement of Travancore.",
         1893, "Land,Revenue,Travancore", 1),
        ("Court Proceedings — Munsiff Court",
         "Selected proceedings of the Munsiff Court covering civil suits from 1901–1904.",
         1904, "Court,Civil,Munsiff", 0),
        ("Palm-leaf Manuscripts — Malayanma Script",
         "Digitized palm-leaf manuscripts written in the Malayanma script, awaiting full transliteration.",
         1798, "Palm-leaf,Malayanma,Manuscript", 1),
    ]
    for title, desc, year, tags, available in rows:
        c.execute(
            """INSERT INTO archives
               (title, description, year, tags, available, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (title, desc, year, tags, available, now),
        )


# ---- auth ------------------------------------------------------------------

def check_password(username: str, password: str) -> bool:
    if not username or not password:
        return False
    with _conn() as c:
        row = c.execute(
            "SELECT salt, pw_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        return False
    return _hash_pw(password, row["salt"]) == row["pw_hash"]


# ---- archives --------------------------------------------------------------

def _row_to_archive(row: sqlite3.Row) -> dict:
    tags = [t for t in (row["tags"] or "").split(",") if t]
    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"] or "",
        "year": row["year"],
        "tags": tags,
        "available": bool(row["available"]),
    }


def recent_archives(limit: int = 4) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM archives ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_archive(r) for r in rows]


def search_archives(query: str) -> list[dict]:
    """Flat search (legacy). Use search_with_related for split + tag-related output."""
    return search_with_related(query)["direct"]


_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "about", "above", "below", "into", "over", "under", "out", "old", "new",
    "any", "all", "some", "most", "more", "less", "very", "just", "than",
    "then", "so", "such", "no", "not", "only", "own", "same", "too",
}


def _tokenize(q: str) -> list[str]:
    """Tokens of length >= 3, lower-cased, with hyphenated phrases preserved.
    Stop words and very short tokens are filtered out to avoid noisy LIKE
    matches (e.g. 'in' matching every row containing 'inscriptions')."""
    raw = re.findall(r"[A-Za-z][A-Za-z\-]+", q.lower())
    return [t for t in raw if len(t) >= 3 and t not in _STOP_WORDS]


def _expand_query(tokens: list[str]) -> set[str]:
    """Expand tokens via domain synonyms. Always includes originals."""
    out: set[str] = set(tokens)
    for t in tokens:
        out.update(SEARCH_SYNONYMS.get(t, []))
    return out


def all_archives() -> list[dict]:
    """Every archive row, used to (re)build the embedding index."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM archives ORDER BY id").fetchall()
    return [_row_to_archive(r) for r in rows]


def get_archive(archive_id: int) -> dict | None:
    """Single archive by id, or None if not found."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM archives WHERE id = ?", (archive_id,)
        ).fetchone()
    return _row_to_archive(row) if row else None


def suggest(query: str, limit: int = 5) -> list[dict]:
    """
    Lightweight autocomplete: titles where any prefix-of-token matches,
    falling back to lexical body matches. Returns minimal dicts:
        [{'id', 'title', 'year', 'tags'}, ...]
    Cheap enough to call on every keystroke.
    """
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []
    like_prefix = f"{q}%"
    like_any    = f"%{q}%"
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, title, year, tags FROM archives
            WHERE LOWER(title) LIKE ?
               OR LOWER(title) LIKE ?
               OR LOWER(description) LIKE ?
               OR LOWER(tags) LIKE ?
            ORDER BY
              CASE WHEN LOWER(title) LIKE ? THEN 0 ELSE 1 END,
              id DESC
            LIMIT ?
            """,
            (like_prefix, like_any, like_any, like_any, like_prefix, limit),
        ).fetchall()
    return [
        {
            "id":    r["id"],
            "title": r["title"],
            "year":  r["year"],
            "tags":  [t for t in (r["tags"] or "").split(",") if t][:3],
        }
        for r in rows
    ]


def search_with_related(query: str, *, semantic: bool = True,
                        related_k: int = 6) -> dict:
    """
    Two-stage search.
        direct:  rows where ANY (token or its synonym) appears in title/desc/tags.
        related: top-k rows by **embedding cosine similarity** to the raw query
                 (excluding direct hits). Falls back to tag-overlap if the
                 embedding model is unavailable / disabled.

    Returns {'direct': [...], 'related': [...], 'tokens': [...],
             'expanded': [...], 'related_mode': 'semantic' | 'tags' | 'none'}.
    """
    tokens = _tokenize(query)
    if not tokens:
        return {"direct": [], "related": [], "tokens": [],
                "expanded": [], "related_mode": "none"}

    expanded = _expand_query(tokens)

    # ---- direct stage: lexical, any expanded token ----
    where_parts, args = [], []
    for term in expanded:
        like = f"%{term}%"
        where_parts.append(
            "(LOWER(title) LIKE ? OR LOWER(description) LIKE ? OR LOWER(tags) LIKE ?)"
        )
        args.extend([like, like, like])
    where_clause = " OR ".join(where_parts)

    with _conn() as c:
        direct_rows = c.execute(
            f"SELECT * FROM archives WHERE {where_clause} ORDER BY id DESC",
            args,
        ).fetchall()

    direct = [_row_to_archive(r) for r in direct_rows]
    direct_ids = {r["id"] for r in direct}

    # ---- related stage: prefer semantic, fall back to tag-overlap ----
    related: list[dict] = []
    related_mode = "none"

    if semantic:
        try:
            from . import embeddings
            embeddings.index(all_archives())
            hits = embeddings.nearest(
                query, k=related_k, exclude_ids=direct_ids
            )
            if hits:
                # Map id -> row, attach the score for the UI.
                with _conn() as c:
                    rows_by_id = {
                        r["id"]: r
                        for r in c.execute(
                            "SELECT * FROM archives WHERE id IN ({})".format(
                                ",".join("?" * len(hits))
                            ),
                            [h[0] for h in hits],
                        ).fetchall()
                    }
                for aid, score in hits:
                    if aid in rows_by_id:
                        a = _row_to_archive(rows_by_id[aid])
                        a["score"] = round(float(score), 3)
                        related.append(a)
                related_mode = "semantic"
        except Exception as e:
            # Don't let an embedding failure break the page.
            import logging
            logging.warning("semantic search disabled: %s", e)

    if not related:
        # Tag-overlap fallback — same logic as before, no scores.
        direct_tags: set[str] = set()
        for r in direct:
            for tag in r["tags"]:
                t = tag.strip().lower()
                if t:
                    direct_tags.add(t)
        if direct_tags:
            tag_where = " OR ".join(["LOWER(tags) LIKE ?"] * len(direct_tags))
            tag_args  = [f"%{t}%" for t in direct_tags]
            with _conn() as c:
                tag_rows = c.execute(
                    f"SELECT * FROM archives WHERE {tag_where} ORDER BY id DESC",
                    tag_args,
                ).fetchall()
            for r in tag_rows:
                if r["id"] not in direct_ids:
                    related.append(_row_to_archive(r))
            if related:
                related_mode = "tags"

    return {
        "direct":       direct,
        "related":      related,
        "tokens":       tokens,
        "expanded":     sorted(expanded - set(tokens)),
        "related_mode": related_mode,
    }


# ---- stats ----------------------------------------------------------------

def stats() -> dict:
    """Headline counts shown on the dashboard."""
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) AS n FROM archives").fetchone()["n"]
        avail = c.execute(
            "SELECT COUNT(*) AS n FROM archives WHERE available=1"
        ).fetchone()["n"]
        years = c.execute(
            "SELECT MIN(year) AS y0, MAX(year) AS y1 FROM archives "
            "WHERE year IS NOT NULL"
        ).fetchone()
        contact_n = c.execute("SELECT COUNT(*) AS n FROM contacts").fetchone()["n"]
    return {
        "total":     total,
        "available": avail,
        "year_min":  years["y0"],
        "year_max":  years["y1"],
        "messages":  contact_n,
    }


# ---- contacts --------------------------------------------------------------

def record_contact(name: str, email: str, message: str):
    with _conn() as c:
        c.execute(
            """INSERT INTO contacts (name, email, message, created_at)
               VALUES (?, ?, ?, ?)""",
            (name, email, message, datetime.utcnow().isoformat(timespec="seconds")),
        )
