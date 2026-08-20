"""SQLite cache: two tables, plus the lifecycle rules that keep them honest."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from app.config import CACHE_DIR, DB_PATH, THUMBS_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id     INTEGER PRIMARY KEY,
    path   TEXT    NOT NULL UNIQUE,
    size   INTEGER NOT NULL,
    mtime  INTEGER NOT NULL,          -- st_mtime_ns: integer, so cache hits are exact
    width  INTEGER NOT NULL,
    height INTEGER NOT NULL,
    h0     TEXT NOT NULL,             -- hex, not INTEGER: SQLite ints are signed
    h90    TEXT NOT NULL,
    h180   TEXT NOT NULL,
    h270   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pairs (
    image_a    INTEGER NOT NULL,
    image_b    INTEGER NOT NULL,      -- always image_a < image_b, never equal
    hamming    INTEGER NOT NULL,
    cosine     REAL,                  -- NULL until the Phase 3 CNN fills it in
    rotation_a INTEGER NOT NULL,
    rotation_b INTEGER NOT NULL,
    PRIMARY KEY (image_a, image_b)
) WITHOUT ROWID;
"""

IMAGE_COLUMNS = "id, path, size, mtime, width, height, h0, h90, h180, h270"


def connect() -> sqlite3.Connection:
    """A fresh connection. SQLite objects are not shared across threads."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def prune_missing(conn: sqlite3.Connection) -> int:
    """Drop rows whose file is gone -- photos get moved and deleted outside the app."""
    rows = conn.execute("SELECT id, path FROM images").fetchall()
    missing = [row["id"] for row in rows if not Path(row["path"]).exists()]
    if missing:
        forget_images(conn, missing)
    return len(missing)


def lookup(conn: sqlite3.Connection, path: str, size: int, mtime: int) -> int | None:
    """Cache hit on (path, size, mtime) -> the image id, so hashing is skipped."""
    row = conn.execute(
        "SELECT id FROM images WHERE path = ? AND size = ? AND mtime = ?",
        (path, size, mtime),
    ).fetchone()
    return int(row["id"]) if row else None


def upsert_image(
    conn: sqlite3.Connection,
    *,
    path: str,
    size: int,
    mtime: int,
    width: int,
    height: int,
    hashes: dict[int, str],
) -> int:
    conn.execute(
        """
        INSERT INTO images (path, size, mtime, width, height, h0, h90, h180, h270)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size = excluded.size, mtime = excluded.mtime,
            width = excluded.width, height = excluded.height,
            h0 = excluded.h0, h90 = excluded.h90,
            h180 = excluded.h180, h270 = excluded.h270
        """,
        (path, size, mtime, width, height, hashes[0], hashes[90], hashes[180], hashes[270]),
    )
    row = conn.execute("SELECT id FROM images WHERE path = ?", (path,)).fetchone()
    return int(row["id"])


def images_by_id(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
    ids = list(ids)
    if not ids:
        return {}
    out: dict[int, sqlite3.Row] = {}
    for start in range(0, len(ids), 500):  # stay clear of SQLite's parameter limit
        chunk = ids[start : start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT {IMAGE_COLUMNS} FROM images WHERE id IN ({placeholders})", chunk
        ).fetchall()
        out.update({int(row["id"]): row for row in rows})
    return out


def image_path(conn: sqlite3.Connection, image_id: int) -> str | None:
    row = conn.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
    return row["path"] if row else None


def replace_pairs(conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
    """Wipe and rewrite. Pairs are cheap to recompute (hashes stay cached) and
    they reference ids that shift as files come and go -- stale pairs would
    resurrect deleted photos in the review grid."""
    conn.execute("DELETE FROM pairs")
    conn.executemany(
        """
        INSERT INTO pairs (image_a, image_b, hamming, cosine, rotation_a, rotation_b)
        VALUES (?, ?, ?, NULL, ?, ?)
        """,
        rows,
    )
    conn.commit()


def update_cosines(
    conn: sqlite3.Connection, rows: Iterable[tuple[float | None, int, int]]
) -> None:
    """Fill in the CNN scores for pairs `replace_pairs` inserted as NULL.

    A pure UPDATE, so the candidate set the hash stage chose is never widened by
    the re-ranking stage -- the CNN sorts what it is given, it does not nominate.
    """
    conn.executemany(
        "UPDATE pairs SET cosine = ? WHERE image_a = ? AND image_b = ?", rows
    )
    conn.commit()


def load_pairs(
    conn: sqlite3.Connection, max_hamming: int, min_cosine: float | None = None
) -> list[sqlite3.Row]:
    """Candidate pairs within `max_hamming`, optionally re-ranked by the CNN.

    With `min_cosine` set, pairs the CNN never scored (NULL) drop out: in Smart
    mode a pair that was not embedded has not earned a place, and the fallback
    is to rescan in Fast mode.
    """
    if min_cosine is None:
        return conn.execute(
            """
            SELECT image_a, image_b, hamming, cosine, rotation_a, rotation_b
            FROM pairs WHERE hamming <= ? ORDER BY hamming
            """,
            (max_hamming,),
        ).fetchall()

    return conn.execute(
        """
        SELECT image_a, image_b, hamming, cosine, rotation_a, rotation_b
        FROM pairs
        WHERE hamming <= ? AND cosine IS NOT NULL AND cosine >= ?
        ORDER BY cosine DESC
        """,
        (max_hamming, min_cosine),
    ).fetchall()


def forget_images(conn: sqlite3.Connection, ids: Sequence[int]) -> None:
    """Remove image rows and every pair that references them."""
    ids = list(ids)
    if not ids:
        return
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", chunk)
        conn.execute(
            f"DELETE FROM pairs WHERE image_a IN ({placeholders})"
            f" OR image_b IN ({placeholders})",
            chunk + chunk,
        )
    conn.commit()
