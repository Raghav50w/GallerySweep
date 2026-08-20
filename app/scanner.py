"""Recursive walk + background scan, with one module-level progress state.

Only one scan runs at a time, so there is no job registry -- just `STATE`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from app import db
from app.config import CACHE_DIR, CANDIDATE_HAMMING_GATE, SUPPORTED_EXTENSIONS
from app.hashing import hash_file
from app.matching import find_candidate_pairs, hashes_to_array

log = logging.getLogger(__name__)

MAX_REPORTED_ERRORS = 50


@dataclass
class ScanState:
    # idle | walking | hashing | matching | embedding | done | error
    phase: str = "idle"
    done: int = 0
    total: int = 0
    errors: int = 0
    error_files: list[str] = field(default_factory=list)
    folder: str | None = None
    mode: str = "fast"
    message: str = ""
    image_count: int = 0
    candidate_pairs: int = 0
    cached_hits: int = 0
    embedded: int = 0
    scored_pairs: int = 0
    elapsed: float = 0.0

    def reset(self) -> None:
        """Restore every field to its default. The state object is reused, not
        replaced, so anything holding a reference keeps seeing live values."""
        self.__init__()  # type: ignore[misc]

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "done": self.done,
            "total": self.total,
            "errors": self.errors,
            "error_files": self.error_files[:MAX_REPORTED_ERRORS],
            "folder": self.folder,
            "mode": self.mode,
            "message": self.message,
            "image_count": self.image_count,
            "candidate_pairs": self.candidate_pairs,
            "cached_hits": self.cached_hits,
            "embedded": self.embedded,
            "scored_pairs": self.scored_pairs,
            "elapsed": round(self.elapsed, 2),
        }


STATE = ScanState()
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None


def is_running() -> bool:
    return STATE.phase in {"walking", "hashing", "matching", "embedding"}


def iter_images(folder: Path) -> Iterator[Path]:
    """Every supported image under `folder`, recursively.

    Recursive by default is the point: the same photo in `Folder1` and
    `Folder7` has to land in one group.
    """
    cache = str(CACHE_DIR.resolve()).lower()
    for dirpath, dirnames, filenames in os.walk(folder):
        here = str(Path(dirpath).resolve()).lower()
        if here == cache or here.startswith(cache + os.sep):
            dirnames[:] = []  # never re-scan our own thumbnail cache
            continue
        dirnames.sort()
        for name in sorted(filenames):
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                yield Path(dirpath) / name


def start_scan(folder: Path, mode: str = "fast") -> None:
    """Kick off a background scan. Raises RuntimeError if one is already going."""
    global _THREAD
    with _LOCK:
        if is_running():
            raise RuntimeError("a scan is already running")
        STATE.reset()
        STATE.phase = "walking"
        STATE.folder = str(folder)
        STATE.mode = mode
        _THREAD = threading.Thread(target=_run, args=(folder, mode), daemon=True)
        _THREAD.start()


def _record_error(path: Path, exc: Exception) -> None:
    STATE.errors += 1
    if len(STATE.error_files) < MAX_REPORTED_ERRORS:
        STATE.error_files.append(f"{path.name}: {exc}")
    log.warning("skipping %s: %s", path, exc)


def _run(folder: Path, mode: str) -> None:
    started = time.perf_counter()
    conn = db.connect()
    try:
        db.init_db(conn)
        pruned = db.prune_missing(conn)
        if pruned:
            log.info("pruned %d rows for files that no longer exist", pruned)

        files = list(iter_images(folder))
        STATE.total = len(files)
        STATE.phase = "hashing"

        ids: list[int] = []
        for path in files:
            try:
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns)
                cached = db.lookup(conn, *key)
                if cached is not None:
                    ids.append(cached)
                    STATE.cached_hits += 1
                else:
                    # Real backups hold truncated downloads and PNGs named .jpg.
                    # One bad file must not kill a 5,000-image scan.
                    hashes, width, height = hash_file(path)
                    ids.append(
                        db.upsert_image(
                            conn,
                            path=str(path),
                            size=stat.st_size,
                            mtime=stat.st_mtime_ns,
                            width=width,
                            height=height,
                            hashes=hashes,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - any decode failure is skippable
                _record_error(path, exc)
            finally:
                STATE.done += 1
        conn.commit()

        STATE.phase = "matching"
        STATE.image_count = len(ids)
        _build_pairs(conn, ids)

        if mode == "smart":
            STATE.phase = "embedding"
            _score_pairs(conn)

        STATE.elapsed = time.perf_counter() - started
        STATE.phase = "done"
        STATE.message = (
            f"{STATE.image_count} images, {STATE.candidate_pairs} candidate pairs, "
            f"{STATE.elapsed:.1f}s"
        )
        if mode == "smart":
            STATE.message += f", {STATE.embedded} embedded"
        log.info("scan finished: %s", STATE.message)
    except Exception as exc:  # noqa: BLE001 - surface it instead of dying silently
        STATE.phase = "error"
        STATE.message = str(exc)
        log.exception("scan failed")
    finally:
        conn.close()


def _build_pairs(conn, ids: list[int]) -> None:
    """Hash-stage candidate generation, written to `pairs` from scratch."""
    ordered = sorted(set(ids))
    rows = db.images_by_id(conn, ordered)
    ordered = [i for i in ordered if i in rows]
    hashes = hashes_to_array(
        [(rows[i]["h0"], rows[i]["h90"], rows[i]["h180"], rows[i]["h270"]) for i in ordered]
    )

    candidates = find_candidate_pairs(hashes, CANDIDATE_HAMMING_GATE)
    STATE.candidate_pairs = int(candidates.shape[0])

    # Index order follows sorted ids, and find_candidate_pairs only emits a < b,
    # so image_a < image_b holds without another sort.
    db.replace_pairs(
        conn,
        (
            (ordered[int(a)], ordered[int(b)], int(hamming), int(rot_a), int(rot_b))
            for a, b, hamming, rot_a, rot_b in candidates
        ),
    )
    log.info(
        "candidate gate <= %d bits produced %d pairs from %d images",
        CANDIDATE_HAMMING_GATE,
        STATE.candidate_pairs,
        len(ordered),
    )


def _score_pairs(conn) -> None:
    """Smart mode: re-rank the hash stage's candidates with ResNet-50 cosine.

    Imported here rather than at module scope so Fast mode never pays torch's
    import cost.
    """
    from app.embeddings import embed_images
    from app.matching import cosine_for_pairs, embedding_work_set

    rows = db.load_pairs(conn, CANDIDATE_HAMMING_GATE)
    pairs = [
        (r["image_a"], r["image_b"], r["rotation_a"], r["rotation_b"]) for r in rows
    ]
    if not pairs:
        return

    needed = embedding_work_set(pairs)
    paths = db.images_by_id(conn, sorted({image_id for image_id, _ in needed}))
    work = [
        (image_id, paths[image_id]["path"], rotation)
        for image_id, rotation in needed
        if image_id in paths
    ]

    STATE.done = 0
    STATE.total = len(work)

    def report(count: int, error: str | None) -> None:
        STATE.done += count
        if error is not None:
            STATE.errors += 1
            if len(STATE.error_files) < MAX_REPORTED_ERRORS:
                STATE.error_files.append(error)

    vectors = embed_images(work, progress=report)
    STATE.embedded = len(vectors)

    scores = cosine_for_pairs(pairs, vectors)
    db.update_cosines(conn, scores)
    STATE.scored_pairs = sum(1 for score, _, _ in scores if score is not None)

    # The cascade's whole justification, logged where it can be checked: forward
    # passes actually run versus one per image in the library.
    log.info(
        "embedded %d of %d images (%.1fx less CNN work than embedding everything)"
        " and scored %d of %d candidate pairs",
        len(work),
        STATE.image_count,
        (STATE.image_count / len(work)) if work else 1.0,
        STATE.scored_pairs,
        len(pairs),
    )
