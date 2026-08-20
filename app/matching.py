"""Hamming search over rotation hashes, connected-component grouping, keeper pick."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from app.config import DISTANCE_CHUNK_BYTES, HASH_BITS, ROTATIONS
from app.embeddings import cosine

log = logging.getLogger(__name__)

_ROTATION_DEGREES = np.array(ROTATIONS, dtype=np.int64)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Set bits per element. NumPy >= 2 has this natively; the LUT is a fallback."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x)
    bytes_view = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))
    return _POPCOUNT_LUT[bytes_view].sum(axis=-1, dtype=np.uint8)


def hashes_to_array(rows: Iterable[Sequence[str]]) -> np.ndarray:
    """(n, 4) uint64 array from rows of 16-char hex hashes, in rotation order."""
    values = [[int(h, 16) for h in row] for row in rows]
    if not values:
        return np.zeros((0, len(ROTATIONS)), dtype=np.uint64)
    return np.array(values, dtype=np.uint64)


def hamming_to_similarity(hamming: float) -> float:
    """Normalize a Hamming distance to 0-1, higher = more similar."""
    return max(0.0, 1.0 - float(hamming) / HASH_BITS)


def find_candidate_pairs(hashes: np.ndarray, gate: int) -> np.ndarray:
    """Every pair within `gate` bits, as (a, b, hamming, rotation_a, rotation_b).

    Each image's rotation-0 hash is compared against all four rotations of every
    other image, so a photo matches its own 90-degree copy via `A.h0 == B.h90`.
    Picking a per-image canonical rotation (e.g. min over the four) would be a
    discrete choice that JPEG noise can flip, leaving two copies of one photo
    with different canonical forms that never match.

    Only `a < b` is computed. That loses nothing: the relation is symmetric, so
    if A is B rotated by k, then B is A rotated by -k and one of the two
    directions is always in the upper triangle. It also gives exactly one row
    per pair with `image_a < image_b`, and excludes self-pairs -- a
    near-symmetric photo would otherwise match its own 180-degree rotation.
    """
    n = int(hashes.shape[0])
    empty = np.zeros((0, 5), dtype=np.int64)
    if n < 2:
        return empty

    # Peak allocation is the (chunk, n, 4) uint64 xor buffer; hold it flat.
    chunk = max(1, int(DISTANCE_CHUNK_BYTES // (n * len(ROTATIONS) * 8)))
    index = hashes[np.newaxis, :, :]
    columns = np.arange(n)[np.newaxis, :]
    found: list[np.ndarray] = []

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        queries = hashes[start:stop, 0][:, np.newaxis, np.newaxis]
        distances = _popcount(queries ^ index)  # (c, n, 4)
        best = distances.min(axis=2)
        best_rotation = distances.argmin(axis=2)

        mask = (best <= gate) & (columns > np.arange(start, stop)[:, np.newaxis])
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            continue
        found.append(
            np.stack(
                [
                    (start + rows).astype(np.int64),
                    cols.astype(np.int64),
                    best[rows, cols].astype(np.int64),
                    np.zeros(rows.size, dtype=np.int64),  # rotation_a is always 0
                    _ROTATION_DEGREES[best_rotation[rows, cols]],
                ],
                axis=1,
            )
        )

    return np.concatenate(found, axis=0) if found else empty


def embedding_work_set(
    pairs: Iterable[Sequence[int]],
) -> list[tuple[int, int]]:
    """The distinct (image_id, rotation) the CNN must embed to score `pairs`.

    Rows are `(image_a, image_b, rotation_a, rotation_b)`. `rotation_a` is
    always 0 (see `find_candidate_pairs`), but one image can be the `a` of one
    pair and the rotated `b` of another, so it legitimately appears twice at two
    orientations. Collapsing that to one forward pass per image would compare an
    upright photo against a sideways one and quietly lose the match the hash
    stage already found.
    """
    needed: set[tuple[int, int]] = set()
    for image_a, image_b, rotation_a, rotation_b in pairs:
        needed.add((int(image_a), int(rotation_a)))
        needed.add((int(image_b), int(rotation_b)))
    return sorted(needed)


def cosine_for_pairs(
    pairs: Iterable[Sequence[int]],
    vectors: dict[tuple[int, int], np.ndarray],
) -> list[tuple[float | None, int, int]]:
    """Score each pair, as `(cosine, image_a, image_b)` ready for the DB update.

    A pair whose image the CNN could not read scores `None` and stays NULL, so
    it falls back to hash-only behaviour instead of disappearing from review.
    """
    out: list[tuple[float | None, int, int]] = []
    for image_a, image_b, rotation_a, rotation_b in pairs:
        left = vectors.get((int(image_a), int(rotation_a)))
        right = vectors.get((int(image_b), int(rotation_b)))
        score = None if left is None or right is None else cosine(left, right)
        out.append((score, int(image_a), int(image_b)))
    return out


def group_pairs(
    node_count: int, edges: Iterable[tuple[int, int]]
) -> list[list[int]]:
    """Connected components over the matched pairs, keeping groups of 2 or more.

    Components chain: A~B and B~C puts all three together even when A and C look
    nothing alike. At a loose threshold that can swallow half a library into one
    group. The largest-group warning below is the tripwire for a threshold set
    too loose; it is a known trade-off of graph grouping, not a bug to engineer
    around.
    """
    rows: list[int] = []
    cols: list[int] = []
    for a, b in edges:
        rows.append(a)
        cols.append(b)
    if not rows or node_count == 0:
        return []

    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(node_count, node_count),
    )
    _, labels = connected_components(graph, directed=False)

    buckets: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[int(label)].append(index)

    groups = [members for members in buckets.values() if len(members) > 1]
    if groups:
        largest = max(len(members) for members in groups)
        if largest > max(20, node_count // 10):
            log.warning(
                "largest group has %d of %d images - threshold is probably too loose",
                largest,
                node_count,
            )
    return groups


def assemble_groups(
    pairs: Iterable[tuple[int, int, float]], images: dict[int, dict]
) -> list[dict]:
    """Turn scored pairs into review-ready groups.

    Groups are never stored -- they are derived from `pairs` at whatever
    threshold is being asked for, which is why moving the slider re-groups
    instantly with no rescan.
    """
    scored: dict[tuple[int, int], float] = {}
    for a, b, similarity in pairs:
        if a not in images or b not in images:
            continue  # a file deleted since the scan
        key = (min(a, b), max(a, b))
        scored[key] = max(scored.get(key, 0.0), float(similarity))
    if not scored:
        return []

    ids = sorted({image_id for key in scored for image_id in key})
    position = {image_id: index for index, image_id in enumerate(ids)}
    edges = [(position[a], position[b]) for a, b in scored]

    out: list[dict] = []
    for component in group_pairs(len(ids), edges):
        members = [ids[index] for index in component]
        items = [images[image_id] for image_id in members]
        keeper = members[pick_keeper(items)]

        files = []
        for image_id in members:
            item = images[image_id]
            if image_id == keeper:
                similarity = 1.0
            else:
                # Direct similarity to the keeper when there is a pair for it;
                # transitively-joined members fall back to their best link.
                similarity = scored.get(
                    (min(image_id, keeper), max(image_id, keeper)),
                    max(
                        (
                            scored[(min(image_id, other), max(image_id, other))]
                            for other in members
                            if (min(image_id, other), max(image_id, other)) in scored
                        ),
                        default=0.0,
                    ),
                )
            files.append(
                {
                    "id": image_id,
                    "path": item["path"],
                    "name": item["name"],
                    "width": item["width"],
                    "height": item["height"],
                    "size": item["size"],
                    "similarity": round(similarity, 4),
                    "keep": image_id == keeper,
                }
            )

        files.sort(key=lambda f: (not f["keep"], -f["similarity"], f["name"]))
        out.append(
            {
                "keeper_id": keeper,
                "files": files,
                "reclaimable": sum(f["size"] for f in files if not f["keep"]),
            }
        )

    out.sort(key=lambda g: (-g["reclaimable"], -len(g["files"])))
    return out


def pick_keeper(items: Sequence[dict]) -> int:
    """Index of the photo to keep: most pixels, then largest file, then oldest.

    The original is assumed to be the biggest and the earliest; WhatsApp
    forwards are shrunken copies made later.
    """
    return min(
        range(len(items)),
        key=lambda i: (
            -(items[i]["width"] * items[i]["height"]),
            -items[i]["size"],
            items[i]["mtime"],
        ),
    )
