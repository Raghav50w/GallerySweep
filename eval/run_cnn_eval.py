"""Measure what the CNN buys: hash vs CNN vs cascade.

    python -m eval.run_cnn_eval

`run_eval.py` measures the hash engine alone. This one adds the two arms that
Smart mode introduces, on the same 16,000-image benchmark and the same
definition of a duplicate (two images sharing a `base_id`):

  hash     -- Hamming distance over the 4 normalized rotations, as shipped.
  CNN      -- ResNet-50 cosine over the same 4 rotations, every pair scored.
              Not a shippable engine at this size; it is the ceiling the
              cascade is being compared against.
  cascade  -- what Smart mode actually does: the hash nominates everything
              within CANDIDATE_HAMMING_GATE, the CNN re-ranks only those, and
              recall is still measured against *all* true pairs, so the gate's
              losses are charged to the cascade rather than hidden.

The three-way comparison is the point. Reporting the CNN arm alone would flatter
the method by ignoring that nobody can afford 128 million forward-pass
comparisons on a phone backup.

Embeddings are cached to `data/embeddings.npy` (~2 GB, gitignored), so
re-running the metrics does not re-embed. Budget ~40 minutes for the first run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python eval/run_cnn_eval.py`, not just -m
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    CANDIDATE_HAMMING_GATE,
    EMBED_DIM,
    FAST_HAMMING_THRESHOLD,
    HASH_BITS,
    PROJECT_ROOT,
    ROTATIONS,
)
from app.embeddings import embed_images
from app.matching import _popcount
from eval.run_eval import (
    MANIFEST,
    TARGET_PRECISION,
    average_precision,
    hash_library,
    sweep_table,
    to_uint64,
)

DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
EMBED_CACHE = DATA / "embeddings.npy"

# Cosine thresholds are swept on this grid. It reaches well below anything the
# UI exposes (the slider starts at 0.85) because average precision is the area
# under the whole curve -- truncating the grid at the shipping range would clip
# the low-precision tail and quietly inflate the number.
COSINE_GRID = np.round(np.arange(0.50, 1.0001, 0.0025), 4)

# The band the Phase 3 briefing identified as the CNN's real territory: far
# enough out that the hash is genuinely uncertain, close enough in to be inside
# the candidate gate.
BAND = (12, CANDIDATE_HAMMING_GATE)


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

def embed_library(paths: list[str], force: bool) -> np.ndarray:
    """(n, 4, EMBED_DIM) unit vectors: every image at every rotation.

    All four rotations, because the hash arm gets all four. Handing the CNN a
    single orientation would make the comparison a test of rotation handling
    rather than of the two representations.
    """
    n = len(paths)
    if EMBED_CACHE.exists() and not force:
        cached = np.load(EMBED_CACHE, mmap_mode="r")
        if cached.shape == (n, len(ROTATIONS), EMBED_DIM):
            print(f"using cached embeddings ({n} images x {len(ROTATIONS)} rotations)")
            return np.asarray(cached)
        print("cached embeddings do not match the manifest; re-embedding")

    out = np.zeros((n, len(ROTATIONS), EMBED_DIM), dtype=np.float32)
    total = n * len(ROTATIONS)
    started = time.perf_counter()
    done = 0
    missing = 0

    # Chunked so the intermediate dict never holds the whole 2 GB at once.
    for start in range(0, n, 200):
        stop = min(start + 200, n)
        work = [
            (index, paths[index], rotation)
            for index in range(start, stop)
            for rotation in ROTATIONS
        ]
        vectors = embed_images(work)
        for (index, rotation), vector in vectors.items():
            out[index, ROTATIONS.index(rotation)] = vector
        missing += len(work) - len(vectors)
        done += len(work)
        rate = done / (time.perf_counter() - started)
        remaining = (total - done) / rate if rate else 0
        print(
            f"\r  {done} / {total}  ({rate:.0f} img/s, ~{remaining / 60:.0f} min left)",
            end="",
            flush=True,
        )

    print(f"\n  embedded in {(time.perf_counter() - started) / 60:.1f} min")
    if missing:
        print(f"  {missing} images could not be embedded and are zero vectors")
    np.save(EMBED_CACHE, out)
    return out


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------

def cosine_bucket(similarity: np.ndarray) -> np.ndarray:
    """Index into COSINE_GRID: the loosest threshold that still flags a pair."""
    return np.clip(
        np.searchsorted(COSINE_GRID, similarity, side="right") - 1,
        0,
        len(COSINE_GRID) - 1,
    )


def sweep_all_pairs(
    hashes: np.ndarray, vectors: np.ndarray, base_ids: np.ndarray, chunk: int
) -> dict[str, np.ndarray]:
    """One pass over all ~128M pairs, histogramming all three engines at once.

    Histograms rather than per-pair scores, for the reason `run_eval` gives: no
    metric function accepts 128 million samples, and a bucketed curve is all a
    PR plot needs. Doing the three engines in one pass means the hash distance
    and the cosine for a given pair are computed together, which is what the
    cascade arm needs -- it scores the rotation the *hash* chose, exactly as
    `app/scanner.py` does.
    """
    n = hashes.shape[0]
    rotations = len(ROTATIONS)
    # A reshape view, not a copy -- BLAS takes the transpose as a flag, and a
    # materialized (2048, 64000) copy would be another half a gigabyte.
    flat = vectors.reshape(n * rotations, EMBED_DIM)

    counts = {
        "hash_pos": np.zeros(HASH_BITS + 1, dtype=np.int64),
        "hash_neg": np.zeros(HASH_BITS + 1, dtype=np.int64),
        "cnn_pos": np.zeros(len(COSINE_GRID), dtype=np.int64),
        "cnn_neg": np.zeros(len(COSINE_GRID), dtype=np.int64),
        "cascade_pos": np.zeros(len(COSINE_GRID), dtype=np.int64),
        "cascade_neg": np.zeros(len(COSINE_GRID), dtype=np.int64),
        "band_pos": np.zeros(len(COSINE_GRID), dtype=np.int64),
        "band_neg": np.zeros(len(COSINE_GRID), dtype=np.int64),
    }
    totals = {"positives": 0, "negatives": 0, "candidates": 0, "candidate_pos": 0}
    endpoints = np.zeros(n, dtype=bool)

    columns = np.arange(n)[np.newaxis, :]
    started = time.perf_counter()

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = np.arange(start, stop)

        distances = _popcount(hashes[start:stop, 0][:, None, None] ^ hashes[None, :, :])
        best = distances.min(axis=2)
        best_rotation = distances.argmin(axis=2)

        similar = (vectors[start:stop, 0] @ flat.T).reshape(stop - start, n, rotations)
        cnn_best = similar.max(axis=2)
        # The cascade does not get to pick its own orientation; it inherits the
        # one the hash matched on.
        cascade = np.take_along_axis(similar, best_rotation[:, :, None], axis=2)[..., 0]

        upper = columns > rows[:, None]
        same = base_ids[None, :] == base_ids[rows][:, None]
        positive = upper & same
        negative = upper & ~same

        counts["hash_pos"] += np.bincount(best[positive], minlength=HASH_BITS + 1)
        counts["hash_neg"] += np.bincount(best[negative], minlength=HASH_BITS + 1)

        cnn_bucket = cosine_bucket(cnn_best)
        counts["cnn_pos"] += np.bincount(
            cnn_bucket[positive], minlength=len(COSINE_GRID)
        )
        counts["cnn_neg"] += np.bincount(
            cnn_bucket[negative], minlength=len(COSINE_GRID)
        )

        gated = best <= CANDIDATE_HAMMING_GATE
        cascade_bucket = cosine_bucket(cascade)
        counts["cascade_pos"] += np.bincount(
            cascade_bucket[positive & gated], minlength=len(COSINE_GRID)
        )
        counts["cascade_neg"] += np.bincount(
            cascade_bucket[negative & gated], minlength=len(COSINE_GRID)
        )

        in_band = (best >= BAND[0]) & (best <= BAND[1])
        counts["band_pos"] += np.bincount(
            cascade_bucket[positive & in_band], minlength=len(COSINE_GRID)
        )
        counts["band_neg"] += np.bincount(
            cascade_bucket[negative & in_band], minlength=len(COSINE_GRID)
        )

        totals["positives"] += int(positive.sum())
        totals["negatives"] += int(negative.sum())
        totals["candidates"] += int((upper & gated).sum())
        totals["candidate_pos"] += int((positive & gated).sum())

        # Both endpoints of every candidate pair: the images Smart mode embeds.
        hit = upper & gated
        endpoints[start:stop] |= hit.any(axis=1)
        endpoints |= hit.any(axis=0)

        elapsed = time.perf_counter() - started
        print(f"\r  {stop} / {n} rows ({elapsed:.0f}s)", end="", flush=True)

    print()
    counts["totals"] = totals
    counts["embedded"] = int(endpoints.sum())
    return counts


def cosine_sweep_table(
    positives: np.ndarray, negatives: np.ndarray, all_positives: int
) -> pd.DataFrame:
    """PR at every cosine threshold, counting from the top down.

    `all_positives` is passed separately so the cascade's recall denominator
    stays every true duplicate in the benchmark, including the ones the gate
    threw away before the CNN ever saw them.
    """
    true_positive = np.cumsum(positives[::-1])[::-1]
    false_positive = np.cumsum(negatives[::-1])[::-1]
    flagged = true_positive + false_positive

    precision = np.divide(
        true_positive, flagged, out=np.ones(len(flagged), dtype=float), where=flagged > 0
    )
    recall = true_positive / all_positives
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(precision), where=denominator > 0,
    )

    return pd.DataFrame(
        {
            "threshold": COSINE_GRID,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": all_positives - true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def operating_point(table: pd.DataFrame, floor: float) -> pd.Series:
    """Highest-recall row that still clears the precision floor."""
    eligible = table[table["precision"] >= floor]
    if not len(eligible):
        return table.iloc[table["f1"].idxmax()]
    return eligible.loc[eligible["recall"].idxmax()]


# --------------------------------------------------------------------------
# per-transform
# --------------------------------------------------------------------------

def per_transform_table(
    manifest: pd.DataFrame,
    hashes: np.ndarray,
    vectors: np.ndarray,
    cosine_threshold: float,
) -> pd.DataFrame:
    """Per transform: hash recall, candidate recall, cascade recall.

    Candidate recall is the ceiling -- a pair the gate never nominates cannot be
    recovered by any re-ranker, however good.
    """
    positions = {
        (row.base_id, row.transform): index
        for index, row in enumerate(manifest.itertuples())
    }
    bases = manifest[manifest["transform"] == "base"]["base_id"].tolist()
    transforms = [t for t in manifest["transform"].unique() if t != "base"]

    rows = []
    for transform in transforms:
        pairs = [
            (positions[(base, "base")], positions[(base, transform)])
            for base in bases
            if (base, transform) in positions
        ]
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])

        distances = _popcount(hashes[a, 0][:, None] ^ hashes[b])
        best = distances.min(axis=1)
        best_rotation = distances.argmin(axis=1)

        similar = np.einsum("ik,ijk->ij", vectors[a, 0], vectors[b])
        cascade = np.take_along_axis(similar, best_rotation[:, None], axis=1)[:, 0]

        nominated = best <= CANDIDATE_HAMMING_GATE
        rows.append(
            {
                "transform": transform,
                "pairs": len(pairs),
                "recall_hash": float((best <= FAST_HAMMING_THRESHOLD).mean()),
                "recall_candidate": float(nominated.mean()),
                "recall_cascade": float((nominated & (cascade >= cosine_threshold)).mean()),
                "median_cosine": float(np.median(similar.max(axis=1))),
            }
        )

    return pd.DataFrame(rows).sort_values("recall_cascade", ascending=False)


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------

def write_plot(hash_table, cnn_table, cascade_table) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 5))
    axes.plot(hash_table["recall"], hash_table["precision"], label="hash only")
    axes.plot(cnn_table["recall"], cnn_table["precision"], label="CNN, every pair")
    axes.plot(cascade_table["recall"], cascade_table["precision"],
              label="cascade (shipped)", linestyle="--")
    axes.set_xlabel("recall")
    axes.set_ylabel("precision")
    axes.set_title("Hash vs CNN vs cascade")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(RESULTS / "pr_curve_engines.png", dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument("--reembed", action="store_true")
    parser.add_argument("--precision-floor", type=float, default=TARGET_PRECISION)
    parser.add_argument("--chunk", type=int, default=64,
                        help="query rows per pass; lower it if memory is tight")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N base photos only, for a quick smoke run")
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found -- run `python -m eval.make_dataset` first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype=str)
    if args.limit:
        keep = manifest["base_id"].drop_duplicates().head(args.limit)
        manifest = manifest[manifest["base_id"].isin(set(keep))].reset_index(drop=True)
        # A smoke run must never overwrite committed numbers with a subset.
        global EMBED_CACHE
        EMBED_CACHE = DATA / f"embeddings_limit{args.limit}.npy"
    paths = manifest["path"].tolist()
    print(f"{len(manifest)} images, {manifest['base_id'].nunique()} base photos")

    import os

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    hashes = to_uint64(
        hash_library(paths, workers, args.rehash), ["h0", "h90", "h180", "h270"]
    )
    vectors = embed_library(paths, args.reembed)
    base_ids = pd.factorize(manifest["base_id"])[0]

    print("sweeping all pairs (hash, CNN and cascade in one pass)")
    counts = sweep_all_pairs(hashes, vectors, base_ids, args.chunk)
    totals = counts["totals"]
    all_positives = totals["positives"]

    hash_table = sweep_table(counts["hash_pos"], counts["hash_neg"])
    cnn_table = cosine_sweep_table(counts["cnn_pos"], counts["cnn_neg"], all_positives)
    cascade_table = cosine_sweep_table(
        counts["cascade_pos"], counts["cascade_neg"], all_positives
    )
    band_table = cosine_sweep_table(
        counts["band_pos"], counts["band_neg"], int(counts["band_pos"].sum())
    )

    hash_point = hash_table.loc[FAST_HAMMING_THRESHOLD]
    cnn_point = operating_point(cnn_table, args.precision_floor)
    cascade_point = operating_point(cascade_table, args.precision_floor)

    comparison = pd.DataFrame(
        [
            {
                "engine": "hash (shipped, distance <= 10)",
                "threshold": FAST_HAMMING_THRESHOLD,
                "precision": hash_point["precision"],
                "recall": hash_point["recall"],
                "f1": hash_point["f1"],
                "average_precision": average_precision(hash_table),
                "images_embedded": 0,
            },
            {
                "engine": "CNN, every pair scored",
                "threshold": cnn_point["threshold"],
                "precision": cnn_point["precision"],
                "recall": cnn_point["recall"],
                "f1": cnn_point["f1"],
                "average_precision": average_precision(
                    cnn_table.sort_values("recall")
                ),
                "images_embedded": len(paths),
            },
            {
                "engine": f"cascade (gate <= {CANDIDATE_HAMMING_GATE}, then CNN)",
                "threshold": cascade_point["threshold"],
                "precision": cascade_point["precision"],
                "recall": cascade_point["recall"],
                "f1": cascade_point["f1"],
                "average_precision": average_precision(
                    cascade_table.sort_values("recall")
                ),
                "images_embedded": counts["embedded"],
            },
        ]
    )

    breakdown = per_transform_table(
        manifest, hashes, vectors, float(cascade_point["threshold"])
    )

    # The headline from the Phase 3 briefing: what the hash cannot sort.
    band_hash_flagged = (
        counts["band_pos"].sum() + counts["band_neg"].sum()
    )
    band_point = operating_point(band_table, 0.90)
    band = pd.DataFrame(
        [
            {"metric": f"pairs in distance band {BAND[0]}-{BAND[1]}",
             "value": f"{int(band_hash_flagged)}"},
            {"metric": "true duplicates in the band",
             "value": f"{int(counts['band_pos'].sum())}"},
            {"metric": "hash precision if the band is accepted",
             "value": f"{counts['band_pos'].sum() / max(band_hash_flagged, 1):.4f}"},
            {"metric": "CNN cosine threshold on the band",
             "value": f"{band_point['threshold']:.4f}"},
            {"metric": "CNN precision on the band", "value": f"{band_point['precision']:.4f}"},
            {"metric": "CNN recall within the band", "value": f"{band_point['recall']:.4f}"},
        ]
    )

    cascade_ceiling = totals["candidate_pos"] / all_positives
    summary = pd.DataFrame(
        [
            {"metric": "images", "value": f"{len(paths)}"},
            {"metric": "duplicate pairs", "value": f"{all_positives}"},
            {"metric": "non-duplicate pairs", "value": f"{totals['negatives']}"},
            {"metric": "candidate pairs at the gate", "value": f"{totals['candidates']}"},
            {"metric": "candidate recall (cascade ceiling)", "value": f"{cascade_ceiling:.4f}"},
            {"metric": "images embedded by the cascade", "value": f"{counts['embedded']}"},
            {"metric": "images embedded scoring every pair", "value": f"{len(paths)}"},
            {"metric": "CNN work saved",
             "value": f"{len(paths) / max(counts['embedded'], 1):.2f}x"},
        ]
    )

    if args.limit:
        print("\n--limit is a smoke run; results/ left untouched")
    else:
        cnn_table.to_csv(RESULTS / "cnn_sweep.csv", index=False)
        cascade_table.to_csv(RESULTS / "cascade_sweep.csv", index=False)
        comparison.to_csv(RESULTS / "engine_comparison.csv", index=False)
        breakdown.to_csv(RESULTS / "candidate_recall.csv", index=False)
        band.to_csv(RESULTS / "distance_band.csv", index=False)
        summary.to_csv(RESULTS / "cascade_summary.csv", index=False)
        write_plot(hash_table, cnn_table, cascade_table)

    print()
    print(comparison.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(band.to_string(index=False))
    print()
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
