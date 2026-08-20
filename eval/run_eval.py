"""Measure the hash engine: threshold sweep, PR curves, per-transform recall.

    python -m eval.run_eval

Reads `data/manifest.csv` (written by make_dataset.py), hashes every image
twice -- once with the normalization stage and once without -- and writes CSVs
and plots to `results/`.

The ablation is the point: "with normalization" applies the uniform-border crop
and indexes all four rotations; "without" hashes the file as-is at a single
orientation. The gap between the two curves is what the normalization stage
buys.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python eval/run_eval.py`, not just `-m`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DISTANCE_CHUNK_BYTES, HASH_BITS, HASH_SIZE, PROJECT_ROOT
from app.hashing import hash_variants
from app.matching import _popcount
from app.normalize import border_crop, load_image

DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
MANIFEST = DATA / "manifest.csv"
HASH_CACHE = DATA / "hashes.csv"

# Precision floor used to pick the shipped threshold. A false positive means
# deleting a photo someone wanted, so recall is traded away for it.
TARGET_PRECISION = 0.999

HASH_FIELDS = ["path", "raw", "h0", "h90", "h180", "h270", "sha256"]


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

def hash_one(path: str) -> tuple[str, str, str, str, str, str, str]:
    """Hash one file both ways, plus its checksum, from a single read."""
    blob = Path(path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()

    img = load_image(io.BytesIO(blob))
    raw = str(imagehash.phash(img, hash_size=HASH_SIZE))  # no crop, one orientation
    normalized = hash_variants(border_crop(img))

    return (
        path,
        raw,
        normalized[0],
        normalized[90],
        normalized[180],
        normalized[270],
        digest,
    )


def hash_library(paths: list[str], workers: int, force: bool) -> pd.DataFrame:
    """Hash every image, caching to disk so threshold tweaks do not re-hash."""
    if HASH_CACHE.exists() and not force:
        cached = pd.read_csv(HASH_CACHE, dtype=str)
        if set(cached["path"]) >= set(paths):
            print(f"using cached hashes ({len(cached)} images)")
            return cached.set_index("path").loc[paths].reset_index()

    print(f"hashing {len(paths)} images on {workers} workers")
    started = time.perf_counter()
    rows: list[tuple] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(hash_one, paths, chunksize=32), 1):
            rows.append(row)
            if done % 250 == 0 or done == len(paths):
                rate = done / (time.perf_counter() - started)
                print(f"\r  {done} / {len(paths)}  ({rate:.0f} img/s)", end="", flush=True)
    elapsed = time.perf_counter() - started
    print(f"\n  done in {elapsed:.0f}s ({len(paths) / elapsed:.0f} img/s)")

    with open(HASH_CACHE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HASH_FIELDS)
        writer.writerows(rows)

    return pd.DataFrame(rows, columns=HASH_FIELDS)


def to_uint64(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.array(
        [[int(value, 16) for value in row] for row in frame[columns].to_numpy()],
        dtype=np.uint64,
    )


# --------------------------------------------------------------------------
# distance sweep
# --------------------------------------------------------------------------

def distance_histograms(
    hashes: np.ndarray, base_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Count every pair by distance, split into same-photo and different-photo.

    Histograms rather than a score vector: at full size this is ~128 million
    pairs, which no per-sample metric function will accept, but 65 buckets is
    all a PR curve needs.
    """
    n = hashes.shape[0]
    variants = hashes.shape[1]
    positives = np.zeros(HASH_BITS + 1, dtype=np.int64)
    negatives = np.zeros(HASH_BITS + 1, dtype=np.int64)

    chunk = max(1, int(DISTANCE_CHUNK_BYTES // (n * variants * 8)))
    index = hashes[np.newaxis, :, :]
    columns = np.arange(n)[np.newaxis, :]

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = np.arange(start, stop)
        queries = hashes[start:stop, 0][:, np.newaxis, np.newaxis]
        distances = _popcount(queries ^ index).min(axis=2)

        # Upper triangle only: each unordered pair counted once, no self-pairs.
        upper = columns > rows[:, np.newaxis]
        same = base_ids[np.newaxis, :] == base_ids[rows][:, np.newaxis]

        positives += np.bincount(
            distances[upper & same].astype(np.int64), minlength=HASH_BITS + 1
        )
        negatives += np.bincount(
            distances[upper & ~same].astype(np.int64), minlength=HASH_BITS + 1
        )
        print(f"\r  {min(stop, n)} / {n} rows", end="", flush=True)
    print()
    return positives, negatives


def sweep_table(positives: np.ndarray, negatives: np.ndarray) -> pd.DataFrame:
    """Precision/recall/F1 at every Hamming threshold from 0 to 64."""
    total_positive = positives.sum()
    true_positive = np.cumsum(positives)
    false_positive = np.cumsum(negatives)
    false_negative = total_positive - true_positive

    flagged = true_positive + false_positive
    precision = np.divide(
        true_positive, flagged, out=np.ones_like(true_positive, dtype=float),
        where=flagged > 0,
    )
    recall = true_positive / total_positive
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(precision), where=denominator > 0,
    )

    return pd.DataFrame(
        {
            "threshold": np.arange(HASH_BITS + 1),
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def average_precision(table: pd.DataFrame) -> float:
    """Area under the PR curve, summed the standard way."""
    recall = table["recall"].to_numpy()
    precision = table["precision"].to_numpy()
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def recommend_threshold(table: pd.DataFrame, floor: float) -> int:
    """Loosest threshold that still holds precision at or above the floor.

    Rounded down to an even number, because odd thresholds cannot do anything:
    `imagehash.phash` compares each of the 64 low-frequency DCT coefficients
    against the median of all 64, so every hash has exactly 32 bits set. For two
    such hashes |A xor B| = 64 - 2|A and B|, which is always even. A threshold of
    11 therefore behaves identically to 10, and reporting 11 would imply a
    precision it does not actually buy.
    """
    eligible = table[table["precision"] >= floor]
    if not len(eligible):
        return 0
    return int(eligible["threshold"].max()) // 2 * 2


# --------------------------------------------------------------------------
# per-transform breakdown
# --------------------------------------------------------------------------

def per_transform_table(
    manifest: pd.DataFrame,
    normalized: np.ndarray,
    raw: np.ndarray,
    checksums: np.ndarray,
    threshold: int,
    raw_threshold: int,
) -> pd.DataFrame:
    """Recall for each transform, measured against its own base image."""
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

        norm_distance = _popcount(
            normalized[a, 0][:, np.newaxis] ^ normalized[b]
        ).min(axis=1)
        raw_distance = _popcount(raw[a, 0] ^ raw[b, 0])
        identical = checksums[a] == checksums[b]

        rows.append(
            {
                "transform": transform,
                "pairs": len(pairs),
                "recall_normalized": float((norm_distance <= threshold).mean()),
                "recall_no_normalization": float((raw_distance <= raw_threshold).mean()),
                "recall_checksum": float(identical.mean()),
                "median_distance": float(np.median(norm_distance)),
                "p95_distance": float(np.percentile(norm_distance, 95)),
            }
        )

    return pd.DataFrame(rows).sort_values("recall_normalized", ascending=False)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def write_plots(normalized: pd.DataFrame, raw: pd.DataFrame, histograms: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 5))
    axes.plot(normalized["recall"], normalized["precision"], label="with normalization")
    axes.plot(raw["recall"], raw["precision"], label="without normalization",
              linestyle="--")
    axes.set_xlabel("recall")
    axes.set_ylabel("precision")
    axes.set_title("Perceptual hash, precision vs recall")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(RESULTS / "pr_curve.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7, 4))
    bins = np.arange(HASH_BITS + 1)
    axes.bar(bins, histograms["positives"], width=1.0, alpha=0.75, label="same photo")
    axes.bar(bins, histograms["negatives"], width=1.0, alpha=0.55,
             label="different photos")
    axes.set_yscale("log")
    axes.set_xlabel("Hamming distance (of 64 bits)")
    axes.set_ylabel("pairs (log scale)")
    axes.set_title("Distance distribution, with normalization")
    axes.legend()
    figure.tight_layout()
    figure.savefig(RESULTS / "distance_distribution.png", dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0,
                        help="hashing processes (0 = cpu count - 1)")
    parser.add_argument("--rehash", action="store_true", help="ignore the hash cache")
    parser.add_argument("--precision-floor", type=float, default=TARGET_PRECISION)
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found -- run `python -m eval.make_dataset` first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype=str)
    print(f"{len(manifest)} images, {manifest['base_id'].nunique()} base photos, "
          f"{manifest['transform'].nunique() - 1} transforms")

    import os

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    hashes = hash_library(manifest["path"].tolist(), workers, args.rehash)

    normalized = to_uint64(hashes, ["h0", "h90", "h180", "h270"])
    raw = to_uint64(hashes, ["raw"])
    checksums = hashes["sha256"].to_numpy()
    base_ids = pd.factorize(manifest["base_id"])[0]

    print("sweeping with normalization")
    positives, negatives = distance_histograms(normalized, base_ids)
    print("sweeping without normalization")
    raw_positives, raw_negatives = distance_histograms(raw, base_ids)

    normalized_table = sweep_table(positives, negatives)
    raw_table = sweep_table(raw_positives, raw_negatives)

    normalized_ap = average_precision(normalized_table)
    raw_ap = average_precision(raw_table)
    threshold = recommend_threshold(normalized_table, args.precision_floor)
    raw_threshold = recommend_threshold(raw_table, args.precision_floor)

    breakdown = per_transform_table(
        manifest, normalized, raw, checksums, threshold, raw_threshold
    )

    normalized_table.to_csv(RESULTS / "sweep_normalized.csv", index=False)
    raw_table.to_csv(RESULTS / "sweep_no_normalization.csv", index=False)
    breakdown.to_csv(RESULTS / "per_transform_recall.csv", index=False)
    write_plots(normalized_table, raw_table, {"positives": positives,
                                              "negatives": negatives})

    at = normalized_table.loc[threshold]
    # Values are formatted as strings so counts do not get dragged into
    # scientific notation by the ratios sharing the column.
    summary = pd.DataFrame(
        [
            {"metric": "images", "value": f"{len(manifest)}"},
            {"metric": "base photos", "value": f"{manifest['base_id'].nunique()}"},
            {"metric": "duplicate pairs", "value": f"{int(positives.sum())}"},
            {"metric": "non-duplicate pairs", "value": f"{int(negatives.sum())}"},
            {"metric": "average precision (normalized)", "value": f"{normalized_ap:.4f}"},
            {"metric": "average precision (no normalization)", "value": f"{raw_ap:.4f}"},
            {"metric": "shipped threshold", "value": f"{threshold}"},
            {"metric": "precision at shipped threshold", "value": f"{at['precision']:.6f}"},
            {"metric": "recall at shipped threshold", "value": f"{at['recall']:.4f}"},
            {"metric": "f1 at shipped threshold", "value": f"{at['f1']:.4f}"},
            {"metric": "false positives at shipped threshold",
             "value": f"{int(at['false_positives'])}"},
            {"metric": "best f1", "value": f"{normalized_table['f1'].max():.4f}"},
            {"metric": "best f1 threshold",
             "value": f"{int(normalized_table['f1'].idxmax()) // 2 * 2}"},
        ]
    )
    summary.to_csv(RESULTS / "summary.csv", index=False)

    print()
    print(summary.to_string(index=False))
    print()
    print(breakdown.to_string(index=False))
    print()
    print(f"negative pairs are {negatives.sum() / (positives.sum() + negatives.sum()):.4%} "
          "of the total -- this is why accuracy is not reported")
    print(f"set FAST_HAMMING_THRESHOLD = {threshold} in app/config.py")


if __name__ == "__main__":
    main()
