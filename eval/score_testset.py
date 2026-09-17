"""Score the realistic test set against its ground truth.

Runs the same code path the app does -- normalize, hash 4 rotations, search at
the shipped threshold, group by connected components, pick a keeper -- over
`data/testset`, then compares the result to `ground_truth.csv`.

    python -m eval.score_testset
    python -m eval.score_testset --threshold 12

Where run_eval.py measures the hash function, this measures the product: whether
a person pointing the app at a messy folder gets the right groups, and whether
the photo it offers to keep is the original.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import pandas as pd

from app.config import (
    CANDIDATE_HAMMING_GATE,
    COSINE_DEFAULT,
    FAST_HAMMING_THRESHOLD,
    PROJECT_ROOT,
    SUPPORTED_EXTENSIONS,
)
from app.hashing import hash_file
from app.matching import (
    assemble_groups,
    cosine_for_pairs,
    embedding_work_set,
    find_candidate_pairs,
    hamming_to_similarity,
    hashes_to_array,
)

TESTSET = PROJECT_ROOT / "data" / "testset"
RESULTS = PROJECT_ROOT / "results"


def measure(path: str) -> tuple:
    """Hash one file exactly as scanner.py would."""
    hashes, width, height = hash_file(path)
    stat = Path(path).stat()
    return (
        path,
        hashes[0], hashes[90], hashes[180], hashes[270],
        width, height, stat.st_size, stat.st_mtime_ns,
    )


def scan(paths: list[str], workers: int) -> pd.DataFrame:
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(measure, paths, chunksize=16), 1):
            rows.append(row)
            if done % 100 == 0 or done == len(paths):
                print(f"\r  hashed {done} / {len(paths)}", end="", flush=True)
    elapsed = time.perf_counter() - started
    print(f"  in {elapsed:.1f}s")
    return pd.DataFrame(
        rows,
        columns=["path", "h0", "h90", "h180", "h270",
                 "width", "height", "size", "mtime"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=FAST_HAMMING_THRESHOLD)
    parser.add_argument("--testset", type=Path, default=TESTSET)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smart", action="store_true",
                        help="hash candidates re-ranked by the CNN, as Smart mode does")
    parser.add_argument("--cosine", type=float, default=COSINE_DEFAULT)
    args = parser.parse_args()

    truth_path = args.testset / "ground_truth.csv"
    if not truth_path.exists():
        raise SystemExit(f"{truth_path} not found -- run `python -m eval.make_testset`")

    import os

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    truth = pd.read_csv(truth_path)
    truth["abs"] = [str(args.testset / p) for p in truth["path"]]

    files = sorted(
        str(p) for p in args.testset.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"{len(files)} images under {args.testset}, "
          f"{truth['group'].nunique()} ground-truth photos")

    measured = scan(files, workers)
    order = {path: index for index, path in enumerate(measured["path"])}
    truth = truth[truth["abs"].isin(order)].copy()
    truth["index"] = [order[p] for p in truth["abs"]]

    # --- run the real matcher -------------------------------------------
    hashes = hashes_to_array(measured[["h0", "h90", "h180", "h270"]].to_numpy())

    if args.smart:
        # What Smart mode does: gate wide on the hash, then re-rank the
        # survivors on cosine at the rotation the hash matched on.
        candidates = find_candidate_pairs(hashes, CANDIDATE_HAMMING_GATE)
        pairs = [(int(a), int(b), int(ra), int(rb)) for a, b, _, ra, rb in candidates]
        print(f"{len(pairs)} candidates at Hamming <= {CANDIDATE_HAMMING_GATE}")

        from app.embeddings import embed_images

        needed = embedding_work_set(pairs)
        paths = measured["path"].tolist()
        print(f"embedding {len(needed)} images")
        vectors = embed_images([(i, paths[i], rotation) for i, rotation in needed])
        scored = [
            (a, b, float(score))
            for score, a, b in cosine_for_pairs(pairs, vectors)
            if score is not None and score >= args.cosine
        ]
        print(f"{len(scored)} pairs at cosine >= {args.cosine}")
    else:
        candidates = find_candidate_pairs(hashes, args.threshold)
        scored = [
            (int(a), int(b), hamming_to_similarity(int(d)))
            for a, b, d, _, _ in candidates
        ]
        print(f"{len(scored)} pairs at Hamming <= {args.threshold}")

    images = {
        index: {
            "path": row.path,
            "name": Path(row.path).name,
            "width": row.width,
            "height": row.height,
            "size": row.size,
            "mtime": row.mtime,
        }
        for index, row in enumerate(measured.itertuples())
    }
    groups = assemble_groups(scored, images)

    # --- pair-level precision and recall --------------------------------
    group_of = dict(zip(truth["index"], truth["group"]))
    true_pairs = set()
    for _, rows in truth.groupby("group"):
        for a, b in combinations(sorted(rows["index"]), 2):
            true_pairs.add((a, b))
    predicted = {(int(a), int(b)) for a, b, _ in scored}

    hit = predicted & true_pairs
    precision = len(hit) / len(predicted) if predicted else 1.0
    recall = len(hit) / len(true_pairs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_pairs = len(files) * (len(files) - 1) // 2

    # --- did each copy land in its original's group? --------------------
    found_group = {}
    for number, group in enumerate(groups):
        for entry in group["files"]:
            found_group[entry["id"]] = number

    original_index = {
        row.group: row.index
        for row in truth[truth["role"] == "original"].itertuples()
    }
    truth["together"] = [
        found_group.get(row.index) is not None
        and found_group.get(row.index) == found_group.get(original_index[row.group])
        for row in truth.itertuples()
    ]

    copies = truth[truth["role"] == "copy"].copy()
    copies["kind"] = copies["transform"].str.replace(r"_[-\d.]+%?$", "", regex=True)
    by_kind = (
        copies.groupby("kind")["together"]
        .agg(copies="size", found="sum")
        .assign(recall=lambda d: d["found"] / d["copies"])
        .sort_values("recall", ascending=False)
    )

    # --- group quality and keeper choice --------------------------------
    over_merged = 0
    keeper_right = 0
    keeper_total = 0
    for group in groups:
        members = [entry["id"] for entry in group["files"]]
        labels = {group_of.get(m) for m in members}
        if len(labels) > 1:
            over_merged += 1
            continue
        keeper_total += 1
        keeper = next(e["id"] for e in group["files"] if e["keep"])
        role = truth.loc[truth["index"] == keeper, "role"]
        keeper_right += int(len(role) and role.iloc[0] == "original")

    summary = pd.DataFrame(
        [
            ("images", len(files)),
            ("photos with duplicates", int((truth["role"] == "copy").groupby(truth["group"]).any().sum())),
            ("true duplicate pairs", len(true_pairs)),
            ("candidate pairs at threshold", len(predicted)),
            ("total possible pairs", total_pairs),
            ("pair precision", f"{precision:.4f}"),
            ("pair recall", f"{recall:.4f}"),
            ("pair f1", f"{f1:.4f}"),
            ("false positive pairs", len(predicted - true_pairs)),
            ("groups shown to the user", len(groups)),
            ("groups mixing unrelated photos", over_merged),
            ("keeper was the original", f"{keeper_right}/{keeper_total}"),
        ],
        columns=["metric", "value"],
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Smart results go to their own files so the two engines can be compared
    # rather than one silently overwriting the other.
    tag = "testset_smart" if args.smart else "testset"
    summary.to_csv(RESULTS / f"{tag}_summary.csv", index=False)
    by_kind.to_csv(RESULTS / f"{tag}_per_transform.csv")

    print()
    print(summary.to_string(index=False))
    print()
    print(by_kind.to_string())

    missed = copies[~copies["together"]]
    if len(missed):
        print(f"\n{len(missed)} copies not grouped with their original:")
        print(missed["transform"].value_counts().head(12).to_string())

    for a, b in sorted(predicted - true_pairs)[:10]:
        print(f"false positive: {Path(files[a]).name}  <->  {Path(files[b]).name}")


if __name__ == "__main__":
    main()
