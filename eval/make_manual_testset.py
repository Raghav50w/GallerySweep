"""Turn a hand-sorted folder of real photos into a scoreable test set.

    python -m eval.make_manual_testset --source "D:\\myset"

The benchmark in make_dataset.py is generated: 800 DIV2K photos put through
programmatic transforms. Its negatives are unrealistically easy, because no two
DIV2K photos look alike. This takes real photos sorted by hand instead, so the
negatives are a real gallery and the duplicates carry real transformation
chains -- a WhatsApp round trip, a phone-editor crop -- rather than simulated
ones.

Expected layout. The folders are the only labelling; nothing records *how* a
copy differs, and nothing needs to:

    myset/
      dupes/
        001/     a photo and all its copies
        002/     another photo and its copies
      singles/   every photo with no duplicate, flat

Writes `ground_truth.csv` in the format score_testset.py already reads, so:

    python -m eval.score_testset --testset data/manual_testset
    python -m eval.score_testset --testset data/manual_testset --smart

Originals are copied, never moved or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python eval/make_manual_testset.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, SUPPORTED_EXTENSIONS

DEFAULT_OUT = PROJECT_ROOT / "data" / "manual_testset"


def images_in(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build(source: Path, out: Path) -> None:
    dupes_dir = source / "dupes"
    singles_dir = source / "singles"
    for required in (dupes_dir, singles_dir):
        if not required.is_dir():
            raise SystemExit(f"expected a '{required.name}' folder under {source}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows: list[dict] = []
    group = 0
    demoted: list[str] = []
    # Keyed on content, not filename: IMG_0001.jpg is the most common name on
    # earth, and matching on it would cry wolf on every real gallery.
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []

    def copy_in(source_file: Path, target: Path, where: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        if digest in seen and seen[digest] != where:
            collisions.append((source_file.name, seen[digest], where))
        seen.setdefault(digest, where)

    # Each hand-sorted folder is one duplicate group.
    for folder in sorted(p for p in dupes_dir.iterdir() if p.is_dir()):
        members = images_in(folder)
        if len(members) < 2:
            demoted.append(folder.name)
            for source_file in members:  # a lone file is a single, not an error
                copy_in(source_file, out / "singles" / source_file.name, "singles")
                rows.append({"path": f"singles/{source_file.name}", "group": group,
                             "role": "original", "transform": "-"})
                group += 1
            continue

        for position, source_file in enumerate(members):
            target = out / "dupes" / folder.name / source_file.name
            copy_in(source_file, target, folder.name)
            rows.append({
                "path": str(target.relative_to(out)).replace("\\", "/"),
                "group": group,
                # The first file is taken as the original. On hand-sorted real
                # duplicates that is a guess, so read the keeper metric loosely.
                "role": "original" if position == 0 else "copy",
                "transform": folder.name,
            })
        group += 1

    # Every single is its own group, contributing only negatives.
    singles = images_in(singles_dir, recursive=True)
    for source_file in singles:
        target = out / "singles" / source_file.relative_to(singles_dir)
        copy_in(source_file, target, "singles")
        rows.append({
            "path": str(target.relative_to(out)).replace("\\", "/"),
            "group": group, "role": "original", "transform": "-",
        })
        group += 1

    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "group", "role", "transform"])
        writer.writeheader()
        writer.writerows(rows)

    copies = sum(1 for r in rows if r["role"] == "copy")
    sizes = Counter(r["group"] for r in rows)
    duplicate_groups = sum(1 for size in sizes.values() if size > 1)
    true_pairs = sum(size * (size - 1) // 2 for size in sizes.values())
    total_pairs = len(rows) * (len(rows) - 1) // 2

    print(f"{out}")
    print(f"  {len(rows)} images: {duplicate_groups} duplicate sets "
          f"({copies} copies) + {len(rows) - copies - duplicate_groups} singles")
    print(f"  {true_pairs} true duplicate pairs, "
          f"{total_pairs - true_pairs} non-duplicate pairs")
    if demoted:
        print(f"  {len(demoted)} folder(s) had one file and were treated as singles: "
              f"{', '.join(demoted[:6])}")
    if collisions:
        print(f"\n  {len(collisions)} identical file(s) in two places -- the labels"
              " contradict themselves there:")
        for name, first, second in collisions[:8]:
            print(f"    {name}: '{first}' and '{second}'")

    print("\nnext:")
    print(f"  python -m eval.score_testset --testset {out}")
    print(f"  python -m eval.score_testset --testset {out} --smart")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="folder containing 'dupes' and 'singles'")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"not a folder: {source}")
    build(source, args.out)


if __name__ == "__main__":
    main()
