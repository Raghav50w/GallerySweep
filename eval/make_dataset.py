"""Download DIV2K and generate a labelled duplicate benchmark.

No manual labelling: every file here is produced *from* a known base image, so
the label is a by-product of generating it. Two files made from the same base
are duplicates; two files from different bases are not. Rerunning with the same
seed reproduces the benchmark exactly.

    python -m eval.make_dataset            # all 800 base images
    python -m eval.make_dataset --limit 50 # quick trial run
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):  # allow `python eval/make_dataset.py`, not just `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT
from eval.augment import TRANSFORMS, WHATSAPP_QUALITY, resize, WHATSAPP_LONG_EDGE

DATA = PROJECT_ROOT / "data"
ARCHIVE = DATA / "DIV2K_train_LR_bicubic_X4.zip"
BASE_DIR = DATA / "DIV2K_train_LR_bicubic" / "X4"
VARIANTS = DATA / "variants"
MANIFEST = DATA / "manifest.csv"
PAIRS = DATA / "pairs.csv"

# DIV2K is published for academic and research use. Nothing downloaded here is
# ever committed -- `data/` is gitignored.
URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"

# Transforms whose defining step *is* the encoding, so they are written straight
# to disk at that quality instead of being re-encoded on top of themselves.
JPEG_QUALITIES = {"jpeg_q90": 90, "jpeg_q60": 60, "jpeg_q30": 30}
DEFAULT_QUALITY = 95


def download(url: str, target: Path) -> None:
    if target.exists():
        print(f"already downloaded: {target.name} ({target.stat().st_size / 1e6:.0f} MB)")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB"
                      f"  ({done / total:.0%})", end="", flush=True)
    print()
    partial.rename(target)


def extract(archive: Path, expected: Path) -> None:
    if expected.is_dir() and any(expected.glob("*.png")):
        print(f"already extracted: {expected}")
        return
    print(f"extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA)


def variant_path(transform: str, base: Path) -> Path:
    suffix = ".png" if transform == "identity" else ".jpg"
    return VARIANTS / transform / f"{base.stem}{suffix}"


def write_variant(transform: str, base: Path, target: Path) -> None:
    """Produce one labelled copy of `base` on disk."""
    target.parent.mkdir(parents=True, exist_ok=True)

    if transform == "identity":
        # A byte-for-byte copy. This is the only pair a checksum can find, and
        # the per-transform table exists to show how little that is worth.
        shutil.copy2(base, target)
        return

    with Image.open(base) as img:
        img.load()
        if transform in JPEG_QUALITIES:
            img.convert("RGB").save(target, "JPEG", quality=JPEG_QUALITIES[transform])
            return
        if transform == "whatsapp":
            longest = max(img.size)
            shrunk = img if longest <= WHATSAPP_LONG_EDGE else resize(
                img, WHATSAPP_LONG_EDGE / longest
            )
            shrunk.convert("RGB").save(target, "JPEG", quality=WHATSAPP_QUALITY)
            return
        TRANSFORMS[transform](img).convert("RGB").save(
            target, "JPEG", quality=DEFAULT_QUALITY
        )


def build(limit: int, force: bool, seed: int) -> None:
    bases = sorted(BASE_DIR.glob("*.png"))
    if not bases:
        raise SystemExit(f"no base images under {BASE_DIR}")
    if limit:
        bases = bases[:limit]

    names = list(TRANSFORMS)
    print(f"generating {len(bases)} x {len(names)} = {len(bases) * len(names)} variants")

    rows: list[dict] = []
    for index, base in enumerate(bases, 1):
        base_id = base.stem
        rows.append({"path": str(base), "base_id": base_id, "transform": "base"})
        for transform in names:
            target = variant_path(transform, base)
            if force or not target.exists():
                write_variant(transform, base, target)
            rows.append(
                {"path": str(target), "base_id": base_id, "transform": transform}
            )
        if index % 25 == 0 or index == len(bases):
            print(f"\r  {index} / {len(bases)} base images", end="", flush=True)
    print()

    with open(MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "base_id", "transform"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {MANIFEST} ({len(rows)} images)")

    write_pairs(bases, names, seed)


def write_pairs(bases: list[Path], names: list[str], seed: int) -> None:
    """A readable sample of the labels.

    The full negative set is every pair of files from different base images --
    around 128 million rows at full size, which is why the sweep in run_eval.py
    derives negatives from `base_id` instead of reading them back from a CSV.
    Only the per-transform positives and an equal-sized negative sample are
    written here, for eyeballing.
    """
    positives = [
        (str(base), str(variant_path(transform, base)), 1, transform)
        for base in bases
        for transform in names
    ]

    rng = random.Random(seed)
    negatives = []
    while len(negatives) < len(positives) and len(bases) > 1:
        a, b = rng.sample(bases, 2)
        negatives.append(
            (str(a), str(variant_path(rng.choice(names), b)), 0, "different_photo")
        )

    with open(PAIRS, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_a", "path_b", "is_duplicate", "transform_type"])
        writer.writerows(positives + negatives)
    print(f"wrote {PAIRS} ({len(positives)} positives, {len(negatives)} sampled negatives)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N base images (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="rewrite variants that already exist")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    download(URL, ARCHIVE)
    extract(ARCHIVE, BASE_DIR)
    build(args.limit, args.force, args.seed)


if __name__ == "__main__":
    sys.exit(main())
