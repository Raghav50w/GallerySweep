"""Build a realistic messy photo folder: 800 originals plus 200 stray copies.

Unlike the benchmark in make_dataset.py -- which applies all 19 transforms to
every photo, so each one is equally represented -- this set is shaped like a
real phone backup. Most photos have no duplicate at all. The ones that do have
either one, two or three extra copies, each mangled a different way and dropped
in a different subfolder, with phone-style filenames.

    python -m eval.make_testset
    python -m eval.make_testset --photos 200 --duplicates 50 --out data/small

`ground_truth.csv` records which file came from which photo, so the folder can
be scored rather than only eyeballed.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):  # allow `python eval/make_testset.py`, not just `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT
from eval.augment import brightness, crop, pad, resize, rotate

BASE_DIR = PROJECT_ROOT / "data" / "DIV2K_train_LR_bicubic" / "X4"
DEFAULT_OUT = PROJECT_ROOT / "data" / "testset"

# How many extra copies a duplicated photo gets. Most have one; a few have three.
COPY_COUNTS = [1, 2, 3]
COPY_WEIGHTS = [0.55, 0.28, 0.17]

# Where files land, and what they get named. Duplicates pick a folder
# independently of their original, so copies routinely end up somewhere else
# entirely -- which is the case the recursive scan exists for.
FOLDERS = [
    ("Camera", "IMG_{stamp}.jpg"),
    ("Camera", "DSC_{n:04d}.jpg"),
    ("WhatsApp/Media/WhatsApp Images", "IMG-{date}-WA{n:04d}.jpg"),
    ("Screenshots", "Screenshot_{date}-{n:06d}.png"),
    ("Downloads", "image_{date}_{n}.jpg"),
    ("Telegram", "photo_{date}_{n}.jpg"),
    ("Pictures/Saved", "{n}.jpg"),
]


def phone_name(rng: random.Random, taken: set[str]) -> tuple[str, str]:
    """A plausible folder and filename that has not been used yet."""
    while True:
        folder, pattern = rng.choice(FOLDERS)
        year = rng.randint(2019, 2026)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        name = pattern.format(
            stamp=f"{year}{month:02d}{day:02d}_{rng.randint(0, 235959):06d}",
            date=f"{year}{month:02d}{day:02d}",
            n=rng.randint(1, 9999),
        )
        key = f"{folder}/{name}"
        if key not in taken:
            taken.add(key)
            return folder, name


# --- the mangles a stray copy can have suffered ----------------------------

def _light_crop(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    fraction = rng.uniform(0.02, 0.08)
    return crop(img, fraction), f"crop_{fraction:.0%}"


def _heavy_crop(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    fraction = rng.uniform(0.12, 0.22)
    return crop(img, fraction), f"crop_{fraction:.0%}"


def _bars(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    kind = rng.choice(["black", "white", "colour"])
    colour = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "colour": tuple(rng.randrange(20, 220) for _ in range(3)),
    }[kind]
    fraction = rng.uniform(0.06, 0.20)
    axis = rng.choice(["vertical", "horizontal"])
    return pad(img, colour, fraction=fraction, axis=axis), f"bars_{kind}_{axis}"


def _rotated(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    degrees = rng.choice([90, 180, 270])
    return rotate(img, degrees), f"rotate_{degrees}"


def _shrunk(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    scale = rng.uniform(0.35, 0.85)
    return resize(img, scale), f"resize_{scale:.0%}"


def _brightened(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    factor = rng.choice([rng.uniform(0.75, 0.9), rng.uniform(1.1, 1.3)])
    return brightness(img, factor), f"brightness_{factor:.2f}"


def _untouched(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    return img.copy(), "recompressed"


# Weighted so the folder looks like a real backup: forwarded and re-saved copies
# dominate, light crops and bars are common, heavy crops are the rare hard case.
MANGLES = [
    (_untouched, 0.20),
    (_bars, 0.20),
    (_light_crop, 0.16),
    (_rotated, 0.14),
    (_shrunk, 0.14),
    (_brightened, 0.08),
    (_heavy_crop, 0.08),
]


def build(base_dir: Path, out: Path, photos: int, duplicates: int, seed: int) -> None:
    bases = sorted(base_dir.glob("*.png"))
    if len(bases) < photos:
        raise SystemExit(f"only {len(bases)} base photos under {base_dir}, need {photos}")

    rng = random.Random(seed)
    bases = rng.sample(bases, photos)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    taken: set[str] = set()
    rows: list[dict] = []

    # 1. Every photo goes in once, as its own original.
    originals: list[Path] = []
    for group, base in enumerate(bases):
        folder, name = phone_name(rng, taken)
        target = out / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(base) as img:
            img.load()
            if target.suffix == ".png":
                img.convert("RGB").save(target, "PNG")
            else:
                img.convert("RGB").save(target, "JPEG", quality=rng.randint(88, 95))
        originals.append(target)
        rows.append(
            {"path": str(target.relative_to(out)), "group": group,
             "role": "original", "transform": "-"}
        )

    # 2. Hand out the extra copies: some photos get one, a few get three.
    plan: list[tuple[int, int]] = []
    remaining = duplicates
    for group in rng.sample(range(photos), photos):
        if remaining <= 0:
            break
        count = min(rng.choices(COPY_COUNTS, weights=COPY_WEIGHTS)[0], remaining)
        plan.append((group, count))
        remaining -= count
    if remaining:
        raise SystemExit(f"could not place {remaining} copies -- too few photos")

    functions = [m[0] for m in MANGLES]
    weights = [m[1] for m in MANGLES]

    made = 0
    for group, count in plan:
        source = originals[group]
        for _ in range(count):
            folder, name = phone_name(rng, taken)
            target = out / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source) as img:
                img.load()
                mangle = rng.choices(functions, weights=weights)[0]
                variant, label = mangle(img, rng)

                # An exact byte copy is its own case, and the only one a
                # checksum would ever find.
                if label == "recompressed" and rng.random() < 0.35:
                    shutil.copy2(source, target.with_suffix(source.suffix))
                    target = target.with_suffix(source.suffix)
                    label = "exact_copy"
                elif target.suffix == ".png":
                    variant.convert("RGB").save(target, "PNG")
                else:
                    variant.convert("RGB").save(
                        target, "JPEG", quality=rng.randint(60, 92)
                    )

            rows.append(
                {"path": str(target.relative_to(out)), "group": group,
                 "role": "copy", "transform": label}
            )
            made += 1

    truth = out / "ground_truth.csv"
    with open(truth, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "group", "role", "transform"])
        writer.writeheader()
        writer.writerows(rows)

    sizes: dict[int, int] = {}
    for group, count in plan:
        sizes[count] = sizes.get(count, 0) + 1

    print(f"{out}")
    print(f"  {photos} originals + {made} copies = {photos + made} images")
    print(f"  {len(plan)} photos have duplicates, "
          f"{photos - len(plan)} have none")
    for count in sorted(sizes):
        print(f"    {sizes[count]} photos with {count} extra "
              f"cop{'y' if count == 1 else 'ies'}")
    print(f"  ground truth: {truth.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=int, default=800)
    parser.add_argument("--duplicates", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    build(BASE_DIR, args.out, args.photos, args.duplicates, args.seed)


if __name__ == "__main__":
    main()
