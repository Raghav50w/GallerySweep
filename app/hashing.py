"""Perceptual hashing of the 4 normalized rotations."""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

from app.config import HASH_SIZE
from app.normalize import normalize, rotations


def hash_variants(img: Image.Image) -> dict[int, str]:
    """pHash each rotation, as 16-char hex.

    Hex, not int: SQLite's INTEGER is signed 64-bit, so roughly half of all
    pHashes (the ones with the high bit set) either overflow or make sqlite3
    raise "Python int too large to convert to SQLite INTEGER".

    Note that imagehash resizes to a square, ignoring aspect ratio, and that is
    exactly why this works: resizing to a square and rotating by a multiple of
    90 degrees commute, so the 32x32 grid of a rotated image is the rotated
    32x32 grid of the original. Preserving aspect ratio would break it.
    """
    return {
        degrees: str(imagehash.phash(variant, hash_size=HASH_SIZE))
        for degrees, variant in rotations(img).items()
    }


def hash_file(path: str | Path) -> tuple[dict[int, str], int, int]:
    """Normalize a file and hash it. Returns (hashes, orig_width, orig_height)."""
    img, width, height = normalize(path)
    return hash_variants(img), width, height
