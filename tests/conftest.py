"""Synthetic photos, so the suite needs no committed image files."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter


def make_photo(seed: int, size: tuple[int, int] = (800, 600)) -> Image.Image:
    """A deterministic photo-like image: low-frequency colour field plus shapes.

    pHash reads low-frequency DCT coefficients, so pure noise would produce
    unstable hashes and prove nothing. Big smooth regions are what real photos
    look like to the hash.
    """
    rng = np.random.default_rng(seed)
    width, height = size

    field = rng.integers(30, 226, (6, 8, 3), dtype=np.uint8)
    photo = Image.fromarray(field).resize((width, height), Image.Resampling.BICUBIC)

    draw = ImageDraw.Draw(photo)
    for _ in range(9):
        x0 = int(rng.integers(0, width * 3 // 4))
        y0 = int(rng.integers(0, height * 3 // 4))
        x1 = x0 + int(rng.integers(width // 8, width // 3))
        y1 = y0 + int(rng.integers(height // 8, height // 3))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            draw.ellipse([x0, y0, x1, y1], fill=colour)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=colour)

    return photo.filter(ImageFilter.GaussianBlur(1.2))


def add_dark_texture_border(photo: Image.Image, thickness: int = 40) -> Image.Image:
    """Dark but *non-uniform* edges -- a real photo shot into shadow.

    border_crop must leave this alone; over-cropping real content is the
    failure mode of that function.
    """
    rng = np.random.default_rng(99)
    array = np.array(photo.convert("RGB"), dtype=np.int16)
    mask = np.ones(array.shape[:2], dtype=bool)
    mask[thickness:-thickness, thickness:-thickness] = False
    noise = rng.integers(0, 60, array.shape, dtype=np.int16)
    array[mask] = noise[mask]
    return Image.fromarray(array.astype(np.uint8))


@pytest.fixture
def photo() -> Image.Image:
    return make_photo(1)


@pytest.fixture
def other_photo() -> Image.Image:
    """A genuinely unrelated image; it must never join a duplicate group."""
    return make_photo(77)
