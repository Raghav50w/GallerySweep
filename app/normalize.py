"""Shared normalization: EXIF orientation, uniform-border crop, 4 rotations.

Both detection engines run on this, because neither is rotation-invariant.
pHash reads low-frequency DCT coefficients, and a 90-degree rotation permutes
the DCT basis, so an image and its own rotation land ~32/64 bits apart --
statistically indistinguishable from two unrelated photos.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

from PIL import Image, ImageChops, ImageOps

from app.config import BORDER_MIN_AREA_REDUCTION, BORDER_TOLERANCE, ROTATIONS

# Multi-frame files (animated gif/webp, multipage tiff) decode frame 0.
Image.MAX_IMAGE_PIXELS = None

_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def load_image(path: str | Path | IO[bytes]) -> Image.Image:
    """Open a file (or an open byte stream) and apply its EXIF orientation tag."""
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im) or im


def _uniform_corner_colour(rgb: Image.Image, tolerance: int) -> tuple[int, int, int] | None:
    """Return the border colour if all four corners agree, else None."""
    w, h = rgb.size
    if w < 2 or h < 2:
        return None
    corners = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((w - 1, 0)),
        rgb.getpixel((0, h - 1)),
        rgb.getpixel((w - 1, h - 1)),
    ]
    for channel in zip(*corners):
        if max(channel) - min(channel) > tolerance:
            return None
    return tuple(sum(channel) // len(channel) for channel in zip(*corners))  # type: ignore[return-value]


def border_crop(
    img: Image.Image,
    tolerance: int = BORDER_TOLERANCE,
    min_area_reduction: float = BORDER_MIN_AREA_REDUCTION,
) -> Image.Image:
    """Crop away a uniform border of any colour (black, white, or coloured bars).

    `getbbox()` on its own is an exact-match test and finds nothing on real
    JPEGs, where a "black" bar is full of 0/2/3 values. The tolerance threshold
    below is what makes this work -- without it the padded-screenshot feature is
    silently dead while appearing to run.
    """
    rgb = img.convert("RGB")
    colour = _uniform_corner_colour(rgb, tolerance)
    if colour is None:
        return img

    bg = Image.new("RGB", rgb.size, colour)
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda p: 255 if p > tolerance else 0)
    bbox = mask.getbbox()
    if bbox is None:
        # The whole image is one flat colour; there is no content to keep.
        return img

    width, height = rgb.size
    kept = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    full = width * height
    if kept == 0 or (full - kept) / full <= min_area_reduction:
        return img
    return img.crop(bbox)


def rotations(img: Image.Image) -> dict[int, Image.Image]:
    """The 4 axis-aligned rotations. No mirroring -- mirrored photos are rare."""
    out = {0: img}
    for degrees in ROTATIONS[1:]:
        out[degrees] = img.transpose(_ROTATE_OPS[degrees])
    return out


def normalize(path: str | Path) -> tuple[Image.Image, int, int]:
    """Load, orient and de-pad an image.

    Returns the normalized image plus the *original* (oriented, uncropped) width
    and height -- that is what the UI shows and what keeper selection ranks on.
    """
    img = load_image(path)
    width, height = img.size
    return border_crop(img), width, height
