"""Transform suite used to manufacture labelled duplicates.

Written in Phase 1 because the unit tests need to generate rotated, padded and
re-compressed copies anyway; Phase 2's benchmark then comes almost free. Every
transformed copy is a known positive and every other pair a known negative, so
the labels cost nothing and the whole benchmark is reproducible from a seed.
"""

from __future__ import annotations

import io
from typing import Callable

from PIL import Image, ImageEnhance

# A saturated non-grey colour, so "coloured bars" exercises a different code
# path in border_crop than plain black or white.
BAR_COLOUR = (36, 92, 168)

# WhatsApp re-encodes to roughly this.
WHATSAPP_LONG_EDGE = 1600
WHATSAPP_QUALITY = 80


def identity(img: Image.Image) -> Image.Image:
    return img.copy()


def rotate(img: Image.Image, degrees: int) -> Image.Image:
    ops = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    return img.transpose(ops[degrees])


def pad(
    img: Image.Image,
    colour: tuple[int, int, int],
    fraction: float = 0.15,
    axis: str = "vertical",
) -> Image.Image:
    """Add uniform bars, the way a screenshot gets letterboxed."""
    rgb = img.convert("RGB")
    width, height = rgb.size
    if axis == "vertical":
        bar = max(1, int(height * fraction))
        canvas = Image.new("RGB", (width, height + 2 * bar), colour)
        canvas.paste(rgb, (0, bar))
    else:
        bar = max(1, int(width * fraction))
        canvas = Image.new("RGB", (width + 2 * bar, height), colour)
        canvas.paste(rgb, (bar, 0))
    return canvas


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG so the result carries real compression artefacts."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    out = Image.open(buffer)
    out.load()
    return out


def resize(img: Image.Image, scale: float) -> Image.Image:
    width = max(1, int(img.width * scale))
    height = max(1, int(img.height * scale))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def crop(img: Image.Image, fraction: float) -> Image.Image:
    """Trim `fraction` off every edge -- the case the CNN stage exists for."""
    dx = int(img.width * fraction)
    dy = int(img.height * fraction)
    return img.crop((dx, dy, img.width - dx, img.height - dy))


def brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def whatsapp(img: Image.Image) -> Image.Image:
    """Downscale to a 1600px long edge and re-encode at q80."""
    longest = max(img.size)
    shrunk = img if longest <= WHATSAPP_LONG_EDGE else resize(img, WHATSAPP_LONG_EDGE / longest)
    return jpeg(shrunk, WHATSAPP_QUALITY)


TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "identity": identity,
    "rotate_90": lambda im: rotate(im, 90),
    "rotate_180": lambda im: rotate(im, 180),
    "rotate_270": lambda im: rotate(im, 270),
    "pad_black": lambda im: pad(im, (0, 0, 0)),
    "pad_white": lambda im: pad(im, (255, 255, 255)),
    "pad_colour": lambda im: pad(im, BAR_COLOUR, axis="horizontal"),
    "jpeg_q90": lambda im: jpeg(im, 90),
    "jpeg_q60": lambda im: jpeg(im, 60),
    "jpeg_q30": lambda im: jpeg(im, 30),
    "resize_75": lambda im: resize(im, 0.75),
    "resize_50": lambda im: resize(im, 0.50),
    "resize_25": lambda im: resize(im, 0.25),
    "crop_5": lambda im: crop(im, 0.05),
    "crop_10": lambda im: crop(im, 0.10),
    "crop_20": lambda im: crop(im, 0.20),
    "brightness_up": lambda im: brightness(im, 1.2),
    "brightness_down": lambda im: brightness(im, 0.8),
    "whatsapp": whatsapp,
}


def apply(name: str, img: Image.Image) -> Image.Image:
    return TRANSFORMS[name](img)
