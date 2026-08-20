"""256px webp thumbnail cache, generated on demand.

Nothing is ever written into the folder being cleaned -- thumbnails live in
`cache/thumbs/` inside the project.
"""

from __future__ import annotations

from pathlib import Path

from app.config import THUMB_QUALITY, THUMB_SIZE, THUMBS_DIR
from app.normalize import load_image


def thumb_path(image_id: int) -> Path:
    return THUMBS_DIR / f"{image_id}.webp"


def ensure_thumb(image_id: int, source: str | Path) -> Path:
    """Return the cached thumbnail, rendering it first if it is missing or stale."""
    source = Path(source)
    target = thumb_path(image_id)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    img = load_image(source)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    img.save(target, "WEBP", quality=THUMB_QUALITY)
    return target


def discard_thumb(image_id: int) -> None:
    thumb_path(image_id).unlink(missing_ok=True)
