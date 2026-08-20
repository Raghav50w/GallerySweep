"""ResNet-50 embeddings for the Smart-mode re-ranking stage.

The hash stage nominates candidate pairs; this module scores them. Only images
named by a candidate pair are ever embedded, which is the whole point of the
cascade -- embedding a 5,000-photo library outright would cost orders of
magnitude more forward passes than scoring the few hundred pairs that survive
the Hamming gate.

torch is imported lazily. `python -m app.main` must start in about a second,
and importing torch at module scope costs several.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable, Sequence

import numpy as np

from app.config import EMBED_BATCH_SIZE, EMBED_DIM
from app.normalize import normalize, rotations

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MODEL = None
_TRANSFORM = None

# (image_id, path, rotation_degrees)
Work = tuple[int, str, int]


def _load_model():
    """Build the feature extractor once, on first use.

    ResNet-50's `fc` is the 1000-way ImageNet classifier; replacing it with an
    identity exposes the 2048-d global-average-pooled features underneath. That
    layer is what "transfer learning" means here -- no training happens, the
    representation is reused as-is.
    """
    global _MODEL, _TRANSFORM
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _TRANSFORM

        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        log.info("loading %s (first run downloads the weights)", weights)
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()

        _MODEL = model
        # The weights carry their own preprocessing: resize 232, centre-crop
        # 224, ImageNet mean/std. Hand-rolling it is the classic way to get
        # embeddings that are subtly wrong and still look plausible.
        _TRANSFORM = weights.transforms()
        return _MODEL, _TRANSFORM


def _prepare(path: str, rotation: int):
    """Normalized, rotated, preprocessed tensor for one image.

    Reuses the same normalization the hash stage uses, so both engines see the
    same de-padded, EXIF-oriented picture. `rotation` comes from the pair row:
    the hash matched image B's 90-degree variant, so the CNN must compare that
    same variant or the two engines disagree about what they are looking at.
    """
    _, transform = _load_model()
    img, _, _ = normalize(path)
    if rotation:
        img = rotations(img)[rotation]
    return transform(img.convert("RGB"))


def embed_images(
    work: Sequence[Work],
    progress: Callable[[int, str | None], None] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Embed every (image, rotation) in `work`.

    Returns L2-normalized float32 vectors keyed by `(image_id, rotation)`.
    Normalizing here rather than at comparison time makes cosine similarity a
    plain dot product everywhere downstream.

    A file that fails to decode is skipped and simply absent from the result;
    real backups hold truncated downloads, and one of them must not take down a
    whole scan.
    """
    if not work:
        return {}

    import torch

    model, _ = _load_model()
    out: dict[tuple[int, int], np.ndarray] = {}

    batch: list = []
    keys: list[tuple[int, int]] = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode():
            features = model(torch.stack(batch))
        features = torch.nn.functional.normalize(features, dim=1)
        for key, vector in zip(keys, features.numpy().astype(np.float32)):
            out[key] = vector
        batch.clear()
        keys.clear()

    for image_id, path, rotation in work:
        try:
            batch.append(_prepare(path, rotation))
            keys.append((image_id, rotation))
        except Exception as exc:  # noqa: BLE001 - a bad file is skippable
            log.warning("could not embed %s: %s", path, exc)
            if progress is not None:
                progress(1, f"{path}: {exc}")
            continue

        if len(batch) >= EMBED_BATCH_SIZE:
            flush()
            if progress is not None:
                progress(EMBED_BATCH_SIZE, None)

    pending = len(batch)
    flush()
    if pending and progress is not None:
        progress(pending, None)

    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalized vectors."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def zero_vector() -> np.ndarray:
    return np.zeros(EMBED_DIM, dtype=np.float32)


def work_paths(work: Iterable[Work]) -> set[str]:
    """Distinct files a work list will open -- used for the compute-saved log."""
    return {path for _, path, _ in work}
