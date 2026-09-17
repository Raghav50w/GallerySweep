"""Cascade scoring: the CNN must compare the same rotation the hash matched."""

import numpy as np
import pytest

from app.matching import cosine_for_pairs


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_cosine_for_pairs_uses_the_rotation_hint():
    vectors = {
        (1, 0): unit(1.0, 0.0),
        (2, 90): unit(1.0, 0.0),   # the rotated variant that matched
        (2, 0): unit(0.0, 1.0),    # upright, and unlike image 1
    }
    scored = cosine_for_pairs([(1, 2, 0, 90)], vectors)
    assert scored == [(pytest.approx(1.0), 1, 2)]
