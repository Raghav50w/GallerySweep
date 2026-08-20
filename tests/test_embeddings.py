"""Cascade scoring logic, plus one opt-in test that runs the real model.

The default run must not download ResNet-50's 100 MB of weights, so everything
here except the `slow` test works on hand-made vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.embeddings import cosine
from app.matching import cosine_for_pairs, embedding_work_set
from app.normalize import rotations


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# --- work set ---------------------------------------------------------------

def test_work_set_deduplicates():
    pairs = [(1, 2, 0, 0), (1, 3, 0, 0), (2, 3, 0, 0)]
    assert embedding_work_set(pairs) == [(1, 0), (2, 0), (3, 0)]


def test_work_set_keeps_both_orientations_of_one_image():
    """Image 2 is an upright `a` in one pair and a sideways `b` in another.

    Both entries must survive: comparing image 2 upright against a partner the
    hash matched at 90 degrees would throw away a real duplicate.
    """
    pairs = [(2, 5, 0, 0), (1, 2, 0, 90)]
    work = embedding_work_set(pairs)
    assert (2, 0) in work and (2, 90) in work
    assert len(work) == 4


def test_work_set_is_smaller_than_the_library():
    """The cascade's premise: only candidates get embedded."""
    pairs = [(1, 2, 0, 0), (3, 4, 0, 180)]
    assert len(embedding_work_set(pairs)) == 4  # not the 50 images in the folder


# --- scoring ----------------------------------------------------------------

def test_cosine_of_identical_and_orthogonal_vectors():
    a = unit(1.0, 0.0, 0.0)
    b = unit(0.0, 1.0, 0.0)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_stays_in_range():
    a = unit(1.0, 1.0)
    assert -1.0 <= cosine(a, -a) <= 1.0


def test_cosine_for_pairs_uses_the_rotation_hint():
    vectors = {
        (1, 0): unit(1.0, 0.0),
        (2, 90): unit(1.0, 0.0),   # the rotated variant that matched
        (2, 0): unit(0.0, 1.0),    # upright, and unlike image 1
    }
    scored = cosine_for_pairs([(1, 2, 0, 90)], vectors)
    assert scored == [(pytest.approx(1.0), 1, 2)]


def test_missing_vector_scores_none():
    """An unreadable file leaves the pair NULL rather than dropping it."""
    scored = cosine_for_pairs([(1, 2, 0, 0)], {(1, 0): unit(1.0, 0.0)})
    assert scored == [(None, 1, 2)]


def test_scores_come_back_in_db_update_order():
    """`(cosine, image_a, image_b)` is what `db.update_cosines` binds."""
    vectors = {(1, 0): unit(1.0, 0.0), (2, 0): unit(1.0, 0.0)}
    score, image_a, image_b = cosine_for_pairs([(1, 2, 0, 0)], vectors)[0]
    assert (image_a, image_b) == (1, 2)
    assert score == pytest.approx(1.0)


# --- the real model ---------------------------------------------------------

@pytest.mark.slow
def test_real_embeddings_match_a_rotated_copy(tmp_path, photo):
    """End-to-end through torch: a photo and its own 90-degree copy.

    This is the test that would catch a rotation-hint bug -- the pure-logic
    tests above cannot, because they never build a real vector.
    """
    from app.embeddings import embed_images

    original = tmp_path / "original.jpg"
    photo.save(original, quality=95)

    rotated = tmp_path / "rotated.jpg"
    rotations(photo)[90].save(rotated, quality=95)

    vectors = embed_images(
        [(1, str(original), 0), (2, str(rotated), 90), (2, str(rotated), 0)]
    )

    matched = cosine(vectors[(1, 0)], vectors[(2, 90)])
    assert matched > 0.9
    # Sanity: the rotation hint is doing work, not passing by accident.
    assert cosine(vectors[(1, 0)], vectors[(2, 0)]) < matched


@pytest.mark.slow
def test_real_embeddings_are_unit_length(tmp_path, photo):
    from app.embeddings import embed_images

    path = tmp_path / "photo.jpg"
    photo.save(path, quality=95)
    vectors = embed_images([(1, str(path), 0)])

    assert np.linalg.norm(vectors[(1, 0)]) == pytest.approx(1.0, abs=1e-5)
