"""Candidate search, grouping and keeper selection.

The headline test is the one from the plan's verification section: from one base
image, every listed transform must end up in the same group, and a genuinely
unrelated photo must never join it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import FAST_HAMMING_THRESHOLD
from app.hashing import hash_variants
from app.matching import (
    _POPCOUNT_LUT,
    _popcount,
    assemble_groups,
    find_candidate_pairs,
    group_pairs,
    hamming_to_similarity,
    hashes_to_array,
    pick_keeper,
)
from app.normalize import border_crop
from eval.augment import TRANSFORMS

# Everything the plan requires to group. Crops are deliberately absent: a
# heavily cropped copy can sit ~25 bits away and never get nominated, which is
# the known blind spot the Phase 3 CNN stage addresses.
REQUIRED_TRANSFORMS = [
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "pad_black",
    "pad_white",
    "pad_colour",
    "jpeg_q60",
    "resize_50",
]


def hashes_for(images):
    """Normalize (border-crop) then hash, the same way the scanner does."""
    return hashes_to_array(
        [
            tuple(hash_variants(border_crop(img))[d] for d in (0, 90, 180, 270))
            for img in images
        ]
    )


def test_popcount_fallback_matches_the_native_path():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**63 - 1, size=(3, 5, 4), dtype=np.int64).astype(np.uint64)

    native = np.bitwise_count(values)
    fallback = (
        _POPCOUNT_LUT[np.ascontiguousarray(values).view(np.uint8).reshape(values.shape + (8,))]
        .sum(axis=-1, dtype=np.uint8)
    )

    assert np.array_equal(native, fallback)
    assert np.array_equal(_popcount(values), native)


def test_every_required_transform_groups_and_the_stranger_stays_out(photo, other_photo):
    variants = [TRANSFORMS[name](photo) for name in REQUIRED_TRANSFORMS]
    images = variants + [other_photo]
    stranger = len(images) - 1

    hashes = hashes_for(images)
    pairs = find_candidate_pairs(hashes, FAST_HAMMING_THRESHOLD)
    groups = group_pairs(len(images), [(int(a), int(b)) for a, b, *_ in pairs])

    assert len(groups) == 1, "the transforms should form exactly one group"
    group = set(groups[0])
    assert group == set(range(len(variants))), {
        REQUIRED_TRANSFORMS[i] for i in range(len(variants)) if i not in group
    }
    assert stranger not in group


def test_candidate_pairs_are_ordered_and_never_self_pairs(photo):
    images = [photo, TRANSFORMS["rotate_180"](photo), TRANSFORMS["jpeg_q60"](photo)]
    pairs = find_candidate_pairs(hashes_for(images), 16)

    assert len(pairs) > 0
    for a, b, _hamming, rotation_a, _rotation_b in pairs:
        assert a < b, "each pair is stored once, with image_a < image_b"
        assert rotation_a == 0

    seen = {(int(a), int(b)) for a, b, *_ in pairs}
    assert len(seen) == len(pairs), "no duplicate rows"


def test_a_rotated_copy_is_found_at_its_rotation(photo):
    images = [photo, TRANSFORMS["rotate_90"](photo)]
    pairs = find_candidate_pairs(hashes_for(images), FAST_HAMMING_THRESHOLD)

    assert len(pairs) == 1
    _a, _b, hamming, _rot_a, rotation_b = pairs[0]
    assert hamming <= FAST_HAMMING_THRESHOLD
    assert rotation_b in (90, 270)  # whichever direction lines the two up


def test_find_candidate_pairs_handles_a_single_image(photo):
    assert find_candidate_pairs(hashes_for([photo]), 16).shape == (0, 5)


def test_grouping_is_connected_components():
    groups = group_pairs(5, [(0, 1), (1, 2), (3, 4)])

    assert sorted(sorted(g) for g in groups) == [[0, 1, 2], [3, 4]]


def test_grouping_ignores_isolated_nodes():
    assert group_pairs(4, [(0, 1)]) == [[0, 1]]


def test_similarity_is_normalized():
    assert hamming_to_similarity(0) == 1.0
    assert hamming_to_similarity(64) == 0.0
    assert hamming_to_similarity(16) == pytest.approx(0.75)


def test_keeper_prefers_pixels_then_bytes_then_age():
    items = [
        {"width": 800, "height": 600, "size": 100, "mtime": 5},
        {"width": 1600, "height": 1200, "size": 50, "mtime": 9},   # most pixels
        {"width": 800, "height": 600, "size": 900, "mtime": 1},
    ]
    assert pick_keeper(items) == 1

    tied_pixels = [
        {"width": 800, "height": 600, "size": 100, "mtime": 5},
        {"width": 800, "height": 600, "size": 900, "mtime": 9},    # biggest file
    ]
    assert pick_keeper(tied_pixels) == 1

    tied_bytes = [
        {"width": 800, "height": 600, "size": 100, "mtime": 5},
        {"width": 800, "height": 600, "size": 100, "mtime": 1},    # oldest
    ]
    assert pick_keeper(tied_bytes) == 1


def test_assemble_groups_marks_one_keeper_and_checks_the_rest():
    images = {
        1: {"path": "a.jpg", "name": "a.jpg", "width": 4000, "height": 3000, "size": 900, "mtime": 1},
        2: {"path": "b.jpg", "name": "b.jpg", "width": 800, "height": 600, "size": 90, "mtime": 2},
        3: {"path": "c.jpg", "name": "c.jpg", "width": 800, "height": 600, "size": 80, "mtime": 3},
    }
    groups = assemble_groups([(1, 2, 0.98), (2, 3, 0.95)], images)

    assert len(groups) == 1
    group = groups[0]
    assert group["keeper_id"] == 1
    assert [f["keep"] for f in group["files"]] == [True, False, False]
    assert group["reclaimable"] == 170
    # 3 joined transitively, so it has no direct pair with the keeper.
    assert group["files"][0]["similarity"] == 1.0


def test_assemble_groups_skips_images_that_have_since_been_deleted():
    images = {
        1: {"path": "a.jpg", "name": "a.jpg", "width": 10, "height": 10, "size": 1, "mtime": 1},
    }
    assert assemble_groups([(1, 2, 0.99)], images) == []


def test_assemble_groups_is_indifferent_to_which_engine_scored_the_pairs():
    """Smart mode feeds cosines through the same grouping code path.

    `similarity` is 0-1 and engine-independent by design, so nothing downstream
    of scoring needed changing when the CNN landed.
    """
    images = {
        1: {"path": "a.jpg", "name": "a.jpg", "width": 4000, "height": 3000, "size": 900, "mtime": 1},
        2: {"path": "b.jpg", "name": "b.jpg", "width": 800, "height": 600, "size": 90, "mtime": 2},
    }
    from_hash = assemble_groups([(1, 2, hamming_to_similarity(4))], images)
    from_cnn = assemble_groups([(1, 2, 0.9375)], images)  # same number, cosine

    assert from_hash == from_cnn
    assert from_cnn[0]["keeper_id"] == 1
