"""Candidate search, grouping and keeper selection.

The headline test is the one from the plan's verification section: from one base
image, every listed transform must end up in the same group, and a genuinely
unrelated photo must never join it.
"""

from app.config import FAST_HAMMING_THRESHOLD
from app.hashing import hash_variants
from app.matching import (
    find_candidate_pairs,
    group_pairs,
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
