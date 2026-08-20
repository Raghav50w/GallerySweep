"""pHash over the 4 rotations, and the cross-rotation match that makes it work."""

from __future__ import annotations

import numpy as np

from app.config import HASH_BITS
from app.hashing import hash_variants
from app.matching import hashes_to_array
from eval.augment import jpeg, rotate


def hamming(a: str, b: str) -> int:
    return int(np.bitwise_count(np.uint64(int(a, 16)) ^ np.uint64(int(b, 16))))


def test_hashes_are_16_char_hex(photo):
    hashes = hash_variants(photo)

    assert sorted(hashes) == [0, 90, 180, 270]
    for value in hashes.values():
        assert len(value) == 16
        assert int(value, 16) >= 0


def test_high_bit_hashes_round_trip_through_uint64():
    """Half of all pHashes have the high bit set; hex keeps them intact."""
    array = hashes_to_array([("ffffffffffffffff", "8000000000000000", "0", "1")])

    assert array.dtype == np.uint64
    assert array[0, 0] == np.uint64(0xFFFFFFFFFFFFFFFF)
    assert array[0, 1] == np.uint64(0x8000000000000000)


def test_a_rotated_copy_matches_at_the_corresponding_rotation(photo):
    """If A is B rotated 90 degrees, then A.h0 == B.h90."""
    original = hash_variants(photo)

    for degrees in (90, 180, 270):
        rotated = hash_variants(rotate(photo, degrees))
        assert hamming(rotated[0], original[degrees]) <= 2, f"{degrees} deg"


def test_a_rotated_copy_does_not_match_at_rotation_zero(photo):
    """Why normalization is needed at all: rotation scrambles the DCT bits."""
    original = hash_variants(photo)
    rotated = hash_variants(rotate(photo, 90))

    assert hamming(rotated[0], original[0]) > 12


def test_recompression_barely_moves_the_hash(photo):
    original = hash_variants(photo)[0]

    assert hamming(hash_variants(jpeg(photo, 60))[0], original) <= 6


def test_unrelated_photos_land_far_apart(photo, other_photo):
    distance = hamming(hash_variants(photo)[0], hash_variants(other_photo)[0])

    # Unrelated 64-bit hashes are ~Binomial(64, 0.5): mean 32, sd 4.
    assert distance > 20
    assert distance <= HASH_BITS
