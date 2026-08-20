"""border_crop is where the padded-screenshot feature lives or dies."""

from __future__ import annotations

import pytest
from PIL import Image

from app.normalize import border_crop, rotations
from conftest import add_dark_texture_border
from eval.augment import BAR_COLOUR, jpeg, pad


def area(img: Image.Image) -> int:
    return img.width * img.height


@pytest.mark.parametrize(
    ("colour", "axis"),
    [
        ((0, 0, 0), "vertical"),
        ((255, 255, 255), "vertical"),
        (BAR_COLOUR, "horizontal"),
    ],
)
def test_border_crop_removes_uniform_bars(photo, colour, axis):
    padded = pad(photo, colour, fraction=0.2, axis=axis)
    cropped = border_crop(padded)

    assert area(cropped) < area(padded)
    # Back to roughly the original frame, allowing a few pixels of slack where
    # the outermost content row happens to match the bar colour.
    assert cropped.width == pytest.approx(photo.width, abs=6)
    assert cropped.height == pytest.approx(photo.height, abs=6)


def test_border_crop_survives_jpeg_compression(photo):
    """The test that catches a `getbbox()`-only implementation.

    Compressed black bars hold values like 0, 2, 3, so an exact-match diff is
    non-zero everywhere and the bounding box is the whole image.
    """
    padded = jpeg(pad(photo, (0, 0, 0), fraction=0.2), quality=60)
    cropped = border_crop(padded)

    assert area(cropped) < area(padded) * 0.85
    assert cropped.height == pytest.approx(photo.height, abs=10)


def test_border_crop_is_a_noop_on_dark_textured_edges(photo):
    """Dark edges are not a border. Over-cropping real content is the bug."""
    textured = add_dark_texture_border(photo)
    assert border_crop(textured).size == textured.size


def test_border_crop_is_a_noop_below_the_area_threshold(photo):
    """A hairline frame is not padding worth removing."""
    framed = pad(photo, (0, 0, 0), fraction=0.004)
    assert border_crop(framed).size == framed.size


def test_border_crop_leaves_an_ordinary_photo_alone(photo):
    assert border_crop(photo).size == photo.size


def test_border_crop_handles_a_completely_flat_image():
    flat = Image.new("RGB", (100, 100), (12, 12, 12))
    assert border_crop(flat).size == flat.size


def test_rotations_are_the_four_axis_aligned_ones(photo):
    variants = rotations(photo)

    assert sorted(variants) == [0, 90, 180, 270]
    assert variants[0] is photo
    assert variants[90].size == (photo.height, photo.width)
    assert variants[180].size == photo.size
    assert list(variants[180].transpose(Image.Transpose.ROTATE_180).getdata()) == list(
        photo.convert(variants[180].mode).getdata()
    )
