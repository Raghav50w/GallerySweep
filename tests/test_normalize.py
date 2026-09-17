"""border_crop is where the padded-screenshot feature lives or dies."""

import pytest
from PIL import Image

from app.normalize import border_crop
from conftest import add_dark_texture_border
from eval.augment import BAR_COLOUR, pad


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


def test_border_crop_is_a_noop_on_dark_textured_edges(photo):
    """Dark edges are not a border. Over-cropping real content is the bug."""
    textured = add_dark_texture_border(photo)
    assert border_crop(textured).size == textured.size
