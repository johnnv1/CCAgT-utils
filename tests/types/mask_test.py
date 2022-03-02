from __future__ import annotations

import numpy as np
import pytest

from CCAgT_utils.errors import ShapeError
from CCAgT_utils.types.mask import Mask


def test_wrong_init():
    with pytest.raises(ShapeError):
        Mask([1, 1])


def test_shape(mask, shape):
    assert mask.height == shape[0]
    assert mask.width == shape[1]


def test_unique_ids(mask):
    assert mask.unique_ids == {1, 2, 3, 4}


def test_colorized(mask, get_color_rgb, mask_colorized):
    colorized = mask.colorized(get_color_rgb)
    assert np.array_equal(colorized, mask_colorized)


def test_cmap(mask, get_color_rgba_norm):
    cmap = mask.cmap(get_color_rgba_norm)
    assert len(cmap.colors) == len(mask.unique_ids)
    assert cmap.colors[0] == get_color_rgba_norm[1]
