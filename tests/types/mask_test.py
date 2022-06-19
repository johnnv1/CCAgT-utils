from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

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


def test_colorized(mask, categories_infos, mask_colorized):
    colorized = mask.colorized(categories_infos)
    assert np.array_equal(colorized, mask_colorized)


def test_cmap(mask, categories_infos, get_color_rgba_norm):
    cmap = mask.cmap(categories_infos)
    assert len(cmap.colors) == len(mask.unique_ids)
    assert cmap.colors[0] == get_color_rgba_norm[1]


def test_save(mask, tmpdir):
    out_path = os.path.join(tmpdir, 'test.png')
    mask.save(out_path)
    assert os.path.isfile(out_path)

    msk = Image.open(out_path).convert('L')
    assert np.array_equal(mask.categorical, msk)


@pytest.mark.slow
def test_save_colorized(mask, categories_infos, tmpdir):
    out_path = os.path.join(tmpdir, 'test_colorized.png')
    mask.save(out_path, categories_infos)
    msk = Image.open(out_path)
    assert os.path.isfile(out_path)
    assert np.array_equal(mask.colorized(categories_infos), msk)
