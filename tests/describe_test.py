from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from CCAgT_utils import describe
from testing import create


def test_statistics():
    a = describe.Statistics()
    b = describe.Statistics(1., 1., 1., -1., 1)

    a.join_stats(b)

    assert a.count == 1
    assert a.mean == 1.
    a.join_stats(b)

    assert a.count == 2
    assert a.mean == 1.


def test_describe_from_array():
    arr = np.array([[10, 10, 10], [20, 20, 20], [1, 2, 3]])
    stats = describe.from_array(arr)

    assert stats.mean == pytest.approx(10.66, 0.01)
    assert stats.std == pytest.approx(7.378, 0.01)
    assert stats.max == 20
    assert stats.min == 1
    assert stats.count == 1


def test_single_core_from_image_files_empty():
    with pytest.raises(ValueError):
        describe.single_core_from_image_files([])


def test_single_core_from_image_files(shape, tmpdir):
    image = create.mask_colorized(shape)
    filename = os.path.join(tmpdir, 'test.png')
    Image.fromarray(image).save(filename)

    stats = describe.single_core_from_image_files([filename])
    assert np.array_equal(stats.mean, np.mean(image, axis=(0, 1)))
    assert np.array_equal(stats.std, np.std(image, axis=(0, 1)))
    assert np.array_equal(stats.max, np.max(image, axis=(0, 1)))
    assert np.array_equal(stats.min, np.min(image, axis=(0, 1)))
    assert stats.count == 1


@pytest.mark.slow
def test_from_image_files(shape, tmpdir):
    image = create.mask_colorized(shape)
    for i in range(2):
        Image.fromarray(image).save(os.path.join(tmpdir, f'test{i}.png'))

    stats = describe.from_image_files(tmpdir, '.png')
    assert np.array_equal(stats.mean, np.mean(image, axis=(0, 1)))
    assert np.array_equal(stats.std, np.std(image, axis=(0, 1)))
    assert np.array_equal(stats.max, np.max(image, axis=(0, 1)))
    assert np.array_equal(stats.min, np.min(image, axis=(0, 1)))
    assert stats.count == 2


def test_from_image_files_empty(tmpdir):
    stats = describe.from_image_files(tmpdir, '.WrongExtension')
    assert stats.count == 0
