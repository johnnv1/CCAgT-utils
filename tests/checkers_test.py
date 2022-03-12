from __future__ import annotations

import os
import tempfile

import pytest

from CCAgT_utils import checkers
from testing import create


def test_has_all_into_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = open(os.path.join(tmpdir, 'file.png'), 'w')
        f.close()

        f = open(os.path.join(tmpdir, 'file.jpg'), 'w')
        f.close()

        assert checkers.has_all_into_dir(tmpdir, ['file.png'], extension=('.png'))


def test_single_core_mask_has(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['test'], create_image=False) as paths:
        _, mask_dir, _ = paths
        o = checkers.single_core_mask_has({os.path.join(mask_dir, 'test.png')}, {0})
        assert len(o) == 0

        o = checkers.single_core_mask_has({os.path.join(mask_dir, 'test.png')}, {0, 1})
        assert len(o) == 1
        assert o == {'test'}


@pytest.mark.slow
def test_masks_that_has(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['test'], create_image=False) as paths:
        _, mask_dir, _ = paths
        o = checkers.masks_that_has(mask_dir, {0})
        assert len(o) == 0

        o = checkers.masks_that_has(mask_dir, {0, 1})
        assert len(o) == 1
        assert o == {'test'}
