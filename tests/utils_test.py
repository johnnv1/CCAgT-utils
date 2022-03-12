from __future__ import annotations

import os
import tempfile

import pytest

from CCAgT_utils import utils


def test_get_basename():
    filename = 'tmp/example/file.json'

    assert utils.basename(filename) == 'file'
    assert utils.basename(filename, True) == 'file.json'


def test_get_traceback():
    assert utils.get_traceback(lambda: 'test')() == 'test'


def test_get_traceback_raise():
    def raise_exception():
        raise Exception

    with pytest.raises(Exception):
        utils.get_traceback(raise_exception)()


def test_items_from_filename():
    filename = '/tmp/G_11_000_999.jpg'
    items = utils.items_from_filename(filename)

    assert items == ['G', '11', '000', '999']


def test_slide_from_filename():
    filename = '/tmp/G_11_000_999.jpg'
    slide_id = utils.slide_from_filename(filename)

    assert slide_id == 'G'


def test_find_files():

    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, 'test/')
        os.makedirs(subdir)
        filename_txt = os.path.join(tmpdir, 'file.txt')
        f = open(filename_txt, 'w')
        f.close()

        filename_bin = os.path.join(subdir, 'file.bin')
        f = open(filename_bin, 'w')
        f.close()

        files = utils.find_files(tmpdir, ('.txt', '.bin'), False)
        expected = {'file.txt': filename_txt}
        assert files == expected

        files = utils.find_files(tmpdir, ('.txt', '.bin'), True)

        expected['file.bin'] = filename_bin
        assert files == expected


def test_create_structure():
    with tempfile.TemporaryDirectory() as tmpdir:

        utils.create_structure(tmpdir, {'A'})
        assert os.path.isdir(os.path.join(tmpdir, 'images/A/'))
        assert os.path.isdir(os.path.join(tmpdir, 'masks/A/'))
