from __future__ import annotations

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


def test_find_files(tmpdir):
    subdir = tmpdir.mkdir('test')
    filename_txt = tmpdir.join('file.txt')
    filename_txt.write('content')

    filename_bin = subdir.join('file.bin')
    filename_bin.write('content')

    files = utils.find_files(str(tmpdir), ('.txt', '.bin'), False)
    expected = {'file.txt': str(filename_txt)}
    assert files == expected

    files = utils.find_files(str(tmpdir), ('.txt', '.bin'), True)

    expected['file.bin'] = str(filename_bin)
    assert files == expected


def test_create_structure(tmpdir):
    utils.create_structure(tmpdir, {'A'})
    assert tmpdir.join('images').join('A').check()
    assert tmpdir.join('masks').join('A').check()
