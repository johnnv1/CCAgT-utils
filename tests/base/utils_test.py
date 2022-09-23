from __future__ import annotations

import pytest

from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.base.utils import basename
from CCAgT_utils.base.utils import create_structure
from CCAgT_utils.base.utils import find_files
from CCAgT_utils.base.utils import get_traceback
from CCAgT_utils.base.utils import items_from_filename
from CCAgT_utils.base.utils import open_and_read_json
from CCAgT_utils.base.utils import slide_from_filename


def test_get_basename():
    filename = 'tmp/example/file.json'

    assert basename(filename) == 'file'
    assert basename(filename, True) == 'file.json'


def test_get_traceback():
    assert get_traceback(lambda: 'test')() == 'test'


def test_get_traceback_raise():
    def raise_exception():
        raise Exception

    with pytest.raises(Exception):
        get_traceback(raise_exception)()


def test_items_from_filename():
    filename = '/tmp/G_11_000_999.jpg'
    items = items_from_filename(filename)

    assert items == ['G', '11', '000', '999']


def test_slide_from_filename():
    filename = '/tmp/G_11_000_999.jpg'
    slide_id = slide_from_filename(filename)

    assert slide_id == 'G'


def test_find_files(tmpdir):
    subdir = tmpdir.mkdir('test')
    filename_txt = tmpdir.join('file.txt')
    filename_txt.write('content')

    filename_bin = subdir.join('file.bin')
    filename_bin.write('content')

    files = find_files(str(tmpdir), ('.txt', '.bin'), False)
    expected = {'file.txt': str(filename_txt)}
    assert files == expected

    files = find_files(str(tmpdir), ('.txt', '.bin'), True)

    expected['file.bin'] = str(filename_bin)
    assert files == expected


def test_create_structure(tmpdir):
    create_structure(tmpdir, {'A'})
    assert tmpdir.join('images').join('A').check()
    assert tmpdir.join('masks').join('A').check()


def test_open_and_read_json(lbox_raw_sample_complete, lbox_sample_complete):
    out = open_and_read_json(lbox_sample_complete)
    assert out == lbox_raw_sample_complete


def test_open_and_read_json_wrong_type():
    with pytest.raises(FileTypeError):
        open_and_read_json('filename')
