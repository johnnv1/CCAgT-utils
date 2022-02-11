from __future__ import annotations

from CCAgT_utils import utils


def test_get_basename():
    filename = 'tmp/example/file.json'

    assert utils.basename(filename) == 'file'
    assert utils.basename(filename, True) == 'file.json'
