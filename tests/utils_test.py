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
