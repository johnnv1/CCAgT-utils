from __future__ import annotations

from CCAgT_utils import CCAgT


def test_items_from_filename():
    filename = "/tmp/G_11_000_999.jpg"
    items = CCAgT.items_from_filename(filename)

    assert items == ["G", '11', '000', '999']


def test_slide_from_filename():
    filename = "/tmp/G_11_000_999.jpg"
    slide_id = CCAgT.slide_from_filename(filename)

    assert slide_id == "G"
