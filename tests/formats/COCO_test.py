from __future__ import annotations

from CCAgT_utils.base.colors import Color
from CCAgT_utils.formats.coco import color_to_id


def test_color_to_id():
    c = Color(Red=10, Green=20, Blue=15)
    expected = 1314570
    out = color_to_id(c)
    assert out == expected
