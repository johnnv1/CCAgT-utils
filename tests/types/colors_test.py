from __future__ import annotations

from CCAgT_utils.types import colors


def test_hex_to_rgb():
    c_hex = '#FFFFAA'
    c_rgb = colors.hex_to_rgb(c_hex)
    assert c_rgb == (255, 255, 170)
