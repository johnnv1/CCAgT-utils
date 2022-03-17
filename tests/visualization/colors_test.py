from __future__ import annotations

import pytest

from CCAgT_utils.types import colors


def test_rgb_to_rgba():
    c1_rgb = [0, 255, 0]
    c1_rgba = colors.rgb_to_rgba(c1_rgb, normalize=False, bytes_precision=8, alpha_value=128)
    assert c1_rgba == [0, 255, 0, 128]

    c1_rgba = colors.rgb_to_rgba(c1_rgb, normalize=False, bytes_precision=8, alpha_value=128)
    assert c1_rgba == [0, 255, 0, 128]

    c1_rgba_normalized = colors.rgb_to_rgba(c1_rgb, normalize=True, bytes_precision=8, alpha_value=128)

    assert c1_rgba_normalized[:-1] == [0., 1., 0.]
    assert c1_rgba_normalized[-1] == pytest.approx(0.5, 0.51)

    c1_rgba_normalized = colors.rgb_to_rgba(c1_rgba_normalized, normalize=True, bytes_precision=8, alpha_value=128)
    assert c1_rgba_normalized[:-1] == [0., 1., 0.]
    assert c1_rgba_normalized[-1] == pytest.approx(0.5, 0.51)

    c2_rgb_normalized = [0.5, .75, .2]
    c2_rgba_normalized = colors.rgb_to_rgba(c2_rgb_normalized, normalize=True, bytes_precision=8, alpha_value=128)
    assert c2_rgba_normalized[:-1] == [0.5, .75, .2]
    assert c2_rgba_normalized[-1] == pytest.approx(0.5, 0.51)


def test_wrong_value_rgb_to_rgba():
    with pytest.raises(ValueError):
        colors.rgb_to_rgba([0])


def test_hex_to_rgb():
    c_hex = '#FFFFAA'
    c_rgb = colors.hex_to_rgb(c_hex)
    assert c_rgb == [255, 255, 170]
