from __future__ import annotations

import numpy as np
import pytest

from CCAgT_utils.base import colors


def test_Color_with_wrong_values():
    with pytest.raises(ValueError):
        colors.Color(300, 0, 0, size=8).rgb_normalized
    with pytest.raises(ValueError):
        colors.Color(0, 300, 0, size=8).rgb_normalized
    with pytest.raises(ValueError):
        colors.Color(0, 0, 300, size=8).rgb_normalized
    with pytest.raises(ValueError):
        colors.Color(0, 0, 0, 300, size=8).rgb_normalized


def test_Color_rgb():
    assert colors.Color(Red=111, Green=222, Blue=25).rgb == (111, 222, 25)


def test_Color_rgb_normalizado():
    assert colors.Color(
        Red=111, Green=222, Blue=25, size=8,
    ).rgb_normalized == (111 / 255, 222 / 255, 25 / 255)


def test_Color_rgba():
    assert colors.Color(
        Red=111, Green=222, Blue=25,
        Alpha=10,
    ).rgba == (111, 222, 25, 10)


def test_Color_rgba_normalizado():
    assert colors.Color(
        Red=111,
        Green=222,
        Blue=25,
        Alpha=10,
        size=8,
    ).rgba_normalized == (
        111 / 255,
        222 / 255,
        25 / 255,
        10 / 255,
    )


def test_from_tuple():
    assert colors.from_tuple((10, 11, 12), 13, 8) == colors.Color(
        Red=10, Green=11, Blue=12, Alpha=13, size=8,
    )


def test_hex_to_rgb():
    c_hex = '#FFFFAA'
    c_rgb = colors.hex_to_rgb(c_hex)
    assert c_rgb == (255, 255, 170)


@pytest.mark.parametrize(
    'rgb, max_dist', [
        ((111, 222, 25), 10),
        ((222, 0, 80), 50),
        ((25, 127, 255), 30),
    ],
)
def test_random_color_from_base(rgb, max_dist):
    r, g, b = rgb
    base_color = colors.Color(Red=r, Green=g, Blue=b)
    rand_color = colors.random_color_from_base(base_color, max_dist=max_dist)
    expected_r, expected_g, expected_b = base_color.rgb
    out_r, out_g, out_b = rand_color.rgb
    assert np.minimum(0, expected_r - max_dist) <= out_r <= np.minimum(
        base_color._max_value, expected_r + max_dist,
    )
    assert np.minimum(0, expected_g - max_dist) <= out_g <= np.minimum(
        base_color._max_value, expected_g + max_dist,
    )
    assert np.minimum(0, expected_b - max_dist) <= out_b <= np.minimum(
        base_color._max_value, expected_b + max_dist,
    )
