from __future__ import annotations

from CCAgT_utils.types import checkers


def test_is_2d():
    assert checkers.is_2d((100, 200))
    assert not checkers.is_2d((100, 200, 300))


def test_is_rgb_shape():
    assert checkers.is_rgb_shape((100, 200, 3))
    assert not checkers.is_rgb_shape((100, 200, 300))
    assert not checkers.is_rgb_shape((100, 200))
