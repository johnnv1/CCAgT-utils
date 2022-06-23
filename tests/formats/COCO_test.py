from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.base.colors import Color
from CCAgT_utils.converters import COCO


def test_bounds_to_coco_bb():
    p = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    bounds = p.bounds
    coco_bb = COCO.bounds_to_coco_bb(bounds, decimals=2)

    assert coco_bb == [0., 0., 10, 10]


def test_geometry_to_coco_segment():
    coords_1 = [(0, 0), (0, 10), (10, 10), (10, 0)]
    p = Polygon(coords_1)
    seg = COCO.geometry_to_coco_segment(p)

    coords_1_out = [0., 0., 0., 10., 10., 10., 10., 0., 0., 0.]
    assert seg == [coords_1_out]

    coords_2 = [(20, 20), (20, 210), (210, 210), (210, 20)]

    mp = MultiPolygon([p, Polygon(coords_2)])
    seg = COCO.geometry_to_coco_segment(mp)

    coords_2_out = [20., 20., 20., 210., 210., 210., 210., 20., 20., 20.]
    assert seg == [coords_1_out, coords_2_out]


def test_wrong_geometry_to_coco_segment():
    with pytest.raises(TypeError):
        COCO.geometry_to_coco_segment(Point(0, 0))


def test_COCO_PS_color_to_id(categories_infos):
    c = Color(Red=10, Green=20, Blue=15)
    expected = 1314570
    out = COCO.color_to_id(c)
    assert out == expected
