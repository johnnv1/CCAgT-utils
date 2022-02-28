from __future__ import annotations

import copy

import pytest
from shapely.geometry import Polygon

from CCAgT_utils.types import annotation


@pytest.fixture
def bbox_params():
    x = {'x_init': 100,
         'y_init': 500,
         'width': 35,
         'height': 88,
         'category_id': 0}
    x['x_end'] = x['x_init'] + x['width']
    x['y_end'] = x['y_init'] + x['height']
    return x


@pytest.fixture
def bbox_example(bbox_params):
    params = bbox_params.copy()
    del params['x_end']
    del params['y_end']
    return annotation.BBox(**params)


def test_x_end(bbox_example, bbox_params):
    assert bbox_example.x_end == bbox_params['x_end']


def test_y_end(bbox_example, bbox_params):
    assert bbox_example.y_end == bbox_params['y_end']


def test_upper_left_point(bbox_example, bbox_params):
    assert bbox_example.upper_left_point == (bbox_params['x_init'], bbox_params['y_init'])


def test_bottom_left_point(bbox_example, bbox_params):
    assert bbox_example.bottom_left_point == (bbox_params['x_init'], bbox_params['y_end'])


def test_upper_right_point(bbox_example, bbox_params):
    assert bbox_example.upper_right_point == (bbox_params['x_end'], bbox_params['y_init'])


def test_bottom_right_point(bbox_example, bbox_params):
    assert bbox_example.bottom_right_point == (bbox_params['x_end'], bbox_params['y_end'])


def test_coords(bbox_example, bbox_params):
    x_init = bbox_params['x_init']
    y_init = bbox_params['y_init']
    x_end = bbox_params['x_end']
    y_end = bbox_params['y_end']

    assert bbox_example.coords == [(x_init, y_init), (x_end, y_init),
                                   (x_end, y_end), (x_init, y_end)]


def test_xy(bbox_example, bbox_params):
    x_init = bbox_params['x_init']
    y_init = bbox_params['y_init']
    x_end = bbox_params['x_end']
    y_end = bbox_params['y_end']
    assert bbox_example.xy == ([x_init, x_end, x_end, x_init],
                               [y_init, y_init, y_end, y_end])


def test_center_point(bbox_example, bbox_params):
    x_center = bbox_params['x_init'] + (bbox_params['width']) // 2
    y_center = bbox_params['y_init'] + (bbox_params['height']) // 2

    assert bbox_example.center_point() == (x_center, y_center)


def test_area(bbox_example, bbox_params):
    assert bbox_example.area() == bbox_params['width'] * bbox_params['height']


def test_to_polygon(bbox_example, bbox_params):
    x_init = bbox_params['x_init']
    y_init = bbox_params['y_init']
    x_end = bbox_params['x_end']
    y_end = bbox_params['y_end']
    coords = [(x_init, y_init), (x_end, y_init),
              (x_end, y_end), (x_init, y_end)]
    p = Polygon(coords)
    assert bbox_example.to_polygon().equals(p)


def test_bounds_to_BBox(bbox_example, bbox_params):
    bounds = (bbox_params['x_init'], bbox_params['y_init'],
              bbox_params['x_end'], bbox_params['y_end'])

    assert bbox_example == annotation.bounds_to_BBox(bounds, bbox_params['category_id'])


def test_count_BBox_categories(bbox_example):
    cat_id_example = bbox_example.category_id
    bbox_example1 = copy.copy(bbox_example)
    bbox_example1.category_id = cat_id_example + 1
    cat_id_example1 = bbox_example1.category_id

    items = [bbox_example, bbox_example, bbox_example,
             bbox_example1, bbox_example1]

    categories_names = {cat_id_example: f'cat {cat_id_example}',
                        cat_id_example1: f'cat {cat_id_example1}',
                        cat_id_example1 + 10: 'not show at result'}

    counter = annotation.count_BBox_categories(items, categories_names)

    assert counter == {f'cat {cat_id_example}': 3, f'cat {cat_id_example1}': 2}
