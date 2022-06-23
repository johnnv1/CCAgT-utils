from __future__ import annotations

import copy

import pytest
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.formats import annotation


@pytest.fixture
def bbox_params():
    x = {
        'x_init': 100,
        'y_init': 500,
        'width': 35,
        'height': 88,
        'category_id': 1,
    }
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

    assert bbox_example.coords == [
        (x_init, y_init), (x_end, y_init),
        (x_end, y_end), (x_init, y_end),
    ]


def test_xy(bbox_example, bbox_params):
    x_init = bbox_params['x_init']
    y_init = bbox_params['y_init']
    x_end = bbox_params['x_end']
    y_end = bbox_params['y_end']
    assert bbox_example.xy == (
        [x_init, x_end, x_end, x_init],
        [y_init, y_init, y_end, y_end],
    )


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
    coords = [
        (x_init, y_init), (x_end, y_init),
        (x_end, y_end), (x_init, y_end),
    ]
    p = Polygon(coords)
    assert bbox_example.to_polygon().equals(p)


def test_bounds_to_BBox(bbox_example, bbox_params):
    bounds = (
        bbox_params['x_init'], bbox_params['y_init'],
        bbox_params['x_end'], bbox_params['y_end'],
    )

    assert bbox_example == annotation.bounds_to_BBox(bounds, bbox_params['category_id'])


def test_slices(bbox_example, bbox_params):
    assert bbox_example.slice_y == slice(bbox_params['y_init'], bbox_params['y_end'])
    assert bbox_example.slice_x == slice(bbox_params['x_init'], bbox_params['x_end'])


def test_count_BBox_categories(bbox_example):
    cat_id_example = bbox_example.category_id
    bbox_example1 = copy.copy(bbox_example)
    bbox_example1.category_id = cat_id_example + 1
    cat_id_example1 = bbox_example1.category_id

    items = [
        bbox_example, bbox_example, bbox_example,
        bbox_example1, bbox_example1,
    ]

    categories_infos = CategoriesInfos([
        {'name': 'Nucleus', 'id': cat_id_example, 'color': (0, 0, 0)},
        {'name': 'Cluster', 'id': cat_id_example1, 'color': (0, 0, 0)},
        {'name': 'Satellite', 'id': cat_id_example1 + 1, 'color': (0, 0, 0)},
    ])

    counter = annotation.count_BBox_categories(items, categories_infos)

    assert counter == {'Nucleus': 3, 'Cluster': 2}


def test_fit_inside(bbox_example, bbox_params):
    x_init, y_init, x_end, y_end = (
        bbox_params['x_init'], bbox_params['y_init'],
        int(bbox_params['x_end'] * 0.8), int(bbox_params['y_end'] * 0.8),
    )

    bbox_example.fit_inside((x_init, y_init, x_end, y_end))

    assert bbox_example.x_init == x_init
    assert bbox_example.y_init == y_init
    assert bbox_example.x_end == x_end
    assert bbox_example.y_end == y_end


def test_add_padding(bbox_example):
    bbox_example1 = copy.copy(bbox_example)
    bbox_example1.add_padding()
    assert bbox_example1.coords == bbox_example.coords

    with pytest.raises(TypeError):
        bbox_example.add_padding('1')


@pytest.mark.parametrize('padding', [1, 5, 50, 5000])
def test_add_padding_in_pixel(bbox_example, bbox_params, padding):
    x_init_expected = bbox_params['x_init'] - padding
    y_init_expected = bbox_params['y_init'] - padding
    x_end_expected = bbox_params['x_end'] + padding
    y_end_expected = bbox_params['y_end'] + padding
    bbox_example.add_padding(
        padding, (
            x_init_expected - 100,
            y_init_expected - 100,
            x_end_expected + 100,
            y_end_expected + 100,
        ),
    )

    assert bbox_example.x_init == x_init_expected
    assert bbox_example.y_init == y_init_expected
    assert bbox_example.x_end == x_end_expected
    assert bbox_example.y_end == y_end_expected


@pytest.mark.parametrize('padding', [.1, .5, 5.0, 50.00])
def test_add_padding_in_percentage(bbox_example, bbox_params, padding):
    x_init_expected = int(bbox_params['x_init'] - (bbox_params['width'] * padding))
    y_init_expected = int(bbox_params['y_init'] - (bbox_params['height'] * padding))
    x_end_expected = int(bbox_params['x_end'] + (bbox_params['width'] * padding))
    y_end_expected = int(bbox_params['y_end'] + (bbox_params['height'] * padding))

    bbox_example.add_padding(
        padding, (
            x_init_expected - 100,
            y_init_expected - 100,
            x_end_expected + 100,
            y_end_expected + 100,
        ),
    )

    assert bbox_example.x_init == x_init_expected
    assert bbox_example.y_init == y_init_expected
    assert bbox_example.x_end == x_end_expected
    assert bbox_example.y_end == y_end_expected


def test_annotation_bbox(nucleus_ex):
    ann = annotation.Annotation(nucleus_ex, 1)

    assert ann.bbox.to_polygon().equals(nucleus_ex)


def test_annotation_geo_type(nucleus_ex):
    ann = annotation.Annotation(nucleus_ex, 1)
    geo_type = ann._geo_type
    assert geo_type == 'Polygon'


def test_annotation_coco_bbox(nucleus_ex):
    ann = annotation.Annotation(nucleus_ex, 1)
    min_x, min_y, max_x, max_y = nucleus_ex.bounds
    assert ann.coco_bbox == [min_x, min_y, int(max_x) - int(min_x), int(max_y) - int(min_y)]


def test_annotation_iter(nucleus_ex, cluster_ex):
    ann = annotation.Annotation(nucleus_ex, 1)

    assert len([geo for geo in ann]) == 1

    mult_p = MultiPolygon([nucleus_ex, cluster_ex])
    ann = annotation.Annotation(mult_p, 1)

    assert len([geo for geo in ann]) == 2


def test_annotation_iter_wrong_geo(satellite_ex):
    with pytest.raises(TypeError):
        iter(annotation.Annotation(satellite_ex, 3))


def test_annotation_copy(nucleus_ex):
    ann = annotation.Annotation(nucleus_ex, 1)
    ann2 = ann.copy()

    assert ann == ann2
    ann2.geometry = ann2.geometry.buffer(10)
    assert not ann.geometry.equals(ann2.geometry)
