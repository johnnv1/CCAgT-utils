from __future__ import annotations

import json
import os

import numpy as np
import pytest

from CCAgT_utils.categories import Categories
from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.categories import CategoryInfo
from CCAgT_utils.categories import CATS_COLORS
from CCAgT_utils.categories import read_json
from CCAgT_utils.errors import FileTypeError
from CCAgT_utils.types.colors import Color


def test_CategoriesInfos_init_without_data():
    categories_infos = CategoriesInfos()
    assert categories_infos[0].labelbox_schemaId is None


def test_CategoriesInfos_init_with_data(categories_aux_data):
    id_with_labelbox_schemaId = categories_aux_data[0]['id']
    categories_infos = CategoriesInfos(categories_aux_data)
    assert categories_infos[id_with_labelbox_schemaId].labelbox_schemaId is not None


def test_CategoriesInfos_init_with_data_with_background(categories_aux_data):
    id_with_labelbox_schemaId = categories_aux_data[0]['id']
    data = categories_aux_data[:]
    data.append({
        'id': 0,
        'color': (0, 0, 0),
        'name': 'background',
        'minimal_area': 0,
    })
    categories_infos = CategoriesInfos(data)
    assert categories_infos[id_with_labelbox_schemaId].labelbox_schemaId is not None


def test_CategoriesInfos_init_with_data_without_color(categories_aux_data):
    del categories_aux_data[0]['color']
    with pytest.raises(KeyError):
        CategoriesInfos(categories_aux_data)


def test_CategoriesInfos_init_with_wrong_category(categories_aux_data):
    categories_aux_data[0]['name'] = 'wrong category'
    with pytest.raises(ValueError):
        CategoriesInfos(categories_aux_data)


def test_CategoriesInfos_init_with_wrong_data():
    with pytest.raises(ValueError):
        CategoriesInfos({})


def test_CategoriesInfos_len(categories_aux_data):
    categories_infos = CategoriesInfos(categories_aux_data)
    assert len(categories_infos) == len(categories_aux_data) + 1


def test_CategoriesInfos_iter(categories_aux_data):
    categories_infos = CategoriesInfos(categories_aux_data)
    categories_ids = [c.id for c in categories_infos]
    expected = [i['id'] for i in categories_aux_data] + [0]
    assert categories_ids == expected


def test_CategoriesInfos_get_min_area(categories_aux_data):
    categories_infos = CategoriesInfos(categories_aux_data)
    minimal_area = [
        categories_infos.get_min_area(1),
        categories_infos.get_min_area(2),
    ]
    expected = [i['minimal_area'] for i in categories_aux_data if i['id'] in {1, 2}]

    assert minimal_area == expected


def test_CategoriesInfos_get_name(categories_aux_data):
    categories_infos = CategoriesInfos(categories_aux_data)
    names = [
        categories_infos.get_name(1),
        categories_infos.get_name(2),
    ]
    expected = [i['name'] for i in categories_aux_data if i['id'] in {1, 2}]

    assert names == expected


def test_CategoriesInfos_get_color(categories_aux_data):
    categories_infos = CategoriesInfos(categories_aux_data)
    colors = [
        categories_infos.get_color(1),
        categories_infos.get_color(2),
    ]
    expected = [i['color'] for i in categories_aux_data if i['id'] in {1, 2}]

    assert colors == expected


def test_CategoriesInfos_generate_random_color(categories_aux_data, monkeypatch):
    c = (127, 127, 127)
    c_color = Color(c[0], c[1], c[2])

    categories_aux_data.append({
        'name': 'background',
        'id': 0,
        'color': c,
        'isthing': 0,
    })
    categories_infos = CategoriesInfos(categories_aux_data)

    color = categories_infos.generate_random_color(1)
    assert color == CATS_COLORS[Categories.NUCLEUS]
    assert color.rgb in categories_infos.taken_colors

    assert c in categories_infos.taken_colors
    color = categories_infos.generate_random_color(0)
    assert color == c_color

    color = categories_infos.generate_random_color(1)
    nucleus_color = CATS_COLORS[Categories.NUCLEUS]
    expected_r, expected_g, expected_b = nucleus_color.rgb
    out_r, out_g, out_b = color.rgb
    assert color != nucleus_color

    assert np.minimum(0, expected_r - 30) <= out_r <= np.minimum(color._max_value, expected_r + 30)
    assert np.minimum(0, expected_g - 30) <= out_g <= np.minimum(color._max_value, expected_g + 30)
    assert np.minimum(0, expected_b - 30) <= out_b <= np.minimum(color._max_value, expected_b + 30)


def test_read_json(tmpdir):
    d = {'categories': [{'id': 0, 'color': (0, 255, 0), 'name': 'background'}]}

    filename = os.path.join(tmpdir, 'helper_file.json')
    with open(filename, 'w') as f:
        json.dump(d, f)

    ccagt_helper = read_json(filename)

    assert ccagt_helper._infos[0].id == d['categories'][0]['id']
    assert ccagt_helper._infos[0].name == d['categories'][0]['name']


def test_wrong_read_json():

    filename = 'helper_file'

    with pytest.raises(FileTypeError):
        read_json(filename)


def test_getCategory_from_categoryInfo():
    cat = Categories.BACKGROUND
    cat_info = CategoryInfo(
        id=cat.value,
        name=cat.name,
        color=CATS_COLORS[cat],
    )

    assert cat_info.category == cat
