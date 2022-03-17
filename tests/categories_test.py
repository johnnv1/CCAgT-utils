from __future__ import annotations

import json
import os
import tempfile

import pytest

from CCAgT_utils import categories
from CCAgT_utils.errors import FileTypeError


def test_init_CategoriesInfos(categories_aux_data):

    ccagt_helper = categories.CategoriesInfos(categories_aux_data)

    expected = categories_aux_data[:]
    expected.append({'id': 0,
                     'color': [0, 0, 0],
                     'name': 'background',
                     'minimal_area': 0})

    assert ccagt_helper._infos[0].id == expected[0]['id']
    assert ccagt_helper._infos[0].name == expected[0]['name']
    assert ccagt_helper._infos[-1].id == expected[-1]['id']
    assert ccagt_helper._infos[-1].name == expected[-1]['name']


def test_init_CategoriesInfos_with_wrong_data_type():
    wrong_data = {'name': 'test', 'id': 0}

    with pytest.raises(ValueError):
        categories.CategoriesInfos(wrong_data)


def test_names_by_id(categories_aux_data, get_categories_name):

    ccagt_helper = categories.CategoriesInfos(categories_aux_data)
    expected = get_categories_name.copy()
    expected[0] = 'background'

    assert ccagt_helper.name_by_category_id == expected


def test_minimal_area_by_id(categories_aux_data, get_min_area):

    ccagt_helper = categories.CategoriesInfos(categories_aux_data)
    expected = get_min_area.copy()
    expected[0] = 0

    assert ccagt_helper.min_area_by_category_id == expected


def test_colors_by_id(categories_aux_data, get_color_rgb):

    data = categories_aux_data[:]
    data.append({'id': 0,
                 'color': (0, 0, 0),
                 'name': 'background',
                 'minimal_area': 0})

    ccagt_helper = categories.CategoriesInfos(data)
    expected = get_color_rgb.copy()
    expected[0] = (0, 0, 0)
    out = {k: v.rgb for k, v in ccagt_helper.colors_by_category_id.items()}
    assert out == expected


def test_wrong_color_format():
    with pytest.raises(TypeError):
        ccagt_helper = categories.CategoriesInfos([{'id': 0, 'color': 255, 'name': 'background'}])
        ccagt_helper.colors_by_category_id


def test_wrong_id_name():
    with pytest.raises(ValueError):
        categories.CategoriesInfos([{'id': 0, 'name': 'wrong name', 'color': (0, 0, 0)}])


def test_read_json():

    d = {'categories': [{'id': 0, 'color': (0, 255, 0), 'name': 'background'}]}

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, 'helper_file.json')
        with open(filename, 'w') as f:
            json.dump(d, f)

        ccagt_helper = categories.read_json(filename)

    assert ccagt_helper._infos[0].id == d['categories'][0]['id']
    assert ccagt_helper._infos[0].name == d['categories'][0]['name']


def test_wrong_read_json():

    filename = 'helper_file'

    with pytest.raises(FileTypeError):
        categories.read_json(filename)
