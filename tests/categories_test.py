from __future__ import annotations

import json
import os
import tempfile

import pytest

from CCAgT_utils import categories
from CCAgT_utils.errors import FileTypeError


def test_init_Categories_Helper(categories_aux_data):

    ccagt_helper = categories.Helper(categories_aux_data)

    expected = categories_aux_data[:]
    expected.append({'id': 0,
                     'color': [0, 0, 0],
                     'name': 'background',
                     'minimal_area': 0})

    assert ccagt_helper.raw_helper == expected


def test_init_Categories_Helper_with_wrong_data_type():
    wrong_data = {'name': 'test', 'id': 0}

    with pytest.raises(ValueError):
        categories.Helper(wrong_data)


def test_names_by_id(categories_aux_data, get_categories_name):

    ccagt_helper = categories.Helper(categories_aux_data)
    expected = get_categories_name.copy()
    expected[0] = 'background'

    assert ccagt_helper.name_by_category_id == expected


def test_minimal_area_by_id(categories_aux_data, get_min_area):

    ccagt_helper = categories.Helper(categories_aux_data)
    expected = get_min_area.copy()
    expected[0] = 0

    assert ccagt_helper.min_area_by_category_id == expected


def test_colors_by_id(categories_aux_data, get_color_rgb):

    data = categories_aux_data[:]
    data.append({'id': 0,
                 'color': '#000000',
                 'name': 'background',
                 'minimal_area': 0})

    ccagt_helper = categories.Helper(data)
    expected = get_color_rgb.copy()
    expected[0] = [0, 0, 0]
    assert ccagt_helper.colors_by_category_id == expected


def test_wrong_color_format():
    with pytest.raises(TypeError):
        ccagt_helper = categories.Helper([{'id': 0, 'color': [127, 255], 'name': 'background'}])
        ccagt_helper.colors_by_category_id

    with pytest.raises(TypeError):
        ccagt_helper = categories.Helper([{'id': 0, 'color': 'FF', 'name': 'background'}])
        ccagt_helper.colors_by_category_id

    with pytest.raises(TypeError):
        ccagt_helper = categories.Helper([{'id': 0, 'color': 255, 'name': 'background'}])
        ccagt_helper.colors_by_category_id


def test_wrong_id_name():
    with pytest.raises(ValueError):
        categories.Helper([{'id': 0, 'name': 'wrong name'}])


def test_read_json():

    d = {'categories': [{'id': 0, 'color': [0, 255, 0], 'name': 'background'}]}

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, 'helper_file.json')
        with open(filename, 'w') as f:
            json.dump(d, f)

        ccagt_helper = categories.read_json(filename)

        assert ccagt_helper.raw_helper == d['categories']


def test_wrong_read_json():

    filename = 'helper_file'

    with pytest.raises(FileTypeError):
        categories.read_json(filename)
