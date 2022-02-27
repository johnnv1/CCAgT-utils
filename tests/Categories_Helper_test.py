from __future__ import annotations

import json
import os
import tempfile

import pytest

from CCAgT_utils import Categories
from CCAgT_utils.errors import FileTypeError


def test_init_Categories_Helper():
    data = [{'name': 'A', 'id': 0},
            {'name': 'B', 'id': 1}]

    ccagt_helper = Categories.Helper(data)
    assert ccagt_helper.raw_helper == data


def test_init_Categories_Helper_with_wrong_data_type():
    wrong_data = {'name': 'test', 'id': 0}

    with pytest.raises(ValueError):
        Categories.Helper(wrong_data)


def test_names_by_id():
    data = [{'name': 'A', 'id': 0},
            {'name': 'B', 'id': 1}]

    ccagt_helper = Categories.Helper(data)

    assert ccagt_helper.name_by_category_id == {0: 'A', 1: 'B'}


def test_minimal_area_by_id():
    data = [{'name': 'A', 'id': 0, 'minimal_area': 50},
            {'name': 'B', 'id': 1, 'minimal_area': 75}]

    ccagt_helper = Categories.Helper(data)

    assert ccagt_helper.min_area_by_category_id == {0: 50, 1: 75}


def test_colors_by_id():
    data = [{'id': 0, 'color': [127, 255, 200]},
            {'id': 1, 'color': '#FFFFFF'}]

    ccagt_helper = Categories.Helper(data)

    assert ccagt_helper.colors_by_category_id == {0: [127, 255, 200], 1: [255, 255, 255]}


def test_wrong_color_format():
    with pytest.raises(TypeError):
        ccagt_helper = Categories.Helper([{'id': 0, 'color': [127, 255]}])
        ccagt_helper.colors_by_category_id

    with pytest.raises(TypeError):
        ccagt_helper = Categories.Helper([{'id': 0, 'color': 'FF'}])
        ccagt_helper.colors_by_category_id

    with pytest.raises(TypeError):
        ccagt_helper = Categories.Helper([{'id': 0, 'color': 255}])
        ccagt_helper.colors_by_category_id


def test_init_Categories_Helper_without_zero_id():
    data = [{'name': 'A', 'id': 1}]

    ccagt_helper = Categories.Helper(data)

    t = False

    for x in ccagt_helper.raw_helper:
        if ((x['id'] == 0) and (x['name'].lower() == 'background')):
            t = True

    assert t


def test_read_json():

    d = {'categories': [{'id': 0, 'color': [0, 255, 0], 'name': 'A'}]}

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, 'helper_file.json')
        with open(filename, 'w') as f:
            json.dump(d, f)

        ccagt_helper = Categories.read_json(filename)

        assert ccagt_helper.raw_helper == d['categories']


def test_wrong_read_json():

    filename = 'helper_file'

    with pytest.raises(FileTypeError):
        Categories.read_json(filename)
