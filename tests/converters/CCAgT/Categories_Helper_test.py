from __future__ import annotations

import pytest

from CCAgT_utils.converters.CCAgT import Categories_Helper


def test_init_Categories_Helper():
    data = [{'name': 'A', 'id': 0},
            {'name': 'B', 'id': 1}]

    ccagt_helper = Categories_Helper(data)
    assert ccagt_helper.raw_helper == data


def test_init_Categories_Helper_with_wrong_data_type():
    wrong_data = {'name': 'test', 'id': 0}

    with pytest.raises(ValueError):
        Categories_Helper(wrong_data)


def test_names_by_id():
    data = [{'name': 'A', 'id': 0},
            {'name': 'B', 'id': 1}]

    ccagt_helper = Categories_Helper(data)

    assert ccagt_helper.name_by_category_id == {0: 'A', 1: 'B'}


def test_minimal_area_by_id():
    data = [{'name': 'A', 'id': 0, 'minimal_area': 50},
            {'name': 'B', 'id': 1, 'minimal_area': 75}]

    ccagt_helper = Categories_Helper(data)

    assert ccagt_helper.min_area_by_category_id == {0: 50, 1: 75}
