from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.converters.LabelBox import LabelBox_Annotations


def test_labelbox_to_dataFrame():
    raw_data = [{'col A': 0, 'col B': 1}, {'col A': 2, 'col B': 3}]
    raw_df = pd.DataFrame(raw_data)
    lbb_ann = LabelBox_Annotations(raw_data)

    assert lbb_ann.raw_dataframe.equals(raw_df)


def test_init_without_raw():
    with pytest.raises(ValueError):
        LabelBox_Annotations(raw_labelbox=None)


def test_instance_categories_map():
    raw_data = [{'col A': 0, 'cat': 'A'}, {'col A': 2, 'cat': 'B'}]
    aux_data = [{"color": [21, 62, 125], "name": "A", "id": 1, "labelbox_schemaId": "<Unique ID for category A>"}]
    lb_ann = LabelBox_Annotations(raw_data, aux_data)

    assert lb_ann.categories_map == aux_data


def test_labelbox_object_to_shapely():
    obj = {"polygon": [{"x": 10, "y": 10}, {"x": 20, "y": 20}, {"x": 25, "y": 15}, {"x": 15, "y": 10}]}
    pol_list = [(p['x'], p['y']) for p in obj["polygon"]]
    pol = Polygon(pol_list)
    assert LabelBox_Annotations.labelbox_to_shapely(obj) == pol

    obj_nan = {"multiline": [{'x': 1, "y": 1}]}
    assert np.isnan(LabelBox_Annotations.labelbox_to_shapely(obj_nan))


def test_labelbox_object_to_shapely_point():
    obj = {"point": {"x": 10, "y": 10}}
    point = Point([10, 10])
    assert LabelBox_Annotations.labelbox_to_shapely(obj) == point
