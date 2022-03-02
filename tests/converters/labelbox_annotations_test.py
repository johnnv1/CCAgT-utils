from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.converters.LabelBox import LabelBox_Annotations


def test_labelbox_to_dataFrame(lbb_ann, lbb_raw_sample_complete):
    raw_df = pd.DataFrame(lbb_raw_sample_complete)
    assert lbb_ann.raw_dataframe.equals(raw_df)
    assert lbb_ann.raw_dataframe.shape[0] == len(lbb_raw_sample_complete)


def test_init_without_raw():
    with pytest.raises(ValueError):
        LabelBox_Annotations(raw_labelbox=None)


def test_init_without_categories_map(lbb_raw_sample_complete):
    LabelBox_Annotations(lbb_raw_sample_complete)


def test_init_without_expected_data():
    with pytest.raises(KeyError):
        LabelBox_Annotations(raw_labelbox=[{'ID': 'a2', 'External ID': 'tmp/A_xxx'}])


def test_instance_categories_map(lbb_ann, categories_aux_data):
    assert lbb_ann.categories_map == categories_aux_data


def test_labelbox_object_to_shapely():
    obj = {'polygon': [{'x': 10, 'y': 10}, {'x': 20, 'y': 20}, {'x': 25, 'y': 15}, {'x': 15, 'y': 10}]}
    pol_list = [(p['x'], p['y']) for p in obj['polygon']]
    pol = Polygon(pol_list)
    assert LabelBox_Annotations.labelbox_to_shapely(obj) == pol

    obj_nan = {'multiline': [{'x': 1, 'y': 1}]}
    assert np.isnan(LabelBox_Annotations.labelbox_to_shapely(obj_nan))


def test_labelbox_object_to_shapely_point():
    obj = {'point': {'x': 10, 'y': 10}}
    point = Point([10, 10])
    assert LabelBox_Annotations.labelbox_to_shapely(obj) == point


def test_to_CCAgT(lbb_ann, lbb_raw_expected_ccagt_df):
    ccagt_ann = lbb_ann.to_CCAgT()
    assert ccagt_ann.df.equals(lbb_raw_expected_ccagt_df)


def test_to_CCAgT_check_categories_maps(lbb_ann, categories_aux_data, lbb_raw_expected_ccagt_df):
    ccagt_ann = lbb_ann.to_CCAgT(categories_aux_data)
    assert ccagt_ann.df.equals(lbb_raw_expected_ccagt_df)

    ccagt_ann = lbb_ann.to_CCAgT({})
    assert ccagt_ann.df.equals(lbb_raw_expected_ccagt_df)

    lbb_ann.categories_map = None
    ccagt_ann = lbb_ann.to_CCAgT(categories_aux_data)
    assert ccagt_ann.df.equals(lbb_raw_expected_ccagt_df)

    lbb_ann.categories_map = None
    with pytest.raises(ValueError):
        lbb_ann.to_CCAgT(None)

    lbb_ann.categories_map = None
    with pytest.raises(ValueError):
        lbb_ann.to_CCAgT('a')


def test_to_CCAgT_with_duplicated_image(lbb_ann,
                                        categories_aux_data,
                                        lbb_raw_single_satellite,
                                        lbb_raw_expected_ccagt_df):
    sample = lbb_raw_single_satellite.copy()
    sample.update(ID='otherID-99x')
    lbb_ann.raw_labelbox.append(sample)
    ccagt_ann = lbb_ann.to_CCAgT(categories_aux_data)
    assert ccagt_ann.df.equals(lbb_raw_expected_ccagt_df)

    sample = lbb_raw_single_satellite.copy()
    sample.update(ID='otherID-99x', Reviews=[{'score': 0, 'labelId': 'otherID-99x'}])
    lbb_ann.raw_labelbox.append(sample)
    ccagt_ann = lbb_ann.to_CCAgT(categories_aux_data)
    assert ccagt_ann.df.equals(lbb_raw_expected_ccagt_df)

    sample = lbb_raw_single_satellite.copy()
    del sample['Label']['objects'][0]['point']
    lbb_ann.raw_labelbox = [sample]
    with pytest.raises(ValueError):
        lbb_ann.to_CCAgT(categories_aux_data)
