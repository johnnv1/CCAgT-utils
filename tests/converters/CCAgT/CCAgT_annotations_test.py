from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Point

from CCAgT_utils.converters.CCAgT import CCAgT_Annotations


def test_init_class():
    df = pd.DataFrame([{'image_name': 'A', 'geometry': Point(1, 1), 'category_id': 1}])
    ccagt_ann = CCAgT_Annotations(df)

    assert ccagt_ann.df.equals(df)


def test_init_wrong_columns():
    with pytest.raises(KeyError):
        df = pd.DataFrame([{'image_name': 'A', 'category_id': 1}])
        CCAgT_Annotations(df)

    with pytest.raises(KeyError):
        df = pd.DataFrame([{'category_id': 1}])
        CCAgT_Annotations(df)

    with pytest.raises(KeyError):
        df = pd.DataFrame([{'image_name': 'A', 'geometry': Point(1, 1)}])
        CCAgT_Annotations(df)


def test_init_wrong_data():
    with pytest.raises(TypeError):
        CCAgT_Annotations([{'image_name': 'A', 'geometry': Point(1, 1)}])


def test_get_slide_id():
    df = pd.DataFrame([{'image_name': 'A_xxx', 'geometry': Point(1, 1), 'category_id': 1}])
    ccagt_ann = CCAgT_Annotations(df)

    assert ccagt_ann.get_slide_id().to_numpy().tolist() == ['A']

# TODO: test for CCAgT_Annotations.satellite_point_to_polygon
# TODO: test for CCAgT_Annotations.fit_geometries_to_image_boundary
# TODO: test for CCAgT_Annotations.geometries_type
# TODO: test for CCAgT_Annotations.geometries_area
# TODO: test for CCAgT_Annotations.generate_ids
# TODO: test for CCAgT_Annotations.delete_by_area
# TODO: test for CCAgT_Annotations.find_intersecting_geometries
# TODO: test for CCAgT_Annotations.find_overlapping_annotations
# TODO: test for CCAgT_Annotations.union_geometries
