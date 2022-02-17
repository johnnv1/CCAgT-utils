from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.converters.CCAgT import CCAgT_Annotations


def test_init_class():
    df = pd.DataFrame([{'image_name': 'A', 'geometry': Point(1, 1), 'category_id': 3}])
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
    df = pd.DataFrame([{'image_name': 'A_xxx', 'geometry': Point(1, 1), 'category_id': 3}])
    ccagt_ann = CCAgT_Annotations(df)

    assert ccagt_ann.get_slide_id().to_numpy().tolist() == ['A']


def test_geometries_type():
    df = pd.DataFrame({'image_name': ['A_xxx', 'B_yyy'],
                       'geometry': [Point(1, 1), Polygon([(0, 0), (2, 2), (2, 0), (1, 0)])],
                       'category_id': [3, 2]})

    ccagt_ann = CCAgT_Annotations(df)

    assert ccagt_ann.geometries_type().to_numpy().tolist() == ['Point', 'Polygon']


def test_satellite_point_to_polygon():
    df = pd.DataFrame([{'image_name': 'A_xxx', 'geometry': Point(1, 1), 'category_id': 3}])
    ccagt_ann = CCAgT_Annotations(df)
    df['geo_type'] = ccagt_ann.geometries_type()
    sat_series = df.loc[(df['category_id'] == 3) & (df['geo_type'] == 'Point'), 'geometry']
    area_size = 70
    resolution = 4
    tolerance = 0.3
    sat_series_pol = ccagt_ann.satellite_point_to_polygon(sat_series,
                                                          area_size=area_size,
                                                          resolution=resolution,
                                                          tolerance=tolerance)

    diameter = np.sqrt(area_size / np.pi)
    assert sat_series_pol.to_numpy()[0].equals(Point(1, 1).buffer(diameter, resolution).simplify(tolerance))

# TODO: test for CCAgT_Annotations.fit_geometries_to_image_boundary
# TODO: test for CCAgT_Annotations.geometries_area
# TODO: test for CCAgT_Annotations.generate_ids
# TODO: test for CCAgT_Annotations.delete_by_area
# TODO: test for CCAgT_Annotations.find_intersecting_geometries
# TODO: test for CCAgT_Annotations.find_overlapping_annotations
# TODO: test for CCAgT_Annotations.union_geometries
