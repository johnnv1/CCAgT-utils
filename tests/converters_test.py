from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.converters import annotations_to_mask
from CCAgT_utils.converters import bounds_to_coco_bb
from CCAgT_utils.converters import from_labelbox
from CCAgT_utils.converters import lbox_geo_to_shapely
from CCAgT_utils.converters import order_annotations_to_draw
from CCAgT_utils.converters import pol_to_coco_segment
from CCAgT_utils.formats.annotation import Annotation


def test_lbox_geo_to_shapely(lbox_raw_single_satellite, satellite_ex):
    geometry = lbox_geo_to_shapely(lbox_raw_single_satellite['Label']['objects'][0])
    assert satellite_ex.equals(geometry)


def test_from_labelbox(lbox_raw_sample_complete, categories_infos, lbox_raw_expected_ccagt_df):
    lbox_df_raw = pd.DataFrame(lbox_raw_sample_complete)
    df = from_labelbox(lbox_df_raw, categories_infos)
    assert df.equals(lbox_raw_expected_ccagt_df)


def test_from_labelbox_with_duplicated_image(
    lbox_raw_sample_complete,
    categories_infos,
    lbox_raw_single_satellite,
    lbox_raw_expected_ccagt_df,
):
    sample = lbox_raw_single_satellite.copy()
    sample.update(ID='otherID-99x')
    lbox_raw_sample_complete.append(sample)

    lbox_df_raw = pd.DataFrame(lbox_raw_sample_complete)
    df = from_labelbox(lbox_df_raw, categories_infos)
    assert df.equals(lbox_raw_expected_ccagt_df)

    sample = lbox_raw_single_satellite.copy()
    sample.update(ID='otherID-99x', Reviews=[{'score': 0, 'labelId': 'otherID-99x'}])
    lbox_raw_sample_complete.append(sample)
    lbox_df_raw = pd.DataFrame(lbox_raw_sample_complete)
    df = from_labelbox(lbox_df_raw, categories_infos)
    assert df.equals(lbox_raw_expected_ccagt_df)

    sample = lbox_raw_single_satellite.copy()
    del sample['Label']['objects'][0]['point']
    lbox_raw_sample_complete = [sample]
    lbox_df_raw = pd.DataFrame(lbox_raw_sample_complete)
    with pytest.raises(ValueError):
        from_labelbox(lbox_df_raw, categories_infos)


def test_order_annotations_to_draw(cluster_ex, satellite_ex, nucleus_ex):
    anns = [Annotation(cluster_ex, 2), Annotation(satellite_ex, 3), Annotation(nucleus_ex, 1)]
    anns = order_annotations_to_draw(anns)

    categories_ids = [ann.category_id for ann in anns]
    assert categories_ids == [1, 2, 3]


def test_annotations_to_mask(cluster_ex, cluster_mask_ex):
    shape = cluster_mask_ex.shape
    msk = annotations_to_mask([Annotation(cluster_ex, 2)], shape[0], shape[1])
    assert np.array_equal(cluster_mask_ex, msk.categorical)


def test_bounds_to_coco_bb():
    p = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    bounds = p.bounds
    coco_bb = bounds_to_coco_bb(bounds, decimals=2)

    assert coco_bb == [0., 0., 10, 10]


def test_pol_to_coco_segment():
    coords_1 = [(0, 0), (0, 10), (10, 10), (10, 0)]
    p = Polygon(coords_1)
    seg = pol_to_coco_segment(p)

    coords_1_out = [0., 0., 0., 10., 10., 10., 10., 0., 0., 0.]
    assert seg == [coords_1_out]

    coords_2 = [(20, 20), (20, 210), (210, 210), (210, 20)]

    mp = MultiPolygon([p, Polygon(coords_2)])
    seg = pol_to_coco_segment(mp)

    coords_2_out = [20., 20., 20., 210., 210., 210., 210., 20., 20., 20.]
    assert seg == [coords_1_out, coords_2_out]


def test_wrong_pol_to_coco_segment():
    with pytest.raises(TypeError):
        pol_to_coco_segment(Point(0, 0))
