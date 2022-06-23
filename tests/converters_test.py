from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CCAgT_utils.converters import annotations_to_mask
from CCAgT_utils.converters import from_labelbox
from CCAgT_utils.converters import lbox_geo_to_shapely
from CCAgT_utils.converters import order_annotations_to_draw
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
