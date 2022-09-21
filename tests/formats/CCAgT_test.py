from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.base.errors import MoreThanOneIDbyItemError
from CCAgT_utils.formats import ccagt


def test_load_wrong_type():
    with pytest.raises(FileTypeError):
        ccagt.load('wrong file.and.type')


def test_load_and_save(ccagt_df_single_nucleus, tmpdir):
    filename = tmpdir.join('ccagt_test.parquet')

    ccagt.save(ccagt_df_single_nucleus, filename)

    assert filename.check()

    ccagt_ann = ccagt.load(str(filename))
    assert ccagt_ann.equals(ccagt_df_single_nucleus)


def test_slides_ids(ccagt_df_single_nucleus):
    assert ccagt.slides_ids(ccagt_df_single_nucleus).tolist() == ['C']


def test_geometries_type(ccagt_df_multi):
    geometries_type = ccagt_df_multi['geometry'].apply(
        lambda g: g.geom_type,
    ).tolist()
    assert ccagt.geometries_type(
        ccagt_df_multi,
    ).tolist() == geometries_type


def test_geometries_area(ccagt_df_single_nucleus, nucleus_ex):
    assert ccagt.geometries_area(
        ccagt_df_single_nucleus,
    ).tolist() == [nucleus_ex.area]


def test_fit_geometries_to_boundary(ccagt_df_single_nucleus, nucleus_ex):
    geos = ccagt.fit_geometries_to_boundary(
        ccagt_df_single_nucleus, 100, 100,
    ).tolist()

    assert geos[0].equals(nucleus_ex)

    geos = ccagt.fit_geometries_to_boundary(
        ccagt_df_single_nucleus, 13, 13,
    ).tolist()
    p = Polygon([(0, 0), (0, 13), (13, 13), (13, 0)])
    assert geos[0].equals(p)

    df = copy.copy(ccagt_df_single_nucleus)
    df['geometry'] = Polygon([
        (100, 100),
        (100, 130),
        (130, 130),
        (130, 100),
    ])
    geos = ccagt.fit_geometries_to_boundary(df, 13, 13).tolist()
    assert np.isnan(geos[0])


def test_generate_ids(ccagt_df_multi):
    img_name_series = ccagt_df_multi['image_name']
    assert ccagt.generate_ids(
        img_name_series,
    ).tolist() == [1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize(
    'min_area,expected,compute_area', [
        (99999, 0, False),
        (0, 1, False),
        (0, 1, True),
    ],
)
def test_delete_by_area(
    ccagt_df_single_nucleus,
    min_area,
    expected,
    compute_area,
):
    df = ccagt_df_single_nucleus
    if compute_area:
        df['area'] = ccagt.geometries_area(df)

    categories_infos = CategoriesInfos(
        [{
            'name': 'Nucleus', 'id': 1, 'color': (
                0, 0, 0,
            ), 'minimal_area': min_area,
        }],
    )
    df = ccagt.delete_by_area(df, categories_infos)
    assert df.shape[0] == expected


def test_delete_by_area_ignore_ids(ccagt_df_multi):
    categories_infos = CategoriesInfos([
        {
            'name': 'Nucleus', 'id': 1,
            'color': (0, 0, 0),
            'minimal_area': 1000,
        },
        {
            'name': 'Cluster', 'id': 2,
            'color': (0, 0, 0),
            'minimal_area': 0,
        },
        {
            'name': 'Satellite', 'id': 3,
            'color': (0, 0, 0),
            'minimal_area': 0,
        },
        {
            'name': 'Nucleus_out_of_focus', 'id': 4,
            'color': (0, 0, 0), 'minimal_area': 0,
        },
    ])
    df = ccagt.delete_by_area(
        ccagt_df_multi,
        categories_infos,
        set({2, 3}),
    )
    assert df.shape[0] == 6


def test_filter_by_categories(ccagt_df_multi):
    df = ccagt.filter_by_categories(ccagt_df_multi, set({2}))
    assert df['category_id'].unique() == [2]


def test_union_geometries(ccagt_df_multi):
    groups = {'A_xx1': [{1, 2}]}

    df = ccagt.union_geometries(
        ccagt_df_multi,
        groups,
        out_category_id=6,
    )
    assert 6 in df['category_id'].tolist()

    with pytest.raises(ValueError):
        df = ccagt.union_geometries(ccagt_df_multi, groups)

    df = ccagt.union_geometries(ccagt_df_multi, {'B_xx1': [{3, 4}]})
    assert df[df['image_name'] == 'B_xx1'].shape[0] == 1

    df = ccagt.union_geometries(ccagt_df_multi, {'B_xx1': [{-99}]})
    assert df.shape == ccagt_df_multi.shape


def test_find_intersecting_geometries(ccagt_df_multi, cluster_ex):
    g = ccagt_df_multi.groupby('image_name')
    df = g.get_group(list(g.groups)[0])
    indexes = ccagt.find_intersecting_geometries(df, cluster_ex, 2)

    assert indexes == [1]

    indexes = ccagt.find_intersecting_geometries(
        pd.DataFrame(),
        cluster_ex,
        2,
    )
    assert np.isnan(indexes)


def test_has_intersecting_geometries(ccagt_df_multi, cluster_ex):
    geometries = ccagt_df_multi['geometry']
    assert ccagt.has_intersecting_geometries(cluster_ex, geometries)
    o = ccagt.has_intersecting_geometries(Point(9999, 9999), geometries)
    assert not o


def test_find_overlapping_annotations(ccagt_df_multi):
    groups = ccagt.find_overlapping_annotations(
        ccagt_df_multi, set({1, 2}),
    )
    assert groups == {'A_xx1': [{1, 2}], 'A_yy2': [{6, 7}, {9, 10}]}

    groups = ccagt.find_overlapping_annotations(
        ccagt_df_multi, set({1, 2}), True,
    )
    assert groups == {'A_xx1': [{1, 2}], 'A_yy2': [{6, 7}, {9, 10}]}

    groups = ccagt.find_overlapping_annotations(
        ccagt_df_multi, set({3}), True,
    )
    assert groups == {}


def test_image_id_by_name(ccagt_df_single_nucleus):
    with pytest.raises(KeyError):
        id = ccagt.image_id_by_name(ccagt_df_single_nucleus, 'C_xx1')

    ccagt_df_single_nucleus['image_id'] = ccagt.generate_ids(
        ccagt_df_single_nucleus['image_name'],
    )
    id = ccagt.image_id_by_name(ccagt_df_single_nucleus, 'C_xx1')
    assert id == 1

    ccagt_df_single_nucleus = pd.concat(
        [ccagt_df_single_nucleus, ccagt_df_single_nucleus],
    ).reset_index()
    ccagt_df_single_nucleus.loc[1, 'image_id'] = 50
    with pytest.raises(MoreThanOneIDbyItemError):
        ccagt.image_id_by_name(ccagt_df_single_nucleus, 'C_xx1')


def test_verify_if_intersects(ccagt_df_multi):
    df = ccagt.verify_if_intersects(ccagt_df_multi, set({1}), set({2}))

    expected = [True, True, True, False, False]

    assert df['has_intersecting'].tolist() == expected

    df = ccagt.verify_if_intersects(ccagt_df_multi, set({1}), None)
    assert df['has_intersecting'].tolist() == [True] * 5
