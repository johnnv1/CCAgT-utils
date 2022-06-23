from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest
from distutils.command.sdist import sdist
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.colors import Color
from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.base.errors import MoreThanOneIDbyItemError
from CCAgT_utils.formats import CCAgT
from testing import create


def test_slides_ids(ccagt_df_single_nucleus):
    assert CCAgT.slides_ids(ccagt_df_single_nucleus).tolist() == ['C']


def test_geometries_type(ccagt_df_multi):
    geometries_type = ccagt_df_multi['geometry'].apply(lambda g: g.geom_type).tolist()
    assert CCAgT.geometries_type(ccagt_df_multi).tolist() == geometries_type


def test_geometries_area(ccagt_df_single_nucleus, nucleus_ex):
    assert CCAgT.geometries_area(ccagt_df_single_nucleus).tolist() == [nucleus_ex.area]


def test_fit_geometries_to_boundary(ccagt_df_single_nucleus, nucleus_ex):
    geos = CCAgT.fit_geometries_to_boundary(ccagt_df_single_nucleus, 100, 100).tolist()

    assert geos[0].equals(nucleus_ex)

    geos = CCAgT.fit_geometries_to_boundary(ccagt_df_single_nucleus, 13, 13).tolist()
    p = Polygon([(0, 0), (0, 13), (13, 13), (13, 0)])
    assert geos[0].equals(p)

    df = copy.copy(ccagt_df_single_nucleus)
    df['geometry'] = Polygon([(100, 100), (100, 130), (130, 130), (130, 100)])
    geos = CCAgT.fit_geometries_to_boundary(df, 13, 13).tolist()
    assert np.isnan(geos[0])


def test_generate_ids(ccagt_df_multi):
    img_name_series = ccagt_df_multi['image_name']
    assert CCAgT.generate_ids(img_name_series).tolist() == [1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize(
    'min_area,expected,compute_area', [
        (99999, 0, False),
        (0, 1, False),
        (0, 1, True),
    ],
)
def test_delete_by_area(ccagt_df_single_nucleus, min_area, expected, compute_area):
    df = ccagt_df_single_nucleus
    if compute_area:
        df['area'] = CCAgT.geometries_area(df)

    categories_infos = CategoriesInfos([{'name': 'Nucleus', 'id': 1, 'color': (0, 0, 0), 'minimal_area': min_area}])
    df = CCAgT.delete_by_area(df, categories_infos)
    assert df.shape[0] == expected


def test_delete_by_area_ignore_ids(ccagt_df_multi):
    categories_infos = CategoriesInfos([
        {'name': 'Nucleus', 'id': 1, 'color': (0, 0, 0), 'minimal_area': 1000},
        {'name': 'Cluster', 'id': 2, 'color': (0, 0, 0), 'minimal_area': 0},
        {'name': 'Satellite', 'id': 3, 'color': (0, 0, 0), 'minimal_area': 0},
        {'name': 'Nucleus_out_of_focus', 'id': 4, 'color': (0, 0, 0), 'minimal_area': 0},
    ])
    df = CCAgT.delete_by_area(ccagt_df_multi, categories_infos, set({2, 3}))
    assert df.shape[0] == 6


def test_filter_by_categories(ccagt_df_multi):
    df = CCAgT.filter_by_categories(ccagt_df_multi, set({2}))
    assert df['category_id'].unique() == [2]


def test_union_geometries(ccagt_df_multi):
    groups = {'A_xx1': [{1, 2}]}

    df = CCAgT.union_geometries(ccagt_df_multi, groups, out_category_id=6)
    assert 6 in df['category_id'].tolist()

    with pytest.raises(ValueError):
        df = CCAgT.union_geometries(ccagt_df_multi, groups)

    df = CCAgT.union_geometries(ccagt_df_multi, {'B_xx1': [{3, 4}]})
    assert df[df['image_name'] == 'B_xx1'].shape[0] == 1

    df = CCAgT.union_geometries(ccagt_df_multi, {'B_xx1': [{-99}]})
    assert df.shape == ccagt_df_multi.shape


def test_find_intersecting_geometries(ccagt_df_multi, cluster_ex):
    g = ccagt_df_multi.groupby('image_name')
    df = g.get_group(list(g.groups)[0])
    indexes = CCAgT.find_intersecting_geometries(df, cluster_ex, 2)

    assert indexes == [1]

    indexes = CCAgT.find_intersecting_geometries(pd.DataFrame(), cluster_ex, 2)
    assert np.isnan(indexes)


def test_has_intersecting_geometries(ccagt_df_multi, cluster_ex):
    geometries = ccagt_df_multi['geometry']
    assert CCAgT.has_intersecting_geometries(cluster_ex, geometries)
    o = CCAgT.has_intersecting_geometries(Point(9999, 9999), geometries)
    assert not o


def test_find_overlapping_annotations(ccagt_df_multi):
    groups = CCAgT.find_overlapping_annotations(ccagt_df_multi, set({1, 2}))
    assert groups == {'A_xx1': [{1, 2}], 'A_yy2': [{6, 7}, {9, 10}]}

    groups = CCAgT.find_overlapping_annotations(ccagt_df_multi, set({1, 2}), True)
    assert groups == {'A_xx1': [{1, 2}], 'A_yy2': [{6, 7}, {9, 10}]}

    groups = CCAgT.find_overlapping_annotations(ccagt_df_multi, set({3}), True)
    assert groups == {}


def test_image_id_by_name(ccagt_df_single_nucleus):
    with pytest.raises(KeyError):
        id = CCAgT.image_id_by_name(ccagt_df_single_nucleus, 'C_xx1')

    ccagt_df_single_nucleus['image_id'] = CCAgT.generate_ids(ccagt_df_single_nucleus['image_name'])
    id = CCAgT.image_id_by_name(ccagt_df_single_nucleus, 'C_xx1')
    assert id == 1

    ccagt_df_single_nucleus = pd.concat([ccagt_df_single_nucleus, ccagt_df_single_nucleus]).reset_index()
    ccagt_df_single_nucleus.loc[1, 'image_id'] = 50
    with pytest.raises(MoreThanOneIDbyItemError):
        CCAgT.image_id_by_name(ccagt_df_single_nucleus, 'C_xx1')


def test_verify_if_intersects(ccagt_df_multi):
    df = CCAgT.verify_if_intersects(ccagt_df_multi, set({1}), set({2}))
    assert df['has_intersecting'].tolist() == [True, True, True, False, False]

    df = CCAgT.verify_if_intersects(ccagt_df_multi, set({1}), None)
    assert df['has_intersecting'].tolist() == [True] * 5


# def test_union_geometries(ccagt_ann_multi):
#     groups = {'A_xx1': [{1, 2}]}

#     t1 = copy.copy(ccagt_ann_multi)
#     df = t1.union_geometries(groups, out_category_id=6)
#     assert 6 in df['category_id'].tolist()

#     t2 = copy.copy(ccagt_ann_multi)
#     with pytest.raises(ValueError):
#         df = t2.union_geometries(groups)

#     t2 = copy.copy(ccagt_ann_multi)
#     df = t2.union_geometries({'B_xx1': [{3, 4}]})
#     assert df[df['image_name'] == 'B_xx1'].shape[0] == 1

#     t3 = copy.copy(ccagt_ann_multi)
#     df = t3.union_geometries({'B_xx1': [{-99}]})
#     assert df.shape == ccagt_ann_multi.df.shape


# def test_to_OD_COCO(ccagt_ann_single_nucleus, coco_OD_ann_single_nucleus):
#     with pytest.raises(KeyError):
#         ccagt_ann_single_nucleus.to_OD_COCO()

#     ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
#     ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
#     ccagt_ann_single_nucleus.df['iscrowd'] = 0

#     coco_OD_ann = ccagt_ann_single_nucleus.to_OD_COCO()

#     assert coco_OD_ann == coco_OD_ann_single_nucleus


# def test_read_and_dump_to_parquet(ccagt_ann_single_nucleus, tmpdir):
#     filename = tmpdir.join('ccagt_test.parquet')

#     ccagt_ann_single_nucleus.to_parquet(filename)

#     assert filename.check()

#     ccagt_ann = CCAgT.read_parquet(str(filename))
#     assert ccagt_ann.df.equals(ccagt_ann_single_nucleus.df)


# def test_single_core_to_OD_COCO(ccagt_ann_single_nucleus, coco_OD_ann_single_nucleus):
#     ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
#     ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
#     ccagt_ann_single_nucleus.df['iscrowd'] = 0
#     ccagt_ann_single_nucleus.df.index = ccagt_ann_single_nucleus.df.index + 1

#     coco_OD_ann = CCAgT.single_core_to_OD_COCO(ccagt_ann_single_nucleus.df)

#     assert coco_OD_ann == coco_OD_ann_single_nucleus


# def test_single_core_to_mask(nucleus_ex, tmpdir):
#     img_name = 'C_xx1'
#     df = pd.DataFrame([create.row_CCAgT(nucleus_ex, 1, img_name)])
#     df['image_id'] = [1]
#     df['slide_id'] = ['C']
#     df['image_width'] = 1600
#     df['image_height'] = 1200

#     outdir = os.path.join(tmpdir, 'C/')
#     os.makedirs(outdir)
#     CCAgT.single_core_to_mask(df, tmpdir, split_by_slide=True)
#     assert os.path.isfile(os.path.join(outdir, img_name + '.png'))

#     CCAgT.single_core_to_mask(df, tmpdir, split_by_slide=False)
#     assert os.path.isfile(os.path.join(tmpdir, img_name + '.png'))


# @pytest.mark.slow
# def test_generate_masks(ccagt_ann_single_nucleus, tmpdir):

#     ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'
#     ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])

#     ccagt_ann_single_nucleus.generate_masks(tmpdir, split_by_slide=True)
#     assert os.path.isfile(tmpdir.join('C/C_xx1.png'))

#     ccagt_ann_single_nucleus.generate_masks(tmpdir, split_by_slide=False)
#     assert os.path.isfile(tmpdir.join('C_xx1.png'))

#     ccagt_ann_single_nucleus.df['slide_id'] = ccagt_ann_single_nucleus.get_slide_id()
#     ccagt_ann_single_nucleus.generate_masks(tmpdir, split_by_slide=True)
#     assert os.path.isfile(tmpdir.join('C/C_xx1.png'))


# def test_single_core_to_PS_COCO(ccagt_ann_single_nucleus, tmpdir):
#     ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'
#     ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
#     ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
#     ccagt_ann_single_nucleus.df['iscrowd'] = 0
#     ccagt_ann_single_nucleus.df['color'] = Color(21, 62, 125)

#     out = CCAgT.single_core_to_PS_COCO(ccagt_ann_single_nucleus.df, tmpdir, False)

#     check1 = all(y in out[0] for y in {'image_id', 'file_name', 'segments_info'})
#     assert check1
#     check2 = all(y in x for x in out[0]['segments_info'] for y in {'id', 'category_id', 'bbox', 'iscrowd'})
#     assert check2

#     assert len(os.listdir(tmpdir)) > 0

#     ccagt_ann_single_nucleus.df['slide_id'] = ccagt_ann_single_nucleus.get_slide_id()
#     tmpdir.mkdir('C')
#     CCAgT.single_core_to_PS_COCO(ccagt_ann_single_nucleus.df, tmpdir, True)
#     assert len(os.listdir(os.path.join(tmpdir, 'C/'))) > 0


# def test_single_core_to_PS_COCO_multisizes(ccagt_ann_multi, tmpdir):
#     ccagt_ann_multi.df['area'] = ccagt_ann_multi.geometries_area()
#     ccagt_ann_multi.df['image_id'] = ccagt_ann_multi.generate_ids(ccagt_ann_multi.df['image_name'])
#     ccagt_ann_multi.df['iscrowd'] = 0
#     ccagt_ann_multi.df['color'] = Color(21, 62, 125)

#     ccagt_ann_multi.df['image_width'] = 1000
#     ccagt_ann_multi.df['image_height'] = 1000

#     ccagt_ann_multi.df.loc[ccagt_ann_multi.df['image_id'] == 1, 'image_height'] = 2000

#     ccagt_ann_multi.df['geo_type'] = ccagt_ann_multi.df['geometry'].apply(lambda g: g.geom_type).tolist()
#     ccagt_ann_multi.df = ccagt_ann_multi.df[ccagt_ann_multi.df['geo_type'] != 'Point']
#     CCAgT.single_core_to_PS_COCO(ccagt_ann_multi.df, tmpdir, False)

#     assert len(os.listdir(tmpdir)) > 0


# def test_CCAgT_to_PS_COCO(ccagt_ann_single_nucleus, categories_infos, tmpdir):
#     ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'
#     ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
#     ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
#     ccagt_ann_single_nucleus.df['iscrowd'] = 0

#     out = ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir, False)

#     check1 = all(y in out[0] for y in {'image_id', 'file_name', 'segments_info'})
#     assert check1
#     check2 = all(y in x for x in out[0]['segments_info'] for y in {'id', 'category_id', 'bbox', 'iscrowd'})
#     assert check2

#     assert len(os.listdir(tmpdir)) > 0

#     ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir, True)
#     ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir, True)
#     assert len(os.listdir(os.path.join(tmpdir, 'C/'))) > 0


# def test_CCAgT_to_PS_COCO_without_cols(ccagt_ann_single_nucleus, categories_infos, tmpdir):
#     with pytest.raises(KeyError):
#         ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir)


# def test_read_parquet_wrong_type():
#     with pytest.raises(FileTypeError):
#         CCAgT.read_parquet('wrong file.and.type')
