from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.converters import CCAgT
from CCAgT_utils.errors import FileTypeError
from CCAgT_utils.errors import MoreThanOneIDbyItemError
from CCAgT_utils.types.colors import Color
from testing import create


def test_init_class(ccagt_ann_multi, ccagt_df_multi):
    assert ccagt_ann_multi.df.equals(ccagt_df_multi)


def test_init_wrong_columns():
    with pytest.raises(KeyError):
        df = pd.DataFrame([{'image_name': 'A', 'category_id': 1}])
        CCAgT.CCAgT(df)

    with pytest.raises(KeyError):
        df = pd.DataFrame([{'category_id': 1}])
        CCAgT.CCAgT(df)

    with pytest.raises(KeyError):
        df = pd.DataFrame([{'image_name': 'A', 'geometry': Point(1, 1)}])
        CCAgT.CCAgT(df)


def test_init_wrong_data():
    with pytest.raises(TypeError):
        CCAgT.CCAgT([{'image_name': 'A', 'geometry': Point(1, 1)}])


def test_get_slide_id(ccagt_ann_single_nucleus):
    assert ccagt_ann_single_nucleus.get_slide_id().tolist() == ['C']


def test_geometries_type(ccagt_ann_multi, ccagt_df_multi):
    geometries_type = ccagt_df_multi['geometry'].apply(lambda g: g.geom_type).tolist()
    assert ccagt_ann_multi.geometries_type().tolist() == geometries_type


@pytest.mark.parametrize(
    'area_size,resolution,tolerance', [
        (70, 4, 0.3),
        (120, 4, 0.3),
        (97, 4, 1),
    ],
)
def test_satellite_point_to_polygon(ccagt_ann_multi, satellite_ex, area_size, resolution, tolerance):
    df = ccagt_ann_multi.df
    df['geo_type'] = ccagt_ann_multi.geometries_type()
    sat_series = df.loc[(df['category_id'] == 3) & (df['geo_type'] == 'Point'), 'geometry']

    area_size = 70
    resolution = 4
    tolerance = 0.3
    sat_series_pol = ccagt_ann_multi.satellite_point_to_polygon(
        sat_series,
        area_size=area_size,
        resolution=resolution,
        tolerance=tolerance,
    )

    diameter = np.sqrt(area_size / np.pi)
    pol_sat = satellite_ex.buffer(diameter, resolution).simplify(tolerance)
    assert sat_series_pol.to_numpy()[0].equals(pol_sat)


def test_fit_geometries_to_image_boundary(ccagt_ann_single_nucleus, nucleus_ex):
    geos = ccagt_ann_single_nucleus.fit_geometries_to_image_boundary(100, 100).tolist()

    assert geos[0].equals(nucleus_ex)

    geos = ccagt_ann_single_nucleus.fit_geometries_to_image_boundary(13, 13).tolist()
    p = Polygon([(0, 0), (0, 13), (13, 13), (13, 0)])
    assert geos[0].equals(p)

    ccagt_aan = copy.copy(ccagt_ann_single_nucleus)
    ccagt_aan.df['geometry'] = Polygon([(100, 100), (100, 130), (130, 130), (130, 100)])
    geos = ccagt_ann_single_nucleus.fit_geometries_to_image_boundary(13, 13).tolist()
    assert np.isnan(geos[0])


def test_geometries_area(ccagt_ann_single_nucleus, nucleus_ex):
    assert ccagt_ann_single_nucleus.geometries_area().tolist() == [nucleus_ex.area]


def test_generate_ids(ccagt_ann_multi):
    img_name_series = ccagt_ann_multi.df['image_name']
    assert ccagt_ann_multi.generate_ids(img_name_series).tolist() == [1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize(
    'min_area,expected,compute_area', [
        (99999, 0, False),
        (0, 1, False),
        (0, 1, True),
    ],
)
def test_delete_by_area(ccagt_ann_single_nucleus, min_area, expected, compute_area):
    if compute_area:
        ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()

    categories_infos = CategoriesInfos([{'name': 'Nucleus', 'id': 1, 'color': (0, 0, 0), 'minimal_area': min_area}])
    df = ccagt_ann_single_nucleus.delete_by_area(categories_infos)
    assert df.shape[0] == expected


def test_delete_by_area_ignore_ids(ccagt_ann_multi):
    categories_infos = CategoriesInfos([
        {'name': 'Nucleus', 'id': 1, 'color': (0, 0, 0), 'minimal_area': 1000},
        {'name': 'Cluster', 'id': 2, 'color': (0, 0, 0), 'minimal_area': 0},
        {'name': 'Satellite', 'id': 3, 'color': (0, 0, 0), 'minimal_area': 0},
        {'name': 'Nucleus_out_of_focus', 'id': 4, 'color': (0, 0, 0), 'minimal_area': 0},
    ])
    df = ccagt_ann_multi.delete_by_area(categories_infos, set({2, 3}))
    assert df.shape[0] == 6


def test_find_intersecting_geometries(ccagt_ann_multi, cluster_ex):
    g = ccagt_ann_multi.df.groupby('image_name')
    df = g.get_group(list(g.groups)[0])
    indexes = CCAgT.CCAgT.find_intersecting_geometries(cluster_ex, 2, df)

    assert indexes == [1]

    indexes = CCAgT.CCAgT.find_intersecting_geometries(cluster_ex, 2, pd.DataFrame())
    assert np.isnan(indexes)


def test_filter_by_category(ccagt_ann_multi):
    df = ccagt_ann_multi.filter_by_category(set({2}))
    assert df['category_id'].unique() == [2]


def test_has_intersecting_geometries(ccagt_ann_multi, cluster_ex):
    geometries = ccagt_ann_multi.df['geometry']
    assert CCAgT.CCAgT.has_intersecting_geometries(cluster_ex, geometries)
    o = CCAgT.CCAgT.has_intersecting_geometries(Point(9999, 9999), geometries)
    assert not o


def test_verify_if_intersects(ccagt_ann_multi):
    df = ccagt_ann_multi.verify_if_intersects(set({1}), set({2}))
    assert df['has_intersecting'].tolist() == [True, True, True, False, False]

    df = ccagt_ann_multi.verify_if_intersects(set({1}), None)
    assert df['has_intersecting'].tolist() == [True] * 5


def test_find_overlapping_annotations(ccagt_ann_multi):
    groups = ccagt_ann_multi.find_overlapping_annotations(set({1, 2}))
    assert groups == {'A_xx1': [{1, 2}], 'A_yy2': [{6, 7}, {9, 10}]}

    groups = ccagt_ann_multi.find_overlapping_annotations(set({1, 2}), True)
    assert groups == {'A_xx1': [{1, 2}], 'A_yy2': [{6, 7}, {9, 10}]}

    groups = ccagt_ann_multi.find_overlapping_annotations(set({3}), True)
    assert groups == {}


def test_union_geometries(ccagt_ann_multi):
    groups = {'A_xx1': [{1, 2}]}

    t1 = copy.copy(ccagt_ann_multi)
    df = t1.union_geometries(groups, out_category_id=6)
    assert 6 in df['category_id'].tolist()

    t2 = copy.copy(ccagt_ann_multi)
    with pytest.raises(ValueError):
        df = t2.union_geometries(groups)

    t2 = copy.copy(ccagt_ann_multi)
    df = t2.union_geometries({'B_xx1': [{3, 4}]})
    assert df[df['image_name'] == 'B_xx1'].shape[0] == 1

    t3 = copy.copy(ccagt_ann_multi)
    df = t3.union_geometries({'B_xx1': [{-99}]})
    assert df.shape == ccagt_ann_multi.df.shape


def test_to_OD_COCO(ccagt_ann_single_nucleus, coco_OD_ann_single_nucleus):
    with pytest.raises(KeyError):
        ccagt_ann_single_nucleus.to_OD_COCO()

    ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
    ccagt_ann_single_nucleus.df['iscrowd'] = 0

    coco_OD_ann = ccagt_ann_single_nucleus.to_OD_COCO()

    assert coco_OD_ann == coco_OD_ann_single_nucleus


def test_image_id_by_name(ccagt_ann_single_nucleus):
    with pytest.raises(KeyError):
        id = ccagt_ann_single_nucleus.image_id_by_name('C_xx1')

    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
    id = ccagt_ann_single_nucleus.image_id_by_name('C_xx1')
    assert id == 1

    ccagt_ann_single_nucleus.df = pd.concat([ccagt_ann_single_nucleus.df, ccagt_ann_single_nucleus.df]).reset_index()
    ccagt_ann_single_nucleus.df.loc[1, 'image_id'] = 50
    with pytest.raises(MoreThanOneIDbyItemError):
        ccagt_ann_single_nucleus.image_id_by_name('C_xx1')


def test_read_and_dump_to_parquet(ccagt_ann_single_nucleus, tmpdir):
    filename = tmpdir.join('ccagt_test.parquet')

    ccagt_ann_single_nucleus.to_parquet(filename)

    assert filename.check()

    ccagt_ann = CCAgT.read_parquet(str(filename))
    assert ccagt_ann.df.equals(ccagt_ann_single_nucleus.df)


def test_single_core_to_OD_COCO(ccagt_ann_single_nucleus, coco_OD_ann_single_nucleus):
    ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
    ccagt_ann_single_nucleus.df['iscrowd'] = 0
    ccagt_ann_single_nucleus.df.index = ccagt_ann_single_nucleus.df.index + 1

    coco_OD_ann = CCAgT.single_core_to_OD_COCO(ccagt_ann_single_nucleus.df)

    assert coco_OD_ann == coco_OD_ann_single_nucleus


def test_single_core_to_mask(nucleus_ex, tmpdir):
    img_name = 'C_xx1'
    df = pd.DataFrame([create.row_CCAgT(nucleus_ex, 1, img_name)])
    df['image_id'] = [1]
    df['slide_id'] = ['C']
    df['image_width'] = 1600
    df['image_height'] = 1200

    outdir = os.path.join(tmpdir, 'C/')
    os.makedirs(outdir)
    CCAgT.single_core_to_mask(df, tmpdir, split_by_slide=True)
    assert os.path.isfile(os.path.join(outdir, img_name + '.png'))

    CCAgT.single_core_to_mask(df, tmpdir, split_by_slide=False)
    assert os.path.isfile(os.path.join(tmpdir, img_name + '.png'))


@pytest.mark.slow
def test_generate_masks(ccagt_ann_single_nucleus, tmpdir):

    ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'
    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])

    ccagt_ann_single_nucleus.generate_masks(tmpdir, split_by_slide=True)
    assert os.path.isfile(tmpdir.join('C/C_xx1.png'))

    ccagt_ann_single_nucleus.generate_masks(tmpdir, split_by_slide=False)
    assert os.path.isfile(tmpdir.join('C_xx1.png'))

    ccagt_ann_single_nucleus.df['slide_id'] = ccagt_ann_single_nucleus.get_slide_id()
    ccagt_ann_single_nucleus.generate_masks(tmpdir, split_by_slide=True)
    assert os.path.isfile(tmpdir.join('C/C_xx1.png'))


def test_single_core_to_PS_COCO(ccagt_ann_single_nucleus, tmpdir):
    ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'
    ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
    ccagt_ann_single_nucleus.df['iscrowd'] = 0
    ccagt_ann_single_nucleus.df['color'] = Color(21, 62, 125)

    out = CCAgT.single_core_to_PS_COCO(ccagt_ann_single_nucleus.df, tmpdir, False)

    check1 = all(y in out[0] for y in {'image_id', 'file_name', 'segments_info'})
    assert check1
    check2 = all(y in x for x in out[0]['segments_info'] for y in {'id', 'category_id', 'bbox', 'iscrowd'})
    assert check2

    assert len(os.listdir(tmpdir)) > 0

    ccagt_ann_single_nucleus.df['slide_id'] = ccagt_ann_single_nucleus.get_slide_id()
    tmpdir.mkdir('C')
    CCAgT.single_core_to_PS_COCO(ccagt_ann_single_nucleus.df, tmpdir, True)
    assert len(os.listdir(os.path.join(tmpdir, 'C/'))) > 0


def test_single_core_to_PS_COCO_multisizes(ccagt_ann_multi, tmpdir):
    ccagt_ann_multi.df['area'] = ccagt_ann_multi.geometries_area()
    ccagt_ann_multi.df['image_id'] = ccagt_ann_multi.generate_ids(ccagt_ann_multi.df['image_name'])
    ccagt_ann_multi.df['iscrowd'] = 0
    ccagt_ann_multi.df['color'] = Color(21, 62, 125)

    ccagt_ann_multi.df['image_width'] = 1000
    ccagt_ann_multi.df['image_height'] = 1000

    ccagt_ann_multi.df.loc[ccagt_ann_multi.df['image_id'] == 1, 'image_height'] = 2000

    ccagt_ann_multi.df['geo_type'] = ccagt_ann_multi.df['geometry'].apply(lambda g: g.geom_type).tolist()
    ccagt_ann_multi.df = ccagt_ann_multi.df[ccagt_ann_multi.df['geo_type'] != 'Point']
    CCAgT.single_core_to_PS_COCO(ccagt_ann_multi.df, tmpdir, False)

    assert len(os.listdir(tmpdir)) > 0


def test_CCAgT_to_PS_COCO(ccagt_ann_single_nucleus, categories_infos, tmpdir):
    ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'
    ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
    ccagt_ann_single_nucleus.df['iscrowd'] = 0

    out = ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir, False)

    check1 = all(y in out[0] for y in {'image_id', 'file_name', 'segments_info'})
    assert check1
    check2 = all(y in x for x in out[0]['segments_info'] for y in {'id', 'category_id', 'bbox', 'iscrowd'})
    assert check2

    assert len(os.listdir(tmpdir)) > 0

    ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir, True)
    ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir, True)
    assert len(os.listdir(os.path.join(tmpdir, 'C/'))) > 0


def test_CCAgT_to_PS_COCO_without_cols(ccagt_ann_single_nucleus, categories_infos, tmpdir):
    with pytest.raises(KeyError):
        ccagt_ann_single_nucleus.to_PS_COCO(categories_infos, tmpdir)


def test_read_parquet_wrong_type():
    with pytest.raises(FileTypeError):
        CCAgT.read_parquet('wrong file.and.type')
