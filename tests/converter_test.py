from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.base.colors import Color
from CCAgT_utils.converter import annotations_to_mask
from CCAgT_utils.converter import bounds_to_coco_bb
from CCAgT_utils.converter import from_labelbox
from CCAgT_utils.converter import lbox_geo_to_shapely
from CCAgT_utils.converter import order_annotations_to_draw
from CCAgT_utils.converter import pol_to_coco_segment
from CCAgT_utils.converter import single_core_to_mask
from CCAgT_utils.converter import single_core_to_OD_COCO
from CCAgT_utils.converter import single_core_to_PS_COCO
from CCAgT_utils.converter import to_mask
from CCAgT_utils.converter import to_OD_COCO
from CCAgT_utils.converter import to_PS_COCO
from CCAgT_utils.formats import CCAgT
from CCAgT_utils.formats.annotation import Annotation
from testing import create


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


def test_single_core_to_mask(nucleus_ex, tmpdir):
    img_name = 'C_xx1'
    df = pd.DataFrame([create.row_CCAgT(nucleus_ex, 1, img_name)])
    df['image_id'] = [1]
    df['slide_id'] = ['C']
    df['image_width'] = 1600
    df['image_height'] = 1200

    outdir = tmpdir.join('C/').mkdir()
    single_core_to_mask(df, tmpdir, split_by_slide=True)
    assert outdir.join(img_name + '.png').check()

    single_core_to_mask(df, tmpdir, split_by_slide=False)
    assert tmpdir.join(img_name + '.png').check()


@pytest.mark.slow
def test_to_mask(ccagt_df_single_nucleus, tmpdir):
    ccagt_df_single_nucleus['image_name'] = 'C_xx1'
    ccagt_df_single_nucleus['image_width'] = 1600
    ccagt_df_single_nucleus['image_height'] = 1200
    ccagt_df_single_nucleus['image_id'] = CCAgT.generate_ids(ccagt_df_single_nucleus['image_name'])
    to_mask(ccagt_df_single_nucleus, str(tmpdir), split_by_slide=True)
    assert tmpdir.join('C/C_xx1.png').check()

    to_mask(ccagt_df_single_nucleus, str(tmpdir), split_by_slide=False)
    assert tmpdir.join('C_xx1.png').check()

    ccagt_df_single_nucleus['slide_id'] = CCAgT.slides_ids(ccagt_df_single_nucleus)
    to_mask(ccagt_df_single_nucleus, str(tmpdir), split_by_slide=True)
    assert tmpdir.join('C/C_xx1.png').check()


def test_to_mask_without_data(capsys):
    to_mask(pd.DataFrame(columns=['image_id', 'image_name']), '')
    _, err = capsys.readouterr()
    assert 'Do not have annotations to generate the masks!\n' == err


def test_single_core_to_OD_COCO(ccagt_df_single_nucleus, coco_OD_ann_single_nucleus):
    ccagt_df_single_nucleus['area'] = CCAgT.geometries_area(ccagt_df_single_nucleus)
    ccagt_df_single_nucleus['image_id'] = CCAgT.generate_ids(ccagt_df_single_nucleus['image_name'])
    ccagt_df_single_nucleus['iscrowd'] = 0
    ccagt_df_single_nucleus.index = ccagt_df_single_nucleus.index + 1

    coco_OD_ann = single_core_to_OD_COCO(ccagt_df_single_nucleus)

    assert coco_OD_ann == coco_OD_ann_single_nucleus


def test_single_core_to_PS_COCO(ccagt_df_single_nucleus, tmpdir):
    ccagt_df_single_nucleus['image_name'] = 'C_xx1'
    ccagt_df_single_nucleus['image_width'] = 1600
    ccagt_df_single_nucleus['image_height'] = 1200
    ccagt_df_single_nucleus['area'] = CCAgT.geometries_area(ccagt_df_single_nucleus)
    ccagt_df_single_nucleus['image_id'] = CCAgT.generate_ids(ccagt_df_single_nucleus['image_name'])
    ccagt_df_single_nucleus['iscrowd'] = 0
    ccagt_df_single_nucleus['color'] = Color(21, 62, 125)

    out = single_core_to_PS_COCO(ccagt_df_single_nucleus, str(tmpdir), False)

    check1 = all(y in out[0] for y in {'image_id', 'file_name', 'segments_info'})
    assert check1
    check2 = all(y in x for x in out[0]['segments_info'] for y in {'id', 'category_id', 'bbox', 'iscrowd'})
    assert check2

    assert len(tmpdir.listdir()) > 0

    ccagt_df_single_nucleus['slide_id'] = CCAgT.slides_ids(ccagt_df_single_nucleus)
    subdir = tmpdir.mkdir('C')
    single_core_to_PS_COCO(ccagt_df_single_nucleus, str(tmpdir), True)
    assert len(subdir.listdir()) > 0


def test_single_core_to_PS_COCO_multisizes(ccagt_df_multi, tmpdir):
    ccagt_df_multi['area'] = CCAgT.geometries_area(ccagt_df_multi)
    ccagt_df_multi['image_id'] = CCAgT.generate_ids(ccagt_df_multi['image_name'])
    ccagt_df_multi['iscrowd'] = 0
    ccagt_df_multi['color'] = Color(21, 62, 125)

    ccagt_df_multi['image_width'] = 1000
    ccagt_df_multi['image_height'] = 1000

    ccagt_df_multi.loc[ccagt_df_multi['image_id'] == 1, 'image_height'] = 2000

    ccagt_df_multi['geo_type'] = ccagt_df_multi['geometry'].apply(lambda g: g.geom_type).tolist()
    ccagt_df_multi = ccagt_df_multi[ccagt_df_multi['geo_type'] != 'Point']
    single_core_to_PS_COCO(ccagt_df_multi, tmpdir, False)

    assert len(tmpdir.listdir()) > 0


def test_to_OD_COCO(ccagt_df_single_nucleus, coco_OD_ann_single_nucleus):
    with pytest.raises(KeyError):
        to_OD_COCO(ccagt_df_single_nucleus)

    ccagt_df_single_nucleus['area'] = CCAgT.geometries_area(ccagt_df_single_nucleus)
    ccagt_df_single_nucleus['image_id'] = CCAgT.generate_ids(ccagt_df_single_nucleus['image_name'])
    ccagt_df_single_nucleus['iscrowd'] = 0

    coco_OD_ann = to_OD_COCO(ccagt_df_single_nucleus)

    assert coco_OD_ann == coco_OD_ann_single_nucleus


def test_to_PS_COCO_without_cols(ccagt_df_single_nucleus, categories_infos, tmpdir):
    with pytest.raises(KeyError):
        to_PS_COCO(ccagt_df_single_nucleus, categories_infos, tmpdir)


def test_to_PS_COCO(ccagt_df_single_nucleus, categories_infos, tmpdir):
    ccagt_df_single_nucleus['image_name'] = 'C_xx1'
    ccagt_df_single_nucleus['image_width'] = 1600
    ccagt_df_single_nucleus['image_height'] = 1200
    ccagt_df_single_nucleus['area'] = CCAgT.geometries_area(ccagt_df_single_nucleus)
    ccagt_df_single_nucleus['image_id'] = CCAgT.generate_ids(ccagt_df_single_nucleus['image_name'])
    ccagt_df_single_nucleus['iscrowd'] = 0

    out = to_PS_COCO(ccagt_df_single_nucleus, categories_infos, tmpdir, False)

    check1 = all(y in out[0] for y in {'image_id', 'file_name', 'segments_info'})
    assert check1
    check2 = all(y in x for x in out[0]['segments_info'] for y in {'id', 'category_id', 'bbox', 'iscrowd'})
    assert check2

    assert len(tmpdir.listdir()) > 0

    to_PS_COCO(ccagt_df_single_nucleus, categories_infos, tmpdir, True)
    assert len(tmpdir.join('C/').listdir()) > 0

    # With slide id
    df = ccagt_df_single_nucleus.copy()
    df['slide_id'] = CCAgT.slides_ids(df)
    to_PS_COCO(df, categories_infos, tmpdir, True)
    assert len(tmpdir.join('C/').listdir()) > 0
