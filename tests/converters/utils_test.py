from __future__ import annotations

import json
import os
import tempfile

import pytest

from CCAgT_utils.converters import CCAgT
from CCAgT_utils.converters import utils
from CCAgT_utils.errors import FileTypeError
from testing.create import RawAuxFiles


def test_build_description(ccagt_df_multi, ccagt_metadata):
    df = ccagt_df_multi.copy()
    CCAgT_ann = CCAgT.CCAgT(df)
    df['image_id'] = CCAgT_ann.generate_ids(df['image_name'])
    df['slide_id'] = CCAgT_ann.get_slide_id()

    template = ccagt_metadata['description_template']

    out = utils.__build_description(template, df)
    assert out == '''
qtd of images = 3
qtd of slides = 2
qtd of annotations = 11
'''


def test_open_and_read_json(lbb_raw_sample_complete):
    aux = [{'': None}]
    with RawAuxFiles(lbb_raw_sample_complete, aux) as paths:
        _, raw_path, aux_path = paths
        out = utils.open_and_read_json(raw_path)
        out1 = utils.open_and_read_json(aux_path)

    assert out == lbb_raw_sample_complete
    assert out1 == aux


def test_labelbox_to_COCO_wrong_filetypes():
    with pytest.raises(FileTypeError):
        utils.labelbox_to_COCO('', 'raw.txt', 'aux.json', 'out.json', '', 2)
    with pytest.raises(FileTypeError):
        utils.labelbox_to_COCO('', 'raw.json', 'aux.txt', 'out.json', '', 2)
    with pytest.raises(FileTypeError):
        utils.labelbox_to_COCO('', 'raw.json', 'aux.json', 'out.txt', '', 2)


def test_labelbox_to_COCO_wrong_target():
    out = utils.labelbox_to_COCO('', 'raw.json', 'aux.json', 'out.json', '', 2)
    assert out == 1


def test_labelbox_to_COCO_OD(lbb_raw_sample_complete, ccagt_aux_data):
    with RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.json')
        o = utils.labelbox_to_COCO('OD', raw_path, aux_path, out_filename, '', 2)
    assert o == 0


def test_labelbox_to_COCO_IS():
    with pytest.raises(NotImplementedError):
        utils.labelbox_to_COCO('IS', 'raw.json', 'aux.json', 'out.json', '', 2)


def test_labelbox_to_COCO_PS():
    with pytest.raises(NotImplementedError):
        utils.labelbox_to_COCO('PS', 'raw.json', 'aux.json', 'out.json', '', 2)


def test_labelbox_to_CCAgT_wrong_filetypes():
    with pytest.raises(FileTypeError):
        utils.labelbox_to_CCAgT('raw.txt', 'aux.json', 'out.parquet', '')

    with pytest.raises(FileTypeError):
        utils.labelbox_to_CCAgT('raw.json', 'aux.txt', 'out.parquet', '')

    with pytest.raises(FileTypeError):
        utils.labelbox_to_CCAgT('raw.json', 'aux.json', 'out.json', '')


def test_labelbox_to_CCAgT(lbb_raw_sample_complete, ccagt_aux_data):
    with RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.parquet')
        out = utils.labelbox_to_CCAgT(raw_path, aux_path, out_filename, '.jpg', True)

        out1_filename = os.path.join(temp_dir, 'out.parquet.gzip')
        out1 = utils.labelbox_to_CCAgT(raw_path, aux_path, out1_filename, '', False)

    assert out == out1 == 0


def test_labelbox_to_CCAgT_without_valid_geometries(lbb_raw_single_wrong_nucleus, ccagt_aux_data):
    with RawAuxFiles([lbb_raw_single_wrong_nucleus], ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.parquet')
        out = utils.labelbox_to_CCAgT(raw_path, aux_path, out_filename, '', True)

    assert out == 0


@pytest.mark.slow
def test_ccagt_generate_masks(ccagt_ann_single_nucleus):
    ccagt_ann_single_nucleus.df['image_id'] = 1
    with tempfile.TemporaryDirectory() as tmp_dir:
        ccagt_path = os.path.join(tmp_dir, 'ccagt.parquet.gzip')
        ccagt_ann_single_nucleus.to_parquet(ccagt_path)
        assert utils.ccagt_generate_masks(ccagt_path, tmp_dir, split_by_slide=False) == 0


def test_ccagt_wrong_file():
    with pytest.raises(FileTypeError):
        utils.ccagt_generate_masks('wrong.name', '/tmp/', False)


def test_CCAgT_to_PS_COCO(ccagt_ann_single_nucleus, categories_infos, tmpdir):
    outfilename = os.path.join(tmpdir, 'test_CCAgT_to_PS_COCO.json')
    ccagt_ann_single_nucleus.df['area'] = ccagt_ann_single_nucleus.geometries_area()
    ccagt_ann_single_nucleus.df['image_id'] = ccagt_ann_single_nucleus.generate_ids(ccagt_ann_single_nucleus.df['image_name'])
    ccagt_ann_single_nucleus.df['slide_id'] = ccagt_ann_single_nucleus.get_slide_id()

    out = utils.CCAgT_to_PS_COCO(ccagt_ann_single_nucleus, categories_infos, tmpdir, outfilename, {'year': 'sample'}, False)

    assert out == 0

    with open(outfilename) as f:
        itens = json.load(f)

    assert all(x in itens for x in {'info', 'categories', 'images', 'annotations'})
    assert all(x in itens['categories'][0] for x in {'supercategory', 'name', 'id'})
    assert all(x in itens['images'][0] for x in {'file_name', 'height', 'width', 'id'})
    assert all(x in itens['annotations'][0] for x in {'image_id', 'file_name', 'segments_info'})
    assert all(
        x in j for j in itens['annotations'][0]['segments_info']
        for x in {'id', 'category_id', 'area', 'bbox', 'iscrowd'}
    )


def test_CCAgT_to_COCO_NotImplemented():
    with pytest.raises(NotImplementedError):
        utils.CCAgT_to_COCO('IS', '', None, '', None, False)


def test_CCAgT_to_COCO_PS_with_auxfile(ccagt_ann_single_nucleus, ccagt_aux_data):
    ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'

    with RawAuxFiles([{'a': None}], ccagt_aux_data) as paths:
        temp_dir, _, aux_path = paths
        ccagt_path = os.path.join(temp_dir, 'ccagt.parquet.gzip')
        ccagt_ann_single_nucleus.to_parquet(ccagt_path)
        out = utils.CCAgT_to_COCO('PS', ccagt_path, aux_path, temp_dir, None, False)

        assert out == 0


def test_CCAgT_to_COCO_OD_with_auxfile(ccagt_ann_single_nucleus, ccagt_aux_data):
    ccagt_ann_single_nucleus.df['image_name'] = 'C_xx1'

    with RawAuxFiles([{'a': None}], ccagt_aux_data) as paths:
        temp_dir, _, aux_path = paths
        ccagt_path = os.path.join(temp_dir, 'ccagt.parquet.gzip')
        ccagt_ann_single_nucleus.to_parquet(ccagt_path)
        out = utils.CCAgT_to_COCO('OD', ccagt_path, aux_path, temp_dir, None, False)

        assert out == 0


def test_CCAgT_to_COCO_wrong_target(ccagt_ann_single_nucleus, tmpdir):
    ccagt_path = os.path.join(tmpdir, 'ccagt.parquet.gzip')
    ccagt_ann_single_nucleus.to_parquet(ccagt_path)
    out = utils.CCAgT_to_COCO('wrong target', ccagt_path, None, tmpdir, None, False)

    assert out == 1
