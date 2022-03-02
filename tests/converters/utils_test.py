from __future__ import annotations

import os

import pytest

from CCAgT_utils.converters import CCAgT
from CCAgT_utils.converters import utils
from CCAgT_utils.errors import FileTypeError
from testing.create import RawAuxFiles


def test_build_description(ccagt_df_multi, ccagt_metadata):
    df = ccagt_df_multi.copy()
    CCAgT_ann = CCAgT.CCAgT_Annotations(df)
    df['image_id'] = CCAgT_ann.generate_ids(df['image_name'])
    df['slide_id'] = CCAgT_ann.get_slide_id()

    template = ccagt_metadata['description_template']

    out = utils.__build_description(template, df)
    assert out == '''
qtd of images = 2
qtd of slides = 2
qtd of annotations = 5
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
    assert out == 0


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