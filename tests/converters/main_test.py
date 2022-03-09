from __future__ import annotations

import os
import tempfile

import pytest

from CCAgT_utils.converters import main
from testing.create import RawAuxFiles


def test_main_error_with_empty_string():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main.main([''])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2


def test_main_labelbox_to_COCO(lbb_raw_sample_complete, ccagt_aux_data):
    with RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.json')
        out = main.main(['labelbox_to_COCO', '-t', 'OD',
                         '-r', raw_path, '-a', aux_path, '-o', out_filename])

    assert out == 0


def test_main_labelbox_to_CCAgT(lbb_raw_sample_complete, ccagt_aux_data):
    with RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.parquet')
        out = main.main(['labelbox_to_CCAgT', '-r', raw_path, '-a', aux_path, '-o', out_filename])

    assert out == 0


def test_main_help():
    with pytest.raises(SystemExit):
        main.main(['help'])

    with pytest.raises(SystemExit):
        main.main([])


def test_main_help_other_command():
    with pytest.raises(SystemExit):
        main.main(['help', 'labelbox_to_COCO'])

    with pytest.raises(SystemExit):
        main.main(['help', 'labelbox_to_CCAgT'])


def test_not_implemented_command():
    exit_code = main.main(['labelbox_to_CCAgT', '-r', '', '-a', '', '-o', ''])
    assert exit_code == 1


def test_main_ccagt_generate_masks(ccagt_ann_single_nucleus):
    ccagt_ann_single_nucleus.df['image_id'] = 1
    with tempfile.TemporaryDirectory() as tmp_dir:
        ccagt_path = os.path.join(tmp_dir, 'ccagt.parquet.gzip')
        ccagt_ann_single_nucleus.to_parquet(ccagt_path)

        out = main.main(['generate_masks', '-l', ccagt_path, '-o', tmp_dir])

    assert out == 0
