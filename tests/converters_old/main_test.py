from __future__ import annotations

import os

import pytest

from CCAgT_utils.converters import main
from testing.create import RawAuxFiles


def test_main_error_with_empty_string():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main.main([''])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2


def test_main_labelbox_to_COCO(lbox_raw_sample_complete, ccagt_aux_data):
    with RawAuxFiles(lbox_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.json')
        out = main.main([
            'labelbox_to_COCO', '-t', 'OD',
            '-r', raw_path, '-a', aux_path, '-o', out_filename,
        ])

    assert out == 0


def test_main_labelbox_to_CCAgT(lbox_raw_sample_complete, ccagt_aux_data):
    with RawAuxFiles(lbox_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, raw_path, aux_path = paths
        out_filename = os.path.join(temp_dir, 'out.parquet')
        out = main.main(['labelbox_to_CCAgT', '-r', raw_path, '-a', aux_path, '-o', out_filename])

    assert out == 0


def test_main_CCAgT_to_COCO(ccagt_aux_data, ccagt_ann_single_nucleus, tmpdir):
    ccagt_path = os.path.join(tmpdir, 'ccagt.parquet.gzip')
    ccagt_ann_single_nucleus.to_parquet(ccagt_path)
    with RawAuxFiles([{'a': None}], ccagt_aux_data) as paths:
        temp_dir, _, aux_path = paths
        out = main.main(['CCAgT_to_COCO', '-t', 'PS', '-l', ccagt_path, '-a', aux_path, '-o', temp_dir])

    assert out == 0


def test_main_help():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main.main(['help'])
    assert pytest_wrapped_e1.value.code == 0

    with pytest.raises(SystemExit) as pytest_wrapped_e2:
        main.main([])
    assert pytest_wrapped_e2.value.code == 0


def test_main_help_other_command():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main.main(['help', 'labelbox_to_COCO'])

    assert pytest_wrapped_e1.value.code == 0

    with pytest.raises(SystemExit) as pytest_wrapped_e2:
        main.main(['help', 'labelbox_to_CCAgT'])

    assert pytest_wrapped_e2.value.code == 0

    with pytest.raises(SystemExit) as pytest_wrapped_e3:
        main.main(['help', 'CCAgT_to_COCO'])

    assert pytest_wrapped_e3.value.code == 0


def test_not_implemented_command():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main.main(['wrong_command', '-r', '', '-a', '', '-o', ''])

    assert pytest_wrapped_e.value.code == 2


def test_main_out():
    out = main.main(['generate_masks', '-l', '', '-o', ''])
    assert out == 1


@pytest.mark.slow
def test_main_ccagt_generate_masks(ccagt_ann_single_nucleus, tmpdir):
    ccagt_ann_single_nucleus.df['image_id'] = 1
    ccagt_path = tmpdir.join('ccagt.parquet.gzip')
    ccagt_ann_single_nucleus.to_parquet(str(ccagt_path))

    out = main.main(['generate_masks', '-l', str(ccagt_path), '-o', str(tmpdir)])

    assert out == 0
    assert len(tmpdir.listdir()) == 2
