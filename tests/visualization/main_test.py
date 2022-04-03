from __future__ import annotations

import os

import pytest

from CCAgT_utils.visualization import main
from testing import create


def test_main_error_with_empty_string():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main.main([''])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2


def test_main_help():
    with pytest.raises(SystemExit):
        main.main(['help'])

    with pytest.raises(SystemExit):
        main.main([])


def test_main_help_other_command():
    with pytest.raises(SystemExit):
        main.main(['help', 'show'])


def test_not_implemented_command():
    exit_code = main.main(['show', '-l', '', '-a', ''])
    assert exit_code == 1


@pytest.mark.slow
def test_main_show_images_with_boxes(ccagt_ann_multi, lbb_raw_sample_complete, ccagt_aux_data):
    with create.RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, _, aux_path = paths
        labels_path = os.path.join(temp_dir, 'labels.parquet.gzip')
        ccagt_ann_multi.to_parquet(labels_path)
        out = main.main([
            'show', '-t', 'image-with-boxes', '-l', labels_path, '-a', aux_path,
            '-d', temp_dir, '-i', 'none_example',
        ])

    assert out == 0


@pytest.mark.slow
def test_main_show_image_and_mask(ccagt_ann_multi, lbb_raw_sample_complete, ccagt_aux_data):
    with create.RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, _, aux_path = paths
        labels_path = os.path.join(temp_dir, 'labels.parquet.gzip')
        ccagt_ann_multi.to_parquet(labels_path)
        out = main.main([
            'show', '--plot-type', 'image-and-mask', '-l', labels_path, '-a', aux_path,
            '-d', temp_dir, '-m', temp_dir, '-i', 'none_example',
        ])

        exit_code = main.main([
            'show', '--plot-type', 'image-and-mask', '-l', labels_path, '-a', aux_path,
            '-d', '', '-m', temp_dir, '-i', 'none_example',
        ])

    assert out == 0
    assert exit_code == 1


def test_main_show_image_with_boxes_and_mask(ccagt_ann_multi, lbb_raw_sample_complete, ccagt_aux_data):
    with create.RawAuxFiles(lbb_raw_sample_complete, ccagt_aux_data) as paths:
        temp_dir, _, aux_path = paths
        labels_path = os.path.join(temp_dir, 'labels.parquet.gzip')
        ccagt_ann_multi.to_parquet(labels_path)
        out = main.main([
            'show', '--plot-type', 'image-with-boxes-and-mask', '-l', labels_path, '-a', aux_path,
            '-d', temp_dir, '-m', temp_dir, '-i', 'none_example',
        ])

        exit_code = main.main([
            'show', '--plot-type', 'image-with-boxes-and-mask', '-l', labels_path, '-a', aux_path,
            '-d', '', '-m', temp_dir, '-i', 'none_example',
        ])

    assert out == 0
    assert exit_code == 1
