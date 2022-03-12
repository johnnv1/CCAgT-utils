from __future__ import annotations

import os

import pytest

from CCAgT_utils import main
from testing import create


def test_create_subdataset(capsys, shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, masks_dir, _ = paths
        out = main.create_subdataset('test', tmp_dir, tmp_dir, None, None, None, None, False, None, '0')
        assert out == 0
        _, _ = capsys.readouterr()

        out = main.create_subdataset('test', tmp_dir, tmp_dir, None, None, None, None, False, None, '0')
        _, err = capsys.readouterr()
        assert out == 1
        assert err[:33] == 'Already exist a dataset with name'

        out = main.create_subdataset('test1', masks_dir, tmp_dir, None, None, None, None, False, None, '0')
        _, err = capsys.readouterr()
        assert out == 1
        assert err[:36] == 'Do not found the original dataset at'


@pytest.mark.slow
def test_create_subdataset_with_slice(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, masks_dir, _ = paths
        out = main.create_subdataset('test', tmp_dir, tmp_dir, (1, 2), None, {1}, {99}, True, None, '0')
        assert out == 0

        out = main.create_subdataset('test1', tmp_dir, tmp_dir, (1, 2), None, {1}, {1}, True, None, '0')
        assert out == 0

        out = main.create_subdataset('test2', tmp_dir, tmp_dir, (1, 2), None, {1}, {99}, False, None, '0')
        assert out == 0


@pytest.mark.slow
def test_create_subdataset_with_extraction(capsys, shape, ccagt_ann_single_nucleus):
    img_name = ccagt_ann_single_nucleus.df.iloc[0]['image_name']
    with create.ImageMaskFiles(shape[0], shape[1], [img_name]) as paths:
        tmp_dir, masks_dir, _ = paths
        labels_path = os.path.join(tmp_dir, 'labels.parquet.gzip')
        ccagt_ann_single_nucleus.to_parquet(labels_path)
        out = main.create_subdataset('test', tmp_dir, tmp_dir, None, {1}, None, None, None, labels_path, '0')
        assert out == 0
        _, _ = capsys.readouterr()

        out = main.create_subdataset('test1', tmp_dir, tmp_dir, None, {1}, None, None, None, None, '0')
        assert out == 1
        _, err = capsys.readouterr()
        assert err[:80] == 'When using `--extract`, please specify the labels files with `--labels` argument'


def test_not_implemented_command():
    exit_code = main.main(['create-subdataset', '--name', '', '--original', '', '--output', ''])
    assert exit_code == 1


def test_main_help():
    with pytest.raises(SystemExit):
        main.main(['help'])

    with pytest.raises(SystemExit):
        main.main([])


def test_main_help_other_command():
    with pytest.raises(SystemExit):
        main.main(['help', 'create-subdataset'])


def test_main_create_subdataset(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, masks_dir, _ = paths
        out = main.main(['create-subdataset', '--name', 'example', '--original', tmp_dir, '--output', tmp_dir])

        assert out == 0
