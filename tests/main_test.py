from __future__ import annotations

import pytest

from CCAgT_utils import main
from testing import create


def test_create_subdataset(capsys, ccagt_ann_multi_path, shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, mask_dir, _ = paths
        out = main.create_subdataset(name='test',
                                     original_dir=tmp_dir,
                                     output_base=tmp_dir,
                                     slice_images=None,
                                     extract=None,
                                     categories_to_keep=None,
                                     categories_to_check=None,
                                     delete=False,
                                     generate_masks=False,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=False)

        assert out == 0
        _, _ = capsys.readouterr()

        out = main.create_subdataset(name='test',
                                     original_dir=tmp_dir,
                                     output_base=tmp_dir,
                                     slice_images=None,
                                     extract=None,
                                     categories_to_keep=None,
                                     categories_to_check=None,
                                     delete=False,
                                     generate_masks=False,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=False)
        _, err = capsys.readouterr()
        assert out == 1
        assert err[:33] == 'Already exist a dataset with name'

        out = main.create_subdataset(name='test1',
                                     original_dir=mask_dir,
                                     output_base=tmp_dir,
                                     slice_images=None,
                                     extract=None,
                                     categories_to_keep=None,
                                     categories_to_check=None,
                                     delete=False,
                                     generate_masks=False,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=False)
        _, err = capsys.readouterr()
        assert out == 1
        assert err[:30] == 'Not found the original data at'


@pytest.mark.slow
def test_create_subdataset_with_slice(shape, ccagt_ann_multi_path, ccagt_ann_multi_image_names):

    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, _, _ = paths
        out = main.create_subdataset(name='test',
                                     original_dir=tmp_dir,
                                     output_base=tmp_dir,
                                     slice_images=(1, 2),
                                     extract=None,
                                     categories_to_keep={1},
                                     categories_to_check={99},
                                     delete=True,
                                     generate_masks=False,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=False)

        assert out == 0

        out = main.create_subdataset('test1',
                                     original_dir=tmp_dir,
                                     output_base=tmp_dir,
                                     slice_images=(1, 2),
                                     extract=None,
                                     categories_to_keep={1},
                                     categories_to_check={99},
                                     delete=True,
                                     generate_masks=True,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=True)
        assert out == 0

        out = main.create_subdataset('test2',
                                     original_dir=tmp_dir,
                                     output_base=tmp_dir,
                                     slice_images=(1, 2),
                                     extract=None,
                                     categories_to_keep={1},
                                     categories_to_check={99},
                                     delete=True,
                                     generate_masks=False,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=False)
        assert out == 0


@pytest.mark.slow
def test_create_subdataset_with_extraction(capsys, shape, ccagt_ann_multi_path, ccagt_ann_multi_image_names):
    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, masks_dir, _ = paths

        out = main.create_subdataset('test',
                                     original_dir=tmp_dir,
                                     output_base=tmp_dir,
                                     slice_images=None,
                                     extract={1},
                                     categories_to_keep=None,
                                     categories_to_check=None,
                                     delete=True,
                                     generate_masks=False,
                                     CCAgT_path=ccagt_ann_multi_path,
                                     paddings='0',
                                     check_if_images_have_annotations=False,
                                     extensions=('.png', '.jpg'),
                                     aux_file_path=None,
                                     overwrite=False)
        assert out == 0
        _, _ = capsys.readouterr()


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


def test_main_create_subdataset(shape, ccagt_ann_multi_path, ccagt_ann_multi_image_names):
    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, masks_dir, _ = paths
        out = main.main(['create-subdataset', '--name', 'example', '--original', tmp_dir, '--output', tmp_dir, '--labels',
                         ccagt_ann_multi_path])

        assert out == 0
