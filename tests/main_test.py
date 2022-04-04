from __future__ import annotations

import os
import shutil

import pytest

from CCAgT_utils import main
from testing import create


def test_create_subdataset(capsys, ccagt_ann_multi, ccagt_ann_multi_path, ccagt_ann_multi_image_names, shape, ccagt_aux_data):
    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, mask_dir, img_dir = paths
        with create.RawAuxFiles([], ccagt_aux_data) as paths:
            _, _, aux_path = paths
            out = main.create_subdataset(
                name='test',
                original_dir=tmp_dir,
                output_base=tmp_dir,
                slice_images=None,
                extract=None,
                categories_to_keep=(0, {1}),
                categories_to_check={1, 2},
                delete=True,
                generate_masks=True,
                CCAgT_path=ccagt_ann_multi_path,
                paddings='0',
                check_if_images_have_annotations=True,
                extensions=('.png', '.jpg'),
                aux_file_path=aux_path,
                overwrite=False,
            )

            assert out == 0
            _, _ = capsys.readouterr()

            out = main.create_subdataset(
                name='test',
                original_dir=tmp_dir,
                output_base=tmp_dir,
                slice_images=None,
                extract=None,
                categories_to_keep=(1, {1}),
                categories_to_check={1, 2},
                delete=True,
                generate_masks=True,
                CCAgT_path=ccagt_ann_multi_path,
                paddings='0',
                check_if_images_have_annotations=True,
                extensions=('.png', '.jpg'),
                aux_file_path=aux_path,
                overwrite=True,
            )
            assert out == 0
            _, _ = capsys.readouterr()

        # Already exist this subdataset
        out = main.create_subdataset(
            name='test',
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
            overwrite=False,
        )
        _, err = capsys.readouterr()
        assert out == 1
        assert err[:33] == 'Already exist a dataset with name'

        # Wrong original dataset path
        out = main.create_subdataset(
            name='test1',
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
            overwrite=False,
        )
        _, err = capsys.readouterr()
        assert out == 1
        assert err[:30] == 'Not found the original data at'

        # try default location for annotations path, and not found the file
        out = main.create_subdataset(
            name='test2',
            original_dir=tmp_dir,
            output_base=tmp_dir,
            slice_images=None,
            extract=None,
            categories_to_keep=None,
            categories_to_check=None,
            delete=False,
            generate_masks=False,
            CCAgT_path=None,
            paddings='0',
            check_if_images_have_annotations=False,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=True,
        )

        _, err = capsys.readouterr()
        assert out == 1
        assert err[:33] == 'Not found the annotations file at'

        # Wrong choice for the removal process
        out = main.create_subdataset(
            name='test3',
            original_dir=tmp_dir,
            output_base=tmp_dir,
            slice_images=None,
            extract=None,
            categories_to_keep=(-1, {1}),
            categories_to_check={1, 2},
            delete=False,
            generate_masks=False,
            CCAgT_path=ccagt_ann_multi_path,
            paddings='0',
            check_if_images_have_annotations=False,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=False,
        )

        assert out == 1
        _, err = capsys.readouterr()
        assert err == 'Unexpected choice for the type of removal proccess.\n'

        # Cover nothing to delete after check_if_images_have_annotations
        ccagt_ann_multi.df = ccagt_ann_multi.df.loc[ccagt_ann_multi.df['image_id'] == 1]
        ccagt_ann_path = os.path.join(tmp_dir, 'CCAgT.parquet.gzip')
        ori_test4 = os.path.join(tmp_dir, 'oritest4')
        ori_test4_img_dir = os.path.join(ori_test4, 'images')
        os.makedirs(ori_test4_img_dir, exist_ok=True)
        bn_img = ccagt_ann_multi.df['image_name'].to_numpy()[0] + '.jpg'
        shutil.copy2(os.path.join(img_dir, bn_img), os.path.join(ori_test4_img_dir, bn_img))
        ccagt_ann_multi.to_parquet(ccagt_ann_path)

        out = main.create_subdataset(
            name='test4',
            original_dir=ori_test4,
            output_base=tmp_dir,
            slice_images=None,
            extract=None,
            categories_to_keep=None,
            categories_to_check=None,
            delete=True,
            generate_masks=False,
            CCAgT_path=ccagt_ann_path,
            paddings='0',
            check_if_images_have_annotations=True,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=False,
        )
        assert out == 0
        _, _ = capsys.readouterr()

        # Cover early stop because don't have any annotation after removal proccess
        out = main.create_subdataset(
            name='test5',
            original_dir=tmp_dir,
            output_base=tmp_dir,
            slice_images=None,
            extract=None,
            categories_to_keep=(1, {}),
            categories_to_check=None,
            delete=True,
            generate_masks=False,
            CCAgT_path=ccagt_ann_multi_path,
            paddings='0',
            check_if_images_have_annotations=True,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=False,
        )
        assert out == 1
        _, err = capsys.readouterr()
        assert err == 'The annotations file has none annotation, just finishing the process!\n'


@pytest.mark.slow
def test_create_subdataset_with_slice(shape, ccagt_ann_multi_path, ccagt_ann_multi_image_names):

    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, _, _ = paths
        out = main.create_subdataset(
            name='test',
            original_dir=tmp_dir,
            output_base=tmp_dir,
            slice_images=(1, 2),
            extract=None,
            categories_to_keep=(0, {1}),
            categories_to_check={99},
            delete=True,
            generate_masks=False,
            CCAgT_path=ccagt_ann_multi_path,
            paddings='0',
            check_if_images_have_annotations=False,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=False,
        )

        assert out == 0

        out = main.create_subdataset(
            name='test1',
            original_dir=tmp_dir,
            output_base=tmp_dir,
            slice_images=(1, 2),
            extract=None,
            categories_to_keep=(0, {1}),
            categories_to_check={99},
            delete=True,
            generate_masks=True,
            CCAgT_path=ccagt_ann_multi_path,
            paddings='0',
            check_if_images_have_annotations=False,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=True,
        )
        assert out == 0

        out = main.create_subdataset(
            name='test2',
            original_dir=tmp_dir,
            output_base=tmp_dir,
            slice_images=(1, 2),
            extract=None,
            categories_to_keep=(0, {1}),
            categories_to_check={99},
            delete=True,
            generate_masks=False,
            CCAgT_path=ccagt_ann_multi_path,
            paddings='0',
            check_if_images_have_annotations=False,
            extensions=('.png', '.jpg'),
            aux_file_path=None,
            overwrite=False,
        )
        assert out == 0


@pytest.mark.slow
def test_create_subdataset_with_extraction(capsys, shape, ccagt_ann_multi_path, ccagt_ann_multi_image_names):
    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, masks_dir, _ = paths

        out = main.create_subdataset(
            name='test',
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
            overwrite=False,
        )
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
        tmp_dir, _, _ = paths
        out = main.main([
            'create-subdataset', '--name', 'example', '--original', tmp_dir, '--output', tmp_dir, '--labels',
            ccagt_ann_multi_path,
        ])
        assert out == 0

        out = main.main([
            'create-subdataset', '--name', 'example1', '--original', tmp_dir, '--output', tmp_dir, '--labels',
            ccagt_ann_multi_path, '--remove-images-without', '1',
        ])
        assert out == 0

        out = main.main([
            'create-subdataset', '--name', 'example2', '--original', tmp_dir, '--output', tmp_dir, '--labels',
            ccagt_ann_multi_path, '--remove-annotations-different', '1', '--check-if-all-have-at-least-one-of',
            '1',
        ])
        assert out == 0

# TODO: a test for the pipelines -> ex: input 1 image slice and check if the output is a sliced image and masks as expected
