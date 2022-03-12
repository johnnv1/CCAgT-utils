from __future__ import annotations

import shutil

import pytest

from CCAgT_utils.categories import Helper
from CCAgT_utils.visualization import _show
from testing import create


@pytest.mark.slow
def test_image_and_mask(remove_plt_show, ccagt_ann_multi, categories_aux_data, shape):
    names = ccagt_ann_multi.df['image_name'].unique()
    with create.ImageMaskFiles(shape[0], shape[1], names) as paths:
        _, mask_dir, image_dir = paths
        _params = {'CCAgT_ann': ccagt_ann_multi,
                   'CCAgT_helper': Helper(categories_aux_data),
                   'dir_path': image_dir,
                   'dir_mask_path': mask_dir,
                   'images_extension': '.jpg',
                   'masks_extension': '.png'}
        out = _show.image_and_mask(**_params, images_names=['example'])
        out1 = _show.image_and_mask(**_params)
        out2 = _show.image_and_mask(**_params, look_recursive=False, shuffle_images=False)

    assert out == 0
    assert out1 == 0
    assert out2 == 0


@pytest.mark.slow
def test_image_and_mask_not_found(capsys, ccagt_ann_multi, categories_aux_data, shape):
    names = ccagt_ann_multi.df['image_name'].unique()
    with create.ImageMaskFiles(shape[0], shape[1], names) as paths:
        _, mask_dir, image_dir = paths
        _params = {'CCAgT_ann': ccagt_ann_multi,
                   'CCAgT_helper': Helper(categories_aux_data),
                   'dir_path': image_dir,
                   'dir_mask_path': mask_dir,
                   'images_extension': '.jpg',
                   'masks_extension': '.png'}

        shutil.rmtree(mask_dir)
        out1 = _show.image_and_mask(**_params, look_recursive=False, shuffle_images=False)
        _, err1 = capsys.readouterr()

        shutil.rmtree(image_dir)
        out2 = _show.image_and_mask(**_params, look_recursive=False, shuffle_images=False)
        _, err2 = capsys.readouterr()

    assert out1 == 0
    assert 'Not found the mask' in err1
    assert out2 == 0
    assert 'Not found the image' in err2


@pytest.mark.slow
def test_image_with_boxes(ccagt_ann_multi, categories_aux_data, shape, remove_plt_show):
    names = ccagt_ann_multi.df['image_name'].unique()
    with create.ImageMaskFiles(shape[0], shape[1], names, create_mask=False) as paths:
        _, _, image_dir = paths
        _params = {'CCAgT_ann': ccagt_ann_multi,
                   'CCAgT_helper': Helper(categories_aux_data),
                   'dir_path': image_dir,
                   'images_extension': '.jpg'}

        out = _show.image_with_boxes(**_params)
        out1 = _show.image_with_boxes(**_params, images_names=['example'])
        out2 = _show.image_with_boxes(**_params, look_recursive=False, shuffle_images=False)

        shutil.rmtree(image_dir)
    assert out == 0
    assert out1 == 0
    assert out2 == 0


@pytest.mark.slow
def test_image_with_boxes_not_found(capsys, ccagt_ann_multi, categories_aux_data, shape, remove_plt_show):
    names = ccagt_ann_multi.df['image_name'].unique()
    with create.ImageMaskFiles(shape[0], shape[1], names, create_mask=False) as paths:
        _, _, image_dir = paths
        _params = {'CCAgT_ann': ccagt_ann_multi,
                   'CCAgT_helper': Helper(categories_aux_data),
                   'dir_path': image_dir,
                   'images_extension': '.jpg'}

        shutil.rmtree(image_dir)
        out = _show.image_with_boxes(**_params, look_recursive=False, shuffle_images=False)
        _, err = capsys.readouterr()

    assert out == 0
    assert 'Not found the image' in err


@pytest.mark.slow
def test_image_with_boxes_and_mask(remove_plt_show, ccagt_ann_multi, categories_aux_data, shape):
    names = ccagt_ann_multi.df['image_name'].unique()
    with create.ImageMaskFiles(shape[0], shape[1], names) as paths:
        _, mask_dir, image_dir = paths
        _params = {'CCAgT_ann': ccagt_ann_multi,
                   'CCAgT_helper': Helper(categories_aux_data),
                   'dir_path': image_dir,
                   'dir_mask_path': mask_dir,
                   'images_extension': '.jpg',
                   'masks_extension': '.png'}
        out = _show.image_with_boxes_and_mask(**_params, images_names=['example'])
        out1 = _show.image_with_boxes_and_mask(**_params)
        out2 = _show.image_with_boxes_and_mask(**_params, look_recursive=False, shuffle_images=False)

    assert out == 0
    assert out1 == 0
    assert out2 == 0


def test_image_with_boxes_and_mask_not_found(capsys, ccagt_ann_multi, categories_aux_data, shape):
    names = ccagt_ann_multi.df['image_name'].unique()
    with create.ImageMaskFiles(shape[0], shape[1], names) as paths:
        _, mask_dir, image_dir = paths
        _params = {'CCAgT_ann': ccagt_ann_multi,
                   'CCAgT_helper': Helper(categories_aux_data),
                   'dir_path': image_dir,
                   'dir_mask_path': mask_dir,
                   'images_extension': '.jpg',
                   'masks_extension': '.png'}

        shutil.rmtree(mask_dir)
        out1 = _show.image_with_boxes_and_mask(**_params, look_recursive=False, shuffle_images=False)
        _, err1 = capsys.readouterr()

        shutil.rmtree(image_dir)
        out2 = _show.image_with_boxes_and_mask(**_params, look_recursive=False, shuffle_images=False)
        _, err2 = capsys.readouterr()

    assert out1 == 0
    assert 'Not found the mask' in err1
    assert out2 == 0
    assert 'Not found the image' in err2
