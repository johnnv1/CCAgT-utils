from __future__ import annotations

import os

import pytest

from CCAgT_utils import slice
from testing import create


def test_slice_image(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['example'], create_image=False) as paths:
        tmp_dir, masks_dir, _ = paths
        mask_path = os.path.join(masks_dir, 'example.png')
        sliced_quantity = slice.image(mask_path, tmp_dir, 2, 1)

        assert sliced_quantity == 2
        assert os.path.isfile(os.path.join(tmp_dir, 'example_1.png'))
        assert os.path.isfile(os.path.join(tmp_dir, 'example_2.png'))


@pytest.mark.slow
def test_image_with_annotation(shape, annotations_ex):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, _, images_dir = paths
        img_path = os.path.join(images_dir, 'A_example.jpg')

        sliced_quantity, annotations_items = slice.image_with_annotation(img_path, tmp_dir, annotations_ex, 2, 2)

        assert sliced_quantity == 1
        assert all('image_name' in x for x in annotations_items)
        assert all('geometry' in x for x in annotations_items)
        assert all('category_id' in x for x in annotations_items)
        assert os.path.isfile(os.path.join(tmp_dir, 'A_example_1.jpg'))


@pytest.mark.slow
def test_single_core_image_and_annotations(shape, ccagt_df_single_nucleus):
    ccagt_df_single_nucleus['image_name'] = 'A_example'
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, _, images_dir = paths

        imgs = {'A_example': os.path.join(images_dir, 'A_example.jpg')}

        images_dir_out = os.path.join(images_dir, 'A')
        os.makedirs(images_dir_out)

        sliced_quantity, annotations_items = slice.single_core_image_and_annotations(imgs,
                                                                                     ccagt_df_single_nucleus,
                                                                                     tmp_dir,
                                                                                     2,
                                                                                     2)
        assert sliced_quantity == 1
        assert os.path.isfile(os.path.join(images_dir_out, 'A_example_1.jpg'))


@pytest.mark.slow
def test_slice_image_and_masks(shape, ccagt_ann_multi_image_names, ccagt_ann_multi_path):
    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, masks_dir, images_dir = paths
        out_label = os.path.join(tmp_dir, 'CCAgT.parquet.gzip')
        slice.images_and_annotations(dir_images=images_dir,
                                     annotations_path=ccagt_ann_multi_path,
                                     dir_output=tmp_dir,
                                     output_annotations_path=out_label,
                                     h_quantity=1,
                                     v_quantity=2,
                                     extension=('.png', '.jpg'))

        slides = [x.split('_')[0] for x in ccagt_ann_multi_image_names]
        assert os.path.isfile(out_label)
        assert all(len(os.listdir(os.path.join(tmp_dir, 'images', s))) == 1 for s in slides)
