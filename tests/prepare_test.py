from __future__ import annotations

import os

from CCAgT_utils import prepare
from testing import create


def test_clean_images_and_masks(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, masks_dir, images_dir = paths
        prepare.clean_images_and_masks(images_dir, masks_dir, {0})

        assert os.path.isfile(os.path.join(masks_dir, 'A', 'A_example_1.png')) is False
        assert os.path.isfile(os.path.join(masks_dir, 'A', 'A_example_2.png')) is False
        assert os.path.isfile(os.path.join(images_dir, 'A', 'A_example_1.jpg')) is False
        assert os.path.isfile(os.path.join(images_dir, 'A', 'A_example_2.jpg')) is False


def test_extract_category_from_image_file(shape, annotations_ex):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example'], create_image=False) as paths:
        tmp_dir, masks_dir, _ = paths
        msk_path = os.path.join(masks_dir, 'A_example.png')

        extracted_quantity = prepare.extract_category_from_image_file(msk_path, tmp_dir, [annotations_ex[0]], 0)

        assert extracted_quantity == 1
        assert os.path.isfile(os.path.join(tmp_dir, 'A_example_1_1.png'))


def test_single_core_extract_image_and_masks(shape, ccagt_df_single_nucleus):
    img_name = ccagt_df_single_nucleus.iloc[0]['image_name']
    slide = img_name.split('_')[0]
    with create.ImageMaskFiles(shape[0], shape[1], [img_name]) as paths:
        tmp_dir, masks_dir, images_dir = paths

        imgs = {img_name: os.path.join(images_dir, f'{img_name}.jpg')}
        msks = {img_name: os.path.join(masks_dir, f'{img_name}.png')}

        masks_dir_out = os.path.join(masks_dir, f'{slide}')
        os.makedirs(masks_dir_out)

        images_dir_out = os.path.join(images_dir, f'{slide}')
        os.makedirs(images_dir_out)

        extracted_quantity = prepare.single_core_extract_image_and_masks(imgs, msks, ccagt_df_single_nucleus, tmp_dir, 0)

        assert extracted_quantity == (1, 1)

        extracted_quantity = prepare.single_core_extract_image_and_masks({}, {}, ccagt_df_single_nucleus, tmp_dir, 0)

        assert extracted_quantity == (0, 0)


def test_extract_image_and_mask_by_category(capsys, shape, ccagt_ann_single_nucleus):
    img_name = ccagt_ann_single_nucleus.df.iloc[0]['image_name']
    slide = img_name.split('_')[0]
    with create.ImageMaskFiles(shape[0], shape[1], [img_name]) as paths:
        tmp_dir, masks_dir, images_dir = paths
        labels_path = os.path.join(tmp_dir, 'labels.parquet.gzip')
        ccagt_ann_single_nucleus.to_parquet(labels_path)

        prepare.extract_image_and_mask_by_category(images_dir, masks_dir, tmp_dir, {1}, labels_path, 0)

        assert os.path.isfile(os.path.join(masks_dir, slide, f'{img_name}_1_1.png'))
        _, _ = capsys.readouterr()

        prepare.extract_image_and_mask_by_category(images_dir, masks_dir, tmp_dir, {99}, labels_path, 0)
        out, _ = capsys.readouterr()
        assert out[:34] == 'Nothing to process with categories'
