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


def test_extract_category(shape, annotations_ex):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example'], create_image=False) as paths:
        tmp_dir, masks_dir, _ = paths
        msk_path = os.path.join(masks_dir, 'A_example.png')

        extracted_quantity, annotations_items = prepare.extract_category(msk_path,
                                                                         tmp_dir,
                                                                         [(annotations_ex[0], [annotations_ex[1]])],
                                                                         0)

        assert extracted_quantity == 1
        assert len(annotations_items) == 2
        assert all('image_name' in x for x in annotations_items)
        assert all('geometry' in x for x in annotations_items)
        assert all('category_id' in x for x in annotations_items)
        image_name_out = [x['image_name'] for x in annotations_items]
        assert all('A_example_1_1' == x for x in image_name_out)
        assert [x['category_id'] for x in annotations_items] == [1, 2]
        assert annotations_items[0]['geometry'].equals(annotations_ex[0].geometry)
        assert annotations_items[1]['geometry'].equals(annotations_ex[1].geometry)
        assert os.path.isfile(os.path.join(tmp_dir, 'A_example_1_1.png'))


def test_single_core_extract_image_and_annotations(shape, ccagt_ann_single_nucleus):
    img_name = ccagt_ann_single_nucleus.df.iloc[0]['image_name']
    slide = img_name.split('_')[0]
    with create.ImageMaskFiles(shape[0], shape[1], [img_name]) as paths:
        tmp_dir, _, images_dir = paths

        imgs = {img_name: os.path.join(images_dir, f'{img_name}.jpg')}

        images_dir_out = os.path.join(images_dir, f'{slide}')
        os.makedirs(images_dir_out)

        extracted_quantity, annotations_items = prepare.single_core_extract_image_and_annotations(imgs,
                                                                                                  ccagt_ann_single_nucleus,
                                                                                                  {1},
                                                                                                  tmp_dir,
                                                                                                  0)

        assert extracted_quantity == 1
        assert len(annotations_items) == 1

        extracted_quantity, annotations_items = prepare.single_core_extract_image_and_annotations(imgs,
                                                                                                  ccagt_ann_single_nucleus,
                                                                                                  {0},
                                                                                                  tmp_dir,
                                                                                                  0)

        assert extracted_quantity == 0
        assert len(annotations_items) == 0


def test_extract_image_and_mask_by_category(shape,
                                            ccagt_ann_multi_path,
                                            ccagt_ann_multi_image_names,
                                            ccagt_df_multi):
    df_filtered = ccagt_df_multi[ccagt_df_multi['category_id'] == 1]
    with create.ImageMaskFiles(shape[0], shape[1], ccagt_ann_multi_image_names) as paths:
        tmp_dir, _, images_dir = paths

        prepare.extract_image_and_annotations_by_category(images_dir, tmp_dir, {1}, ccagt_ann_multi_path, 0)

        assert all(
            os.path.isfile(
                os.path.join(
                    images_dir, row['image_name'].split('_')[0],
                    f'{row["image_name"]}_{row["category_id"]}_1.jpg'
                )
            )
            for _, row in df_filtered.iterrows()
        )
