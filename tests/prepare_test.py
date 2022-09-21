from __future__ import annotations

import os

from CCAgT_utils import prepare
from CCAgT_utils.base.categories import Categories
from testing import create


def test_clean_images_and_masks(shape, tmpdir):
    with create.ImageMaskFiles(
        str(tmpdir), shape[0], shape[1], ['A_example'],
    ) as paths:
        tmp_dir, masks_dir, images_dir = paths
        prepare.clean_images_and_masks(images_dir, masks_dir, {0})

        assert os.path.isfile(
            os.path.join(
                masks_dir, 'A', 'A_example_1.png',
            ),
        ) is False
        assert os.path.isfile(
            os.path.join(
                masks_dir, 'A', 'A_example_2.png',
            ),
        ) is False
        assert os.path.isfile(
            os.path.join(
                images_dir, 'A', 'A_example_1.jpg',
            ),
        ) is False
        assert os.path.isfile(
            os.path.join(
                images_dir, 'A', 'A_example_2.jpg',
            ),
        ) is False


def test_extract_category(shape, annotations_ex, tmpdir):
    with create.ImageMaskFiles(
        str(tmpdir), shape[0], shape[1], ['A_example'], create_image=False,
    ) as paths:
        tmp_dir, masks_dir, _ = paths
        msk_path = os.path.join(masks_dir, 'A_example.png')

        extracted_quantity, annotations_items = prepare.extract_category(
            msk_path,
            tmp_dir,
            [(annotations_ex[0], [annotations_ex[1]])],
            0,
        )

        assert extracted_quantity == 1
        assert len(annotations_items) == 2
        assert all('image_name' in x for x in annotations_items)
        assert all('geometry' in x for x in annotations_items)
        assert all('category_id' in x for x in annotations_items)
        image_name_out = [x['image_name'] for x in annotations_items]
        assert all('A_example_1_1' == x for x in image_name_out)
        assert [x['category_id'] for x in annotations_items] == [1, 2]
        assert annotations_items[0]['geometry'].equals(
            annotations_ex[0].geometry,
        )
        assert annotations_items[1]['geometry'].equals(
            annotations_ex[1].geometry,
        )
        assert os.path.isfile(os.path.join(tmp_dir, 'A_example_1_1.png'))


def test_single_core_extract_image_and_annotations(
    shape, ccagt_df_multi, ccagt_multi_image_names, tmpdir,
):
    df = ccagt_df_multi.copy()
    slides = {img_name.split('_')[0] for img_name in ccagt_multi_image_names}
    with create.ImageMaskFiles(
        str(tmpdir), shape[0], shape[1], ccagt_multi_image_names,
    ) as paths:
        tmp_dir, _, images_dir = paths

        imgs = {
            img_name: os.path.join(
                images_dir, f'{img_name}.jpg',
            ) for img_name in ccagt_multi_image_names
        }

        for slide in slides:
            images_dir_out = os.path.join(images_dir, f'{slide}')
            os.makedirs(images_dir_out, exist_ok=True)

        output = prepare.single_core_extract_image_and_annotations(
            imgs,
            df,
            {Categories.NUCLEUS.value},
            tmp_dir,
            0,
        )
        extracted_qtd, ann_items = output

        assert extracted_qtd == 5
        assert len(ann_items) == 14

        output = prepare.single_core_extract_image_and_annotations(
            imgs,
            df,
            {Categories.BACKGROUND.value},
            tmp_dir,
            0,
        )
        extracted_qtd, ann_items = output
        assert extracted_qtd == 0
        assert len(ann_items) == 0

        df.loc[
            df['category_id'] == Categories.NUCLEUS.value,
            'category_id',
        ] = Categories.OVERLAPPED_NUCLEI.value

        output = prepare.single_core_extract_image_and_annotations(
            imgs,
            df,
            {Categories.OVERLAPPED_NUCLEI.value},
            tmp_dir,
            0,
        )
        extracted_qtd, ann_items = output
        assert extracted_qtd == 5
        assert len(ann_items) == 14


def test_extract_image_and_mask_by_category(
    shape,
    ccagt_ann_multi_path,
    ccagt_multi_image_names,
    ccagt_df_multi,
    tmpdir,
):
    df_filtered = ccagt_df_multi[ccagt_df_multi['category_id'] == 1]
    with create.ImageMaskFiles(
        str(tmpdir), shape[0], shape[1], ccagt_multi_image_names,
    ) as paths:
        tmp_dir, _, images_dir = paths

        prepare.extract_image_and_annotations_by_category(
            images_dir, tmp_dir, {1}, ccagt_ann_multi_path, 0,
        )

        assert all(
            os.path.isfile(
                os.path.join(
                    images_dir, row['image_name'].split('_')[0],
                    f'{row["image_name"]}_{row["category_id"]}_1.jpg',
                ),
            )
            for _, row in df_filtered.iterrows()
        )


def test_ccagt_dataset(ccagt_df_multi, categories_infos):
    in_data = ccagt_df_multi.copy()
    out = prepare.ccagt_dataset(in_data, categories_infos)
    assert out['geo_type'].unique().tolist() == ['Polygon']
    assert 3 not in out['image_id'].unique()
    assert not out.equals(ccagt_df_multi)

    in_data = ccagt_df_multi.copy()
    out = prepare.ccagt_dataset(in_data, categories_infos, '.png', False)
    assert out['geo_type'].unique().tolist() == ['Polygon']
    assert 3 not in out['image_id'].unique()
    assert not out.equals(ccagt_df_multi)

    # Cover skip deletion of Nucleus without NORs and NORs without nuclei
    in_data = ccagt_df_multi.copy()
    in_data = in_data.loc[in_data['category_id'] == -99]
    out = prepare.ccagt_dataset(in_data, categories_infos, '.png', False)
    assert out.empty
