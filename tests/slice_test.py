from __future__ import annotations

import os
from collections import Counter

import pytest
from shapely import affinity

from CCAgT_utils import slice
from CCAgT_utils.formats.annotation import Annotation
from testing import create


def test_slice_image(shape, tmpdir):
    with create.ImageMaskFiles(str(tmpdir), shape[0], shape[1], ['example'], create_image=False) as paths:
        tmp_dir, masks_dir, _ = paths
        mask_path = os.path.join(masks_dir, 'example.png')
        sliced_quantity = slice.image(mask_path, tmp_dir, 2, 1)

        assert sliced_quantity == 2
        assert os.path.isfile(os.path.join(tmp_dir, 'example_1.png'))
        assert os.path.isfile(os.path.join(tmp_dir, 'example_2.png'))


@pytest.mark.slow
def test_image_with_annotation(shape, annotations_ex, cluster_ex, tmpdir):
    with create.ImageMaskFiles(str(tmpdir), shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, _, images_dir = paths
        img_path = os.path.join(images_dir, 'A_example.jpg')

        # test stop when take all annotations, and slice annotations into others (if a annotation intersects the new size)
        sliced_quantity, annotations_items = slice.image_with_annotation(img_path, tmp_dir, annotations_ex, 100, 100)

        assert sliced_quantity == 6
        assert all('image_name' in x for x in annotations_items)
        assert all('geometry' in x for x in annotations_items)
        assert all('category_id' in x for x in annotations_items)
        assert os.path.isfile(os.path.join(tmp_dir, 'A_example_1.jpg'))

        # test stops when finish the slices, and copy the annotation (if a annotation is completely inside of the new size)
        ann_items = annotations_ex[:]
        ann_items.append(Annotation(affinity.translate(cluster_ex, -9999, -9999), 2))
        sliced_quantity, annotations_items = slice.image_with_annotation(img_path, tmp_dir, ann_items, 1, 2)

        assert sliced_quantity == 1
        assert len(annotations_items) == len(annotations_ex)
        assert all('image_name' in x for x in annotations_items)
        assert all('geometry' in x for x in annotations_items)
        assert all('category_id' in x for x in annotations_items)


@pytest.mark.slow
def test_single_core_image_and_annotations(shape, ccagt_df_single_nucleus, tmpdir):
    ccagt_df_single_nucleus['image_name'] = 'A_example'
    with create.ImageMaskFiles(str(tmpdir), shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, _, images_dir = paths

        imgs = {'A_example': os.path.join(images_dir, 'A_example.jpg')}

        images_dir_out = os.path.join(images_dir, 'A')
        os.makedirs(images_dir_out)

        sliced_quantity, annotations_items = slice.single_core_image_and_annotations(
            imgs,
            ccagt_df_single_nucleus,
            tmp_dir,
            2,
            2,
        )
        assert sliced_quantity == 1
        assert len(annotations_items) == 1
        assert os.path.isfile(os.path.join(images_dir_out, 'A_example_1.jpg'))


@pytest.mark.slow
def test_slice_images_and_annotations(shape, ccagt_multi_image_names, ccagt_ann_multi_path, tmpdir):
    with create.ImageMaskFiles(str(tmpdir), shape[0], shape[1], ccagt_multi_image_names) as paths:
        tmp_dir, _, images_dir = paths
        out_label = os.path.join(tmp_dir, 'CCAgT.parquet.gzip')
        slice.images_and_annotations(
            dir_images=images_dir,
            annotations_path=ccagt_ann_multi_path,
            dir_output=tmp_dir,
            output_annotations_path=out_label,
            h_quantity=1,
            v_quantity=2,
            extension=('.png', '.jpg'),
        )

        slides = [x.split('_')[0] for x in ccagt_multi_image_names]
        slides_counter = Counter(slides)
        assert os.path.isfile(out_label)
        assert all(len(os.listdir(os.path.join(tmp_dir, 'images', s))) == qtd for s, qtd in slides_counter.items())
