from __future__ import annotations

import os

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


def test_single_core_image_and_masks(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, masks_dir, images_dir = paths

        imgs = {'A_example': os.path.join(images_dir, 'A_example.jpg')}
        msks = {'A_example': os.path.join(masks_dir, 'A_example.png')}

        masks_dir_out = os.path.join(masks_dir, 'A/')
        os.makedirs(masks_dir_out)

        images_dir_out = os.path.join(images_dir, 'A/')
        os.makedirs(images_dir_out)

        sliced_quantity = slice.single_core_image_and_masks(imgs, msks, tmp_dir, 1, 2)

        assert sliced_quantity == (2, 2)
        assert os.path.isfile(os.path.join(masks_dir_out, 'A_example_1.png'))
        assert os.path.isfile(os.path.join(masks_dir_out, 'A_example_2.png'))
        assert os.path.isfile(os.path.join(images_dir_out, 'A_example_1.jpg'))
        assert os.path.isfile(os.path.join(images_dir_out, 'A_example_2.jpg'))

        msks = {}
        sliced_quantity = slice.single_core_image_and_masks(imgs, msks, tmp_dir, 1, 2)
        assert sliced_quantity == (2, 0)


def test_slice_image_and_masks(shape):
    with create.ImageMaskFiles(shape[0], shape[1], ['A_example']) as paths:
        tmp_dir, masks_dir, images_dir = paths
        slice.images_and_masks(images_dir, masks_dir, tmp_dir, 1, 2, extension=('.png', '.jpg'))

        assert os.path.isfile(os.path.join(masks_dir, 'A/', 'A_example_1.png'))
        assert os.path.isfile(os.path.join(masks_dir, 'A/', 'A_example_2.png'))
        assert os.path.isfile(os.path.join(images_dir, 'A/', 'A_example_1.jpg'))
        assert os.path.isfile(os.path.join(images_dir, 'A/', 'A_example_2.jpg'))
