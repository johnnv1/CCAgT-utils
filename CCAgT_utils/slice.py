from __future__ import annotations

import multiprocessing
import os
from typing import Any

import numpy as np
from PIL import Image

from CCAgT_utils.utils import basename
from CCAgT_utils.utils import create_structure
from CCAgT_utils.utils import find_files
from CCAgT_utils.utils import get_traceback
from CCAgT_utils.utils import slide_from_filename


def image(input_path: str,
          output_path: str,
          h_quantity: int = 4,
          v_quantity: int = 4) -> int:
    im = np.asarray(Image.open(input_path))

    bn, ext = os.path.splitext(basename(input_path, with_extension=True))

    height, width = im.shape[:2]

    tile_h = height // v_quantity
    tile_w = width // h_quantity

    count = 1
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            part = im[y:y + tile_h, x:x + tile_w]
            Image.fromarray(part).save(os.path.join(output_path, f'{bn}_{count}{ext}'),
                                       quality=100,
                                       subsampling=0)
            count += 1

    return count - 1


@get_traceback
def single_core_image_and_masks(image_filenames: dict[str, str],
                                mask_filenames: dict[str, str],
                                base_dir_output: str,
                                h_quantity: int = 4,
                                v_quantity: int = 4) -> tuple[int, int]:
    image_counter = 0
    mask_counter = 0
    for bn in image_filenames:
        image_counter += image(image_filenames[bn],
                               os.path.join(base_dir_output, 'images/', slide_from_filename(bn)),
                               h_quantity,
                               v_quantity)

        if bn in mask_filenames:
            mask_counter += image(mask_filenames[bn],
                                  os.path.join(base_dir_output, 'masks/', slide_from_filename(bn)),
                                  h_quantity,
                                  v_quantity)

    return (image_counter, mask_counter)


def images_and_masks(dir_images: str,
                     dir_masks: str,
                     dir_output: str,
                     h_quantity: int = 4,
                     v_quantity: int = 4,
                     **kwargs: Any) -> None:

    image_filenames = {basename(k): v for k, v in find_files(dir_images, **kwargs).items()}
    mask_filenames = {basename(k): v for k, v in find_files(dir_masks, **kwargs).items()}

    slides = {slide_from_filename(i) for i in image_filenames}
    create_structure(dir_output, slides)

    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)
    filenames_splitted = np.array_split(list(image_filenames), cpu_num)
    print(f'Start the split of images and masks into {h_quantity}x{v_quantity} parts using {cpu_num} cores with '
          f'{len(filenames_splitted[0])} images and masks per core...')
    processes = []
    for filenames in filenames_splitted:
        img_filenames = {k: image_filenames[k] for k in filenames}
        msk_filenames = {k: mask_filenames[k] for k in filenames}
        p = workers.apply_async(single_core_image_and_masks, (img_filenames,
                                                              msk_filenames,
                                                              dir_output,
                                                              h_quantity,
                                                              v_quantity))
        processes.append(p)

    image_counter = 0
    mask_counter = 0
    for p in processes:
        im_counter, msk_counter = p.get()
        image_counter += im_counter
        mask_counter += msk_counter

    print(f'Successful splitted {len(image_filenames)}/{len(mask_filenames)} images/masks into {image_counter}/{mask_counter}'
          ' images/masks')
