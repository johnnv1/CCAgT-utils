from __future__ import annotations

import multiprocessing
import os

import numpy as np
import pandas as pd
from PIL import Image

from CCAgT_utils.checkers import masks_that_has
from CCAgT_utils.converters import CCAgT
from CCAgT_utils.types.annotation import Annotation
from CCAgT_utils.utils import basename
from CCAgT_utils.utils import create_structure
from CCAgT_utils.utils import find_files
from CCAgT_utils.utils import get_traceback
from CCAgT_utils.utils import slide_from_filename


def clean_images_and_masks(dir_images: str,
                           dir_masks: str,
                           categories: set[int],
                           extension: str | tuple[str, ...] = ('.png', '.jpg'),
                           look_recursive: bool = True) -> None:

    basenames_matches = masks_that_has(dir_masks, categories, extension, look_recursive)

    image_filenames_to_remove = {v for k, v in find_files(dir_images, extension, look_recursive).items()
                                 if basename(k) not in basenames_matches}

    mask_filenames_to_remove = {v for k, v in find_files(dir_masks, extension, look_recursive).items()
                                if basename(k) not in basenames_matches}

    print('Deleting images files...')
    for filename in image_filenames_to_remove:
        os.remove(filename)

    print('Deleting masks files...')
    for filename in mask_filenames_to_remove:
        os.remove(filename)


def extract_category_from_image_file(input_path: str,
                                     output_path: str,
                                     annotations: list[Annotation],
                                     padding: int | float) -> int:
    im = np.asarray(Image.open(input_path))

    bn, ext = os.path.splitext(basename(input_path, with_extension=True))

    height, width = im.shape[:2]

    count = 1
    for ann in annotations:
        bb = ann.bbox
        bb.add_padding(padding, (0, 0, width, height))
        part = im[bb.slice_y, bb.slice_x]
        Image.fromarray(part).save(os.path.join(output_path, f'{bn}_{ann.category_id}_{count}{ext}'),
                                   quality=100,
                                   subsampling=0)
        count += 1

    return len(annotations)


@get_traceback
def single_core_extract_image_and_masks(image_filenames: dict[str, str],
                                        mask_filenames: dict[str, str],
                                        df_annotations: pd.DataFrame,
                                        base_dir_output: str,
                                        padding: int | float) -> tuple[int, int]:
    image_counter = 0
    mask_counter = 0
    for bn, sub_df in df_annotations.groupby('image_name'):

        anns = [Annotation(row['geometry'], row['category_id']) for _, row in sub_df.iterrows()]

        if bn in image_filenames:
            image_counter += extract_category_from_image_file(image_filenames[bn],
                                                              os.path.join(base_dir_output, 'images/', slide_from_filename(bn)),
                                                              anns,
                                                              padding)
        if bn in mask_filenames:
            mask_counter += extract_category_from_image_file(mask_filenames[bn],
                                                             os.path.join(base_dir_output, 'masks/', slide_from_filename(bn)),
                                                             anns,
                                                             padding)
    return (image_counter, mask_counter)


def extract_image_and_mask_by_category(dir_images: str,
                                       dir_masks: str,
                                       dir_output: str,
                                       categories: set[int],
                                       CCAgT_path: str,
                                       paddings: int | float,
                                       extension: str | tuple[str, ...] = ('.png', '.jpg'),
                                       look_recursive: bool = True) -> int:
    print(f'Extracting images and masks of the categories {categories} from {dir_images} and {dir_masks}')

    print(f'\tLoading labels from {CCAgT_path}')
    ccagt_annotations = CCAgT.read_parquet(CCAgT_path)

    df = ccagt_annotations.df.loc[ccagt_annotations.df['category_id'].isin(categories),
                                  ['image_name', 'geometry', 'category_id']]

    if df.empty:
        print(f'\tNothing to process with categories {categories} from {CCAgT_path}')
        return 0

    print(f'\tFinding all `{extension}` files into the directory {dir_images}...')
    image_filenames = {basename(k): v for k, v in find_files(dir_images, extension, look_recursive).items()}

    print(f'\tFinding all `{extension}` files into the directory {dir_masks}...')
    mask_filenames = {basename(k): v for k, v in find_files(dir_masks, extension, look_recursive).items()}

    print('\tCreating output directories...')
    slides = {slide_from_filename(i) for i in image_filenames}
    create_structure(dir_output, slides)

    print('\tStart extracting into multiprocessing...')
    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)

    filenames_splitted = np.array_split(df['image_name'].unique().tolist(), cpu_num)
    print(f'\t\tNumber of cores: {cpu_num}, images and masks per core: {len(filenames_splitted[0])}')

    processes = []
    for filenames in filenames_splitted:
        img_filenames = {k: image_filenames[k] for k in filenames}
        msk_filenames = {k: mask_filenames[k] for k in filenames}
        p = workers.apply_async(single_core_extract_image_and_masks, (img_filenames,
                                                                      msk_filenames,
                                                                      df[df['image_name'].isin(filenames)],
                                                                      dir_output,
                                                                      paddings))
        processes.append(p)

    image_counter = 0
    mask_counter = 0
    for p in processes:
        im_counter, msk_counter = p.get()
        image_counter += im_counter
        mask_counter += msk_counter

    print(f'\tSuccessful create {image_counter}/{mask_counter} images/masks of categories {categories}')
    return 0


"""
TODO: Entry point

CCAgT-utils create-subdataset --name <name> (--slice-images [(horizontal), (vertical)] | --extract <category id>)--original
<path to original dataset with images/ and masks/ dirs> --remove-images-without Sequence[categories ids] --output-path <path
to local to create the new subdataset> --check-if-all-have-at-least-one-of Sequence[categories ids]

D2
CCAgT-utils create-subdataset --name Dataset2 --slice-images [4, 4] --original <2.1.1/original/> --remove-images-without
[1,2,3,4,5,6,7] --output-path <2.1.1/>

D3
CCAgT-utils create-subdataset --name Dataset3 --original <2.1.1/original/> --remove-images-without [1] --output-path <2.1.1/>
--check-if-all-have-at-least-one-of {2, 3}

D4
CCAgT-utils create-subdataset --name Dataset4 --slice-images [4, 4] --original <2.1.1/original/> --remove-images-without [1]
--output-path <2.1.1/> --check-if-all-have-at-least-one-of {2, 3}

D5
CCAgT-utils create-subdataset --name Dataset5 --extract 1 --check-if-all-have-at-least-one-of {2, 3}
"""
