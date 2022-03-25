from __future__ import annotations

import multiprocessing
import os

import numpy as np
import pandas as pd
from PIL import Image

from CCAgT_utils.categories import Categories
from CCAgT_utils.categories import CategoriesInfos
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

    print(f'Deleting {len(image_filenames_to_remove)} images files...')
    for filename in image_filenames_to_remove:
        os.remove(filename)

    print(f'Deleting {len(mask_filenames_to_remove)} masks files...')
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
                                       look_recursive: bool = True) -> None:
    ccagt_annotations = CCAgT.read_parquet(CCAgT_path)

    df = ccagt_annotations.df.loc[ccagt_annotations.df['category_id'].isin(categories),
                                  ['image_name', 'geometry', 'category_id']]

    if df.empty:
        print(f'Nothing to process with categories {categories} from {CCAgT_path}')
        return

    image_filenames = {basename(k): v for k, v in find_files(dir_images, extension, look_recursive).items()}
    mask_filenames = {basename(k): v for k, v in find_files(dir_masks, extension, look_recursive).items()}

    slides = {slide_from_filename(i) for i in image_filenames}
    create_structure(dir_output, slides)

    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)

    filenames_splitted = np.array_split(df['image_name'].unique().tolist(), cpu_num)
    print(f'Start the extraction of each category instance at {len(image_filenames)} images/masks using {cpu_num} cores with '
          f'{len(filenames_splitted[0])} images/masks per core...')

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

    print(f'Successful create {image_counter}/{mask_counter} images/masks for the instances of the categories {categories}')


def ccagt_dataset(ccagt: CCAgT.CCAgT,
                  categories_infos: CategoriesInfos,
                  image_extension: str = '',
                  do_fit_geometries: bool = True
                  ) -> CCAgT.CCAgT:
    print('Start the preprocessing default pipeline for CCAGgT dataset...')
    ovlp_ncl = Categories.OVERLAPPED_NUCLEI.value
    sat = Categories.SATELLITE.value
    cur = Categories.CLUSTER.value
    ncl = Categories.NUCLEUS.value

    print(f'Searching overlapping and joining labels for overlapping annotations (category id = {ovlp_ncl})...')
    overlapping_annotations = ccagt.find_overlapping_annotations(categories_id={ovlp_ncl})
    df = ccagt.union_geometries(overlapping_annotations)

    print(f'Define the geometry type and transform Satellite (category id = {sat}) points into Polygons...')
    df['geo_type'] = ccagt.geometries_type()
    sat_series = df.loc[(df['category_id'] == sat) & (df['geo_type'] == 'Point'), 'geometry']

    df.loc[(df['category_id'] == sat) &
           (df['geo_type'] == 'Point'), 'geometry'] = ccagt.satellite_point_to_polygon(sat_series)

    df['geo_type'] = ccagt.geometries_type()

    if do_fit_geometries:
        print('Verify and fit the geometries into the images boundary...')
        df['geometry'] = ccagt.fit_geometries_to_image_boundary()
        df.dropna(axis=0, subset=['geometry'], inplace=True)

    if len(image_extension) > 0:
        print(f'Setting {image_extension} as the images extensions')
        df['image_name'] = df['image_name'].apply(lambda x: x + image_extension)

    print('Computing the annotations area and the images IDs...')
    df['area'] = ccagt.geometries_area()
    df['image_id'] = ccagt.generate_ids(df['image_name'])
    df['slide_id'] = ccagt.get_slide_id()

    print('Deleting annotations based on the minimal area from categories infos')
    df = ccagt.delete_by_area(categories_infos)

    print(f'Searching intersections of nuclei with NORs labels (category id in [{ncl}] and [{cur}, {sat}])...')
    df_base_intersects_target = ccagt.verify_if_intersects(base_categories_id={ncl}, target_categories_id={cur, sat})
    if not df_base_intersects_target.empty:
        index_to_drop = df_base_intersects_target[~df_base_intersects_target['has_intersecting']].index.to_numpy()
        print(f'A total of {len(index_to_drop)} of nuclei (category id = {ncl}) will be deleted.')
        df.drop(index_to_drop, inplace=True)

    print(f'Searching intersections of NORs with nuclei (normal and overlapped) labels (category id in [{cur}, {sat}] and '
          f'[{ncl}, {ovlp_ncl}])..')
    df_base_intersects_target = ccagt.verify_if_intersects(base_categories_id={cur, sat}, target_categories_id={ncl, ovlp_ncl})
    if not df_base_intersects_target.empty:
        index_to_drop = df_base_intersects_target[~df_base_intersects_target['has_intersecting']].index.to_numpy()
        print(f'A total of {len(index_to_drop)} of NORs (category id =  [{cur}, {sat}]) will be deleted.')
        df.drop(index_to_drop, inplace=True)

    return ccagt
