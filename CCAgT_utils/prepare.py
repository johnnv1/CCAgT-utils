from __future__ import annotations

import multiprocessing
import os
from typing import Any

import numpy as np
from PIL import Image
from shapely import affinity

from CCAgT_utils.base.categories import Categories
from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.geometries import point_to_polygon
from CCAgT_utils.base.utils import basename
from CCAgT_utils.base.utils import create_structure
from CCAgT_utils.base.utils import find_files
from CCAgT_utils.base.utils import get_traceback
from CCAgT_utils.base.utils import slide_from_filename
from CCAgT_utils.checkers import masks_that_has
from CCAgT_utils.formats import CCAgT
from CCAgT_utils.formats.annotation import Annotation


def clean_images_and_masks(
    dir_images: str,
    dir_masks: str,
    categories: set[int],
    extension: str | tuple[str, ...] = ('.png', '.jpg'),
    look_recursive: bool = True,
) -> None:

    basenames_matches = masks_that_has(dir_masks, categories, extension, look_recursive)

    image_filenames_to_remove = {
        v for k, v in find_files(dir_images, extension, look_recursive).items()
        if basename(k) not in basenames_matches
    }

    mask_filenames_to_remove = {
        v for k, v in find_files(dir_masks, extension, look_recursive).items()
        if basename(k) not in basenames_matches
    }

    print(f'Deleting {len(image_filenames_to_remove)} images files...')
    for filename in image_filenames_to_remove:
        os.remove(filename)

    print(f'Deleting {len(mask_filenames_to_remove)} masks files...')
    for filename in mask_filenames_to_remove:
        os.remove(filename)


def extract_category(
    input_path: str,
    output_path: str,
    annotations: list[tuple[Annotation, list[Annotation]]],
    padding: int | float,
) -> tuple[int, list[dict[str, Any]]]:
    im = np.asarray(Image.open(input_path))

    bn, ext = os.path.splitext(basename(input_path, with_extension=True))

    height, width = im.shape[:2]
    annotations_out = []
    for count, (ann, group) in enumerate(annotations, start=1):
        basename_img = f'{bn}_{ann.category_id}_{count}'
        filename_img = os.path.join(output_path, f'{basename_img}{ext}')

        bbox = ann.bbox
        bbox.add_padding(padding, (0, 0, width, height))

        x_off = -1 * bbox.x_init
        y_off = -1 * bbox.y_init

        ann.geometry = affinity.translate(ann.geometry, x_off, y_off)
        annotations_out.append({
            'image_name': basename_img,
            'geometry': ann.geometry,
            'category_id': ann.category_id,
            'image_width': bbox.width,
            'image_height': bbox.height,
        })

        for ann_overlapped in group:
            ann_overlapped.geometry = affinity.translate(ann_overlapped.geometry, x_off, y_off)
            annotations_out.append({
                'image_name': basename_img,
                'geometry': ann_overlapped.geometry,
                'category_id': ann_overlapped.category_id,
                'image_width': bbox.width,
                'image_height': bbox.height,
            })

        part = im[bbox.slice_y, bbox.slice_x]
        Image.fromarray(part).save(
            os.path.join(output_path, filename_img),
            quality=100,
            subsampling=0,
        )

    return len(annotations), annotations_out


@get_traceback
def single_core_extract_image_and_annotations(
    image_filenames: dict[str, str],
    ccagt_df: CCAgT.CCAgT,
    categories_to_extract: set[int],
    base_dir_output: str,
    padding: int | float,
) -> tuple[int, list[dict[str, Any]]]:

    def get_match_group(g: list[set[int]], ann_id: int) -> set[int]:
        for i in g:
            if ann_id in i:
                return i
        return set({})

    def get_annotations(idx: set[int], ccagt_ann: CCAgT.CCAgT = ccagt_df) -> list[Annotation]:
        _df = ccagt_ann[ccagt_ann.index.isin(idx)]
        return [Annotation(row['geometry'], row['category_id']) for _, row in _df.iterrows()]

    _c = {Categories.CLUSTER.value, Categories.SATELLITE.value}
    groups: dict[str, list[set[int]]] = {k: [] for k in ccagt_df['image_name'].unique()}
    if Categories.NUCLEUS.value in categories_to_extract:
        _c.add(Categories.NUCLEUS.value)
        groups.update(CCAgT.find_overlapping_annotations(ccagt_df, _c))
    elif Categories.OVERLAPPED_NUCLEI.value in categories_to_extract:
        _c.add(Categories.OVERLAPPED_NUCLEI.value)
        groups.update(CCAgT.find_overlapping_annotations(ccagt_df, _c))

    df_filtered = ccagt_df.loc[
        ccagt_df['category_id'].isin(categories_to_extract),
        ['image_name', 'geometry', 'category_id'],
    ]

    image_counter = 0
    annotations_out = []
    for bn, sub_df in df_filtered.groupby('image_name'):
        ann_items = [
            (
                Annotation(row['geometry'], row['category_id']),
                get_annotations(get_match_group(groups[bn], idx)),
            ) for idx, row in sub_df.iterrows()
        ]

        img_counter, ann_out = extract_category(
            image_filenames[bn],
            os.path.join(base_dir_output, 'images/', slide_from_filename(bn)),
            ann_items,
            padding,
        )
        image_counter += img_counter
        annotations_out.extend(ann_out)
    return (image_counter, annotations_out)


def extract_image_and_annotations_by_category(
    dir_images: str,
    dir_output: str,
    categories: set[int],
    annotations_path: str,
    paddings: int | float,
    extension: str | tuple[str, ...] = ('.png', '.jpg'),
    look_recursive: bool = True,
) -> None:
    ccagt_df = CCAgT.load(annotations_path)
    ann_qtd = ccagt_df.shape[0]

    image_filenames = {basename(k): v for k, v in find_files(dir_images, extension, look_recursive).items()}

    slides = {slide_from_filename(i) for i in image_filenames}
    create_structure(dir_output, slides)

    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)

    filenames_splitted = np.array_split(ccagt_df['image_name'].unique(), cpu_num)
    print(
        f'Start the extraction of each category instance at {len(image_filenames)} images with annotations using {cpu_num} '
        f'cores with {len(filenames_splitted[0])} images with annotations per core...',
    )

    processes = []
    for filenames in filenames_splitted:
        if len(filenames) == 0:
            continue  # pragma: no cover

        img_filenames = {k: image_filenames[k] for k in filenames}
        _ccagt = ccagt_df.copy()
        _ccagt = CCAgT.CCAgT(_ccagt[_ccagt['image_name'].isin(filenames)])

        p = workers.apply_async(
            single_core_extract_image_and_annotations, (
                img_filenames,
                _ccagt,
                categories,
                dir_output,
                paddings,
            ),
        )
        processes.append(p)

    image_counter = 0
    ann_out = []
    for p in processes:
        im_counter, _ann_out = p.get()
        image_counter += im_counter
        ann_out.extend(_ann_out)

    print('Creating the annotation file...')
    CCAgT.save(CCAgT.CCAgT(ann_out), annotations_path)

    print(
        f'Successful transformed from {len(image_filenames)}/{ann_qtd} images/annotations into {image_counter}/{len(ann_out)}'
        ' images/annotations',
    )


def ccagt_dataset(
    ccagt_df: CCAgT.CCAgT,
    categories_infos: CategoriesInfos,
    image_extension: str = '',
    do_fit_geometries: bool = True,
    image_shape: tuple[int, int] = (1600, 1200),
) -> CCAgT.CCAgT:
    print('Start the preprocessing default pipeline for CCAGgT dataset...')
    ovlp_ncl = Categories.OVERLAPPED_NUCLEI.value
    sat = Categories.SATELLITE.value
    cur = Categories.CLUSTER.value
    ncl = Categories.NUCLEUS.value

    print(f'Searching overlapping and joining labels for overlapping annotations (category id = {ovlp_ncl})...')
    overlapping_annotations = CCAgT.find_overlapping_annotations(ccagt_df, categories_id={ovlp_ncl})
    ccagt_df = CCAgT.union_geometries(ccagt_df, overlapping_annotations)

    print(f'Define the geometry type and transform Satellite (category id = {sat}) points into Polygons...')
    ccagt_df['geo_type'] = CCAgT.geometries_type(ccagt_df)
    sat_series = ccagt_df.loc[(ccagt_df['category_id'] == sat) & (ccagt_df['geo_type'] == 'Point'), 'geometry']

    ccagt_df.loc[
        (ccagt_df['category_id'] == sat) &
        (ccagt_df['geo_type'] == 'Point'), 'geometry',
    ] = point_to_polygon(sat_series, area_size=90, resolution=4, tolerance=.3)

    ccagt_df['geo_type'] = CCAgT.geometries_type(ccagt_df)

    if do_fit_geometries:
        print('Verify and fit the geometries into the images boundary...')
        ccagt_df['geometry'] = CCAgT.fit_geometries_to_boundary(ccagt_df, width=image_shape[0], height=image_shape[1])
        ccagt_df.dropna(axis=0, subset=['geometry'], inplace=True)

    if len(image_extension) > 0:
        print(f'Setting {image_extension} as the images extensions')
        ccagt_df['image_name'] = ccagt_df['image_name'].apply(lambda x: x + image_extension)

    print('Computing the annotations area and the images IDs...')
    ccagt_df['area'] = CCAgT.geometries_area(ccagt_df)
    ccagt_df['image_id'] = CCAgT.generate_ids(ccagt_df['image_name'])
    ccagt_df['slide_id'] = CCAgT.slides_ids(ccagt_df)

    print('Deleting annotations based on the minimal area from categories infos')
    ccagt_df = CCAgT.delete_by_area(ccagt_df, categories_infos)

    print(f'Searching intersections of nuclei with NORs labels (category id in [{ncl}] and [{cur}, {sat}])...')
    df_base_intersects_target = CCAgT.verify_if_intersects(ccagt_df, base_categories_id={ncl}, target_categories_id={cur, sat})
    if not df_base_intersects_target.empty:
        index_to_drop = df_base_intersects_target[~df_base_intersects_target['has_intersecting']].index.to_numpy()
        print(f'A total of {len(index_to_drop)} nuclei without NORs (category id = {ncl}) will be deleted.')
        ccagt_df.drop(index_to_drop, inplace=True)

    print(
        f'Searching intersections of NORs with nuclei (normal and overlapped) labels (category id in [{cur}, {sat}] and '
        f'[{ncl}, {ovlp_ncl}])..',
    )
    df_base_intersects_target = CCAgT.verify_if_intersects(
        ccagt_df,
        base_categories_id={cur, sat},
        target_categories_id={ncl, ovlp_ncl},
    )
    if not df_base_intersects_target.empty:
        index_to_drop = df_base_intersects_target[~df_base_intersects_target['has_intersecting']].index.to_numpy()
        print(f'A total of {len(index_to_drop)} NORs without Nucleus (category id =  [{cur}, {sat}]) will be deleted.')
        ccagt_df.drop(index_to_drop, inplace=True)

    return ccagt_df
