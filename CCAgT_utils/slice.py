from __future__ import annotations

import multiprocessing
import os
from typing import Any
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image
from shapely import affinity

from CCAgT_utils.converters.CCAgT import CCAgT
from CCAgT_utils.converters.CCAgT import read_parquet
from CCAgT_utils.types.annotation import Annotation
from CCAgT_utils.types.annotation import BBox
from CCAgT_utils.utils import basename
from CCAgT_utils.utils import create_structure
from CCAgT_utils.utils import find_files
from CCAgT_utils.utils import get_traceback
from CCAgT_utils.utils import slide_from_filename


def __create_xy_slice(height: int, width: int, tile_h: int, tile_w: int) -> Iterator[BBox]:
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            yield BBox(x, y, tile_w, tile_h, -1)


def image(
    input_path: str,
    output_path: str,
    h_quantity: int = 4,
    v_quantity: int = 4,
) -> int:
    im = np.asarray(Image.open(input_path))

    bn, ext = os.path.splitext(basename(input_path, with_extension=True))

    height, width = im.shape[:2]
    tile_h = height // v_quantity
    tile_w = width // h_quantity
    count = 1
    for bbox in __create_xy_slice(height, width, tile_h, tile_w):
        part = im[bbox.slice_y, bbox.slice_x]
        Image.fromarray(part).save(
            os.path.join(output_path, f'{bn}_{count}{ext}'),
            quality=100,
            subsampling=0,
        )
        count += 1

    return count - 1


def image_with_annotation(
    input_path: str,
    output_path: str,
    annotation_items: list[Annotation],
    h_quantity: int = 4,
    v_quantity: int = 4,
) -> tuple[int, list[dict[str, Any]]]:
    im = np.asarray(Image.open(input_path))

    bn, ext = os.path.splitext(basename(input_path, with_extension=True))

    height, width = im.shape[:2]

    tile_h = height // v_quantity
    tile_w = width // h_quantity

    ann_to_ignore = []
    count = 1
    annotations_out = []
    for bbox in __create_xy_slice(height, width, tile_h, tile_w):
        basename_img = f'{bn}_{count}'
        filename_img = os.path.join(output_path, f'{basename_img}{ext}')
        bbox_pol = bbox.to_polygon()
        x_off = -1 * bbox.x_init
        y_off = -1 * bbox.y_init
        img_annotations = []
        for idx, ann in enumerate(annotation_items):
            if idx in ann_to_ignore:
                continue

            _test = False
            ann_to_use = ann.copy()
            if bbox_pol.contains(ann.geometry):
                ann_to_ignore.append(idx)
                _test = True
            elif ann.geometry.intersects(bbox_pol):
                ann_to_use.geometry = ann.geometry.intersection(bbox_pol)
                _test = True

            if _test:
                ann_to_use.geometry = affinity.translate(ann_to_use.geometry, x_off, y_off)
                img_annotations.append({
                    'image_name': basename_img,
                    'geometry': ann_to_use.geometry,
                    'category_id': ann_to_use.category_id,
                    'image_width': bbox.width,
                    'image_height': bbox.height,
                })

        if len(img_annotations) > 0:
            annotations_out.extend(img_annotations)
            part = im[bbox.slice_y, bbox.slice_x]
            Image.fromarray(part).save(
                filename_img,
                quality=100,
                subsampling=0,
            )
            count += 1

        if len(annotation_items) == len(ann_to_ignore):
            break

    return (count - 1, annotations_out)


@get_traceback
def single_core_image_and_annotations(
    image_filenames: dict[str, str],
    df_ccagt: pd.DataFrame,
    base_dir_output: str,
    h_quantity: int = 4,
    v_quantity: int = 4,
) -> tuple[int, list[dict[str, Any]]]:
    image_counter = 0
    annotations_out = []
    for bn, df in df_ccagt.groupby('image_name'):
        ann_items = [Annotation(r['geometry'], r['category_id']) for _, r in df.iterrows()]

        img_counter, ann_out = image_with_annotation(
            image_filenames[bn],
            os.path.join(base_dir_output, 'images/', slide_from_filename(bn)),
            ann_items,
            h_quantity,
            v_quantity,
        )
        image_counter += img_counter
        annotations_out.extend(ann_out)
    return (image_counter, annotations_out)


def images_and_annotations(
    dir_images: str,
    annotations_path: str,
    dir_output: str,
    output_annotations_path: str,
    h_quantity: int = 4,
    v_quantity: int = 4,
    **kwargs: Any
) -> None:

    image_filenames = {basename(k): v for k, v in find_files(dir_images, **kwargs).items()}

    ccagt = read_parquet(annotations_path)
    ann_qtd = ccagt.df.shape[0]
    slides = {slide_from_filename(i) for i in image_filenames}
    create_structure(dir_output, slides)

    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)
    filenames_splitted = np.array_split(list(image_filenames), cpu_num)
    print(
        f'Start the split of images and annotations into {h_quantity}x{v_quantity} parts using {cpu_num} cores with '
        f'{len(filenames_splitted[0])} images and masks per core...',
    )

    processes = []
    for filenames in filenames_splitted:
        if len(filenames) == 0:
            continue  # pragma: no cover

        _ccagt = ccagt.df[ccagt.df['image_name'].isin(filenames)]
        img_filenames = {k: image_filenames[k] for k in filenames}
        p = workers.apply_async(
            single_core_image_and_annotations, (
                img_filenames,
                _ccagt,
                dir_output,
                h_quantity,
                v_quantity,
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
    ccagt_out = CCAgT(pd.DataFrame(ann_out))

    ccagt_out.to_parquet(output_annotations_path)

    print(
        f'Successful splitted {len(image_filenames)}/{ann_qtd} images/annotations into {image_counter}/{len(ann_out)}'
        ' images/annotations',
    )
