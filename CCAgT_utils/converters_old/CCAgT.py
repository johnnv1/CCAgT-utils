from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.utils import basename
from CCAgT_utils.base.utils import get_traceback
from CCAgT_utils.converters import COCO
from CCAgT_utils.converters.masks import annotations_to_mask
from CCAgT_utils.converters.masks import draw_annotation
from CCAgT_utils.converters.masks import order_annotations_to_draw
from CCAgT_utils.formats import CCAgT
from CCAgT_utils.formats.annotation import Annotation


def to_OD_COCO(
    df: pd.DataFrame,
    decimals: int = 2,
) -> list[dict[str, Any]]:

    cols = df.columns
    if not all(c in cols for c in ['area', 'image_id', 'iscrowd']):
        raise KeyError('The dataframe need to have the columns `area`, `image_id`, `iscrowd`!')

    cpu_num = multiprocessing.cpu_count()

    # To ensure that will have sequential index
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1

    # Split equals the annotations for the cpu quantity
    ann_ids_splitted = np.array_split(df.index.tolist(), cpu_num)
    print(f'Number of cores: {cpu_num}, annotations per core: {len(ann_ids_splitted[0])}')

    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for ann_ids in ann_ids_splitted:
        df_to_process = df.loc[df.index.isin(ann_ids), :]
        p = workers.apply_async(single_core_to_OD_COCO, (df_to_process, decimals))
        processes.append(p)

    annotations_coco = []
    for p in processes:
        annotations_coco.extend(p.get())

    return annotations_coco


def generate_masks(
    df: pd.DataFrame,
    out_dir: str,
    split_by_slide: bool = True,
) -> None:

    if split_by_slide:
        if 'slide_id' not in df.columns:
            df['slide_id'] = CCAgT.get_slide_id()

        slide_ids = df['slide_id'].unique()

        for slide_id in slide_ids:
            os.makedirs(os.path.join(out_dir, slide_id), exist_ok=True)

    cpu_num = multiprocessing.cpu_count()

    img_ids = df['image_id'].unique()
    if len(img_ids) == 0:
        print('Do not have annotations to generate the masks!', file=sys.stderr)
        return

    # Split equals the annotations for the cpu quantity
    images_ids_splitted = np.array_split(img_ids, cpu_num)
    print(
        f'Start to generate a total of {len(img_ids)} semantic segmentation masks based on the annotations using '
        f'{cpu_num} cores with {len(images_ids_splitted[0])} masks with annotations per core...',
    )

    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for images_ids in images_ids_splitted:
        df_to_process = df.loc[df['image_id'].isin(images_ids), :]
        p = workers.apply_async(single_core_to_mask, (df_to_process, out_dir, split_by_slide))
        processes.append(p)

    workers.close()
    workers.join()


def to_PS_COCO(
        df: pd.DataFrame,
        categories_infos: CategoriesInfos,
        out_dir: str,
        split_by_slide: bool = True,
) -> list[Any]:
    cols = df.columns
    if not all(c in cols for c in ['image_id', 'iscrowd']):
        raise KeyError('The dataframe need to have the columns `image_id`, `iscrowd`!')

    df['color'] = df['category_id'].apply(lambda cat_id: categories_infos.generate_random_color(cat_id))

    if split_by_slide:
        if 'slide_id' not in df.columns:
            df['slide_id'] = CCAgT.get_slide_id()

        slide_ids = df['slide_id'].unique()

        for slide_id in slide_ids:
            os.makedirs(os.path.join(out_dir, slide_id), exist_ok=True)

    cpu_num = multiprocessing.cpu_count()
    images_ids = df['image_id'].unique()
    images_ids_splitted = np.array_split(images_ids, cpu_num)

    print(
        f'Start compute generate panoptic annotations and masks for {len(images_ids)} files using {cpu_num} cores with '
        f'{len(images_ids_splitted[0])} files per core...',
    )

    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for images_ids in images_ids_splitted:
        if len(images_ids) == 0:
            continue
        df_to_process = df.loc[df['image_id'].isin(images_ids), :]
        p = workers.apply_async(single_core_to_PS_COCO, (df_to_process, out_dir, split_by_slide))
        processes.append(p)

    annotations_panoptic = []
    for p in processes:
        annotations_panoptic.extend(p.get())

    return annotations_panoptic


@get_traceback
def single_core_to_PS_COCO(
    df: pd.DataFrame,
    out_dir: str,
    split_by_slide: bool,
) -> list[dict[str, Any]]:
    annotations_panoptic = []

    _out_dir = out_dir

    if df['image_width'].nunique() == df['image_height'].nunique() == 1:
        w = int(df['image_width'].unique()[0])
        h = int(df['image_height'].unique()[0])
        output_template = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    else:
        output_template = None

    for img_id, df_by_img in df.groupby('image_id'):
        img_name = df_by_img.iloc[0]['image_name']
        output_basename = basename(img_name) + '.png'
        panoptic_record = {
            'image_id': int(img_id),
            'file_name': output_basename,
        }

        annotations_sorted = order_annotations_to_draw([
            Annotation(
                row['geometry'],
                row['category_id'],
                row['iscrowd'],
                row['color'],
            ) for _, row in df_by_img.iterrows()
        ])

        segments_info = []

        if isinstance(output_template, Image.Image):
            out = output_template.copy()
        else:
            w = int(df_by_img['image_width'].unique()[0])
            h = int(df_by_img['image_height'].unique()[0])
            out = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

        for ann in annotations_sorted:
            out = draw_annotation(out, ann, ann.color.rgb)
            segments_info.append({
                'id': COCO.color_to_id(ann.color),
                'category_id': ann.category_id,
                'area': int(ann.geometry.area),
                'bbox': ann.coco_bbox,
                'iscrowd': ann.iscrowd,
            })

        panoptic_record['segments_info'] = segments_info

        if split_by_slide:
            _out_dir = os.path.join(out_dir, df_by_img.iloc[0]['slide_id'])

        output_filename = os.path.join(_out_dir, output_basename)
        out.save(output_filename)
        annotations_panoptic.append(panoptic_record)
    return annotations_panoptic


@get_traceback
def single_core_to_OD_COCO(
    df: pd.DataFrame,
    decimals: int = 2,
) -> list[dict[str, Any]]:
    return df.apply(
        lambda row: {
            'id': row.name,
            'image_id': row['image_id'],
            'category_id': row['category_id'],
            'bbox': COCO.bounds_to_coco_bb(row['geometry'].bounds, decimals),
            'segmentation': COCO.geometry_to_coco_segment(row['geometry'], decimals),
            'area': np.round(row['area'], decimals),
            'iscrowd': row['iscrowd'],
        },
        axis=1,
    ).to_numpy().tolist()


@get_traceback
def single_core_to_mask(
    df: pd.DataFrame,
    out_dir: str,
    split_by_slide: bool,
    extension: str = '.png',
) -> None:
    df_groupped = df.groupby('image_id')
    _out_dir = out_dir

    for _, df_by_img in df_groupped:
        if split_by_slide:
            _out_dir = os.path.join(out_dir, df_by_img.iloc[0]['slide_id'])
        img_name = basename(df_by_img.iloc[0]['image_name']) + extension
        out_path = os.path.join(_out_dir, img_name)

        annotations = [Annotation(row['geometry'], row['category_id']) for _, row in df_by_img.iterrows()]
        w = df_by_img['image_width'].unique()[0]
        h = df_by_img['image_height'].unique()[0]

        mask = annotations_to_mask(annotations, int(w), int(h))

        mask.save(out_path)
