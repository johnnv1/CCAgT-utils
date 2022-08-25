from __future__ import annotations

import collections
import multiprocessing
import os
import sys
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.base.categories import Categories
from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.utils import basename
from CCAgT_utils.base.utils import get_traceback
from CCAgT_utils.formats import CCAgT
from CCAgT_utils.formats import COCO
from CCAgT_utils.formats.annotation import Annotation
from CCAgT_utils.formats.mask import Mask


# -----------------------------------------------------------------------
# Functions to work with data from LabelBox (lbox)
def lbox_geo_to_shapely(object: dict[str, Any]) -> Polygon | Point | np.nan:
    keys = object.keys()

    if 'polygon' in keys:
        polygon = object['polygon']
        geometry = Polygon(np.array([(p['x'], p['y']) for p in polygon]))
    elif 'point' in keys:
        point = object['point']
        geometry = Point(np.array([point['x'], point['y']]))
    else:
        geometry = np.NaN
    return geometry


def __lbox_has_review(reviews: list[dict[str, Any]]) -> bool:
    for x in reviews:
        if x['score'] == 1:
            return True
    else:
        return False


def __lbox_drop_duplicate_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    duplicated_idx = df['image_name'].duplicated(keep=False)
    df_duplicated = df.loc[duplicated_idx, :].copy()

    if df_duplicated.empty:
        return df

    # Check the labels that has review
    df_duplicated['have_review'] = df_duplicated.apply(lambda row: __lbox_has_review(row['Reviews']), axis=1)

    # Count the quantity of labels for each row
    df_duplicated['len'] = df_duplicated.apply(lambda row: len(row['Label']['objects']), axis=1)

    # Sort the DF by the quantity of labels
    df_duplicated = df_duplicated.sort_values(['image_name', 'len'], ascending=False)

    # Drop the duplicates labels and keep the first label will be that have more labels
    df_to_keep = df_duplicated.drop_duplicates(['image_name'], keep='first')

    id_to_remove = df_duplicated.loc[~df_duplicated['ID'].isin(df_to_keep['ID'].to_numpy()), 'ID']
    # the rows without review
    id_to_remove = pd.concat([id_to_remove, df_duplicated.loc[~df_duplicated['have_review'], 'ID']])

    df_without_duplicated = df[~df['ID'].isin(id_to_remove)].copy()

    return df_without_duplicated


def __lbox_explode_images_annotations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df['objects'] = df.apply(lambda row: row['Label']['objects'], axis=1)
    df = df.explode('objects')
    df = df.reset_index()

    df = df.drop(['index', 'Label'], axis=1)
    return df


def __lbox_cast_geometries(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df['geometry'] = df['objects'].apply(lambda obj: lbox_geo_to_shapely(obj))
    df_out = df.dropna(axis=0, subset=['geometry'])

    if df.shape != df_out.shape:
        print(f'Some NaN geometries have been deleted! Original shape = {df.shape} | out shape = {df_out.shape}')

    if df_out.empty:
        raise ValueError('Data without valid geometries! After transform the geometries the dataframe stay empty.')

    return df_out


def from_labelbox(
    df: pd.DataFrame,
    categories_info: CategoriesInfos,
) -> pd.DataFrame:
    '''Transform the raw dataframe from LabelBox data to CCAgT'''

    df_out = df.copy()

    # Drop ignored images at labelling process
    df_out = df.drop(df[df['Skipped']].index)

    # Drop irrelevant columns
    df_out = df_out.drop(
        [
            'DataRow ID', 'Labeled Data', 'Created By', 'Project Name', 'Dataset Name', 'Created At', 'Updated At',
            'Seconds to Label', 'Agreement', 'Benchmark Agreement', 'Benchmark ID', 'View Label',
            'Has Open Issues', 'Skipped',
        ], axis=1, errors='ignore',
    )

    # Get image names
    df_out['image_name'] = df_out.apply(lambda row: basename(row['External ID']), axis=1)
    df_out = df_out.drop(['External ID'], axis=1)

    # Remove duplicated labels
    df_out = __lbox_drop_duplicate_labels(df_out)

    # Explode annotations to each row
    df_out = __lbox_explode_images_annotations(df_out)

    # Transform labelbox annotation to a geometry
    df_out = __lbox_cast_geometries(df_out)

    # Map category IDs
    schematic_to_id = {cat.labelbox_schemaId: cat.id for cat in categories_info}
    df_out['category_id'] = df_out.apply(lambda row: schematic_to_id[row['objects']['schemaId']], axis=1)

    df_out = df_out.drop(['ID', 'objects', 'Reviews'], axis=1)

    return df_out


# -----------------------------------------------------------------------
# Functions to generate masks
# Order of categories IDs available at the metadata
DRAW_ORDER = (
    Categories.LEUKOCYTE_NUCLEUS,
    Categories.NON_VIABLE_NUCLEUS,
    Categories.OVERLAPPED_NUCLEI,
    Categories.NUCLEUS_OUT_OF_FOCUS,
    Categories.NUCLEUS,
    Categories.CLUSTER,
    Categories.SATELLITE,
)


def order_annotations_to_draw(annotations: list[Annotation]) -> list[Annotation]:
    items = collections.defaultdict(list)
    for ann in annotations:
        items[ann.category_id].append(ann)
    out = [ann for cat in DRAW_ORDER for ann in items[cat.value]]

    return out


def annotations_to_mask(
    annotations: list[Annotation],
    width: int = 1600,
    height: int = 1200,
) -> Mask:
    shape = (height, width)

    annotations_sorted = order_annotations_to_draw(annotations)

    out = Image.fromarray(np.zeros(shape, dtype=np.uint8))
    for ann in annotations_sorted:
        out = draw_annotation(out, ann, ann.category_id)

    return Mask(np.array(out))


def draw_annotation(
    target: Image.Image,
    annotation: Annotation,
    value: int | tuple[int, int, int],
) -> Image.Image:
    for geo in annotation:
        pol_x, pol_y = geo.exterior.coords.xy
        coords = list(zip(pol_x, pol_y))
        ImageDraw.Draw(target).polygon(coords, fill=value)

    return target


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


def to_mask(
    ccagt_df: CCAgT.CCAgT,
    out_dir: str,
    split_by_slide: bool = True,
) -> None:

    if split_by_slide:
        if 'slide_id' not in ccagt_df.columns:
            ccagt_df['slide_id'] = CCAgT.slides_ids(ccagt_df)

        slide_ids = ccagt_df['slide_id'].unique()

        for slide_id in slide_ids:
            os.makedirs(os.path.join(out_dir, slide_id), exist_ok=True)

    cpu_num = multiprocessing.cpu_count()

    img_ids = ccagt_df['image_id'].unique()
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
        df_to_process = ccagt_df.loc[ccagt_df['image_id'].isin(images_ids), :]
        p = workers.apply_async(single_core_to_mask, (df_to_process, out_dir, split_by_slide))
        processes.append(p)

    workers.close()
    workers.join()


# -----------------------------------------------------------------------
# Functions work with data to COCO
def bounds_to_coco_bb(
    bounds: tuple[float],
    decimals: int = 2,
) -> list[float]:
    # bounds is in  (minx, miny, maxx, maxy)
    # bb  of coco is in [min(x), min(y), max(x)-min(x), max(y)-min(y)]
    b = tuple(np.round(x, decimals) for x in bounds)
    min_x, min_y, max_x, max_y = b
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def pol_to_coco_segment(
    geo: Polygon | MultiPolygon,
    decimals: int = 2,
) -> list[list[float]]:
    # polygon of shapely is a class
    # polygon or segmentation at coco is a list of [[x0, y0, x1, y1 ...]]

    def coco_pol(geometry: Polygon, decimals: int = decimals) -> list[float]:
        list_of_points = list(zip(*geometry.exterior.coords.xy))
        geometry = list(chain(*list_of_points))
        geometry = list(np.around(np.array(geometry), decimals))
        return geometry

    if geo.geom_type == 'Polygon':
        return [coco_pol(geo)]
    elif geo.geom_type == 'MultiPolygon':
        return [coco_pol(g) for g in geo.geoms]
    else:
        raise TypeError(f'Geometry shape is not a polygon or MultiPolygon. This is a {geo.geom_type}.')


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
            'bbox': bounds_to_coco_bb(row['geometry'].bounds, decimals),
            'segmentation': pol_to_coco_segment(row['geometry'], decimals),
            'area': np.round(row['area'], decimals),
            'iscrowd': row['iscrowd'],
        },
        axis=1,
    ).to_numpy().tolist()


@get_traceback
def single_core_to_PS_COCO(
    df: pd.DataFrame,
    out_dir: str,
    split_by_slide: bool,
) -> list[dict[str, Any]]:
    annotations_panoptic = []

    _out_dir = out_dir

    if df['image_width'].nunique() == df['image_height'].nunique() == 1:
        w = np.integer(df['image_width'].unique()[0])
        h = np.integer(df['image_height'].unique()[0])
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


def to_PS_COCO(
        ccagt_df: CCAgT.CCAgT,
        categories_infos: CategoriesInfos,
        out_dir: str,
        split_by_slide: bool = True,
) -> list[Any]:
    cols = ccagt_df.columns
    if not all(c in cols for c in ['image_id', 'iscrowd']):
        raise KeyError('The dataframe need to have the columns `image_id`, `iscrowd`!')

    ccagt_df['color'] = ccagt_df['category_id'].apply(lambda cat_id: categories_infos.generate_random_color(cat_id))

    if split_by_slide:
        if 'slide_id' not in ccagt_df.columns:
            ccagt_df['slide_id'] = CCAgT.slides_ids(ccagt_df)

        slide_ids = ccagt_df['slide_id'].unique()

        for slide_id in slide_ids:
            os.makedirs(os.path.join(out_dir, slide_id), exist_ok=True)

    cpu_num = multiprocessing.cpu_count()
    images_ids = ccagt_df['image_id'].unique()
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
        df_to_process = ccagt_df.loc[ccagt_df['image_id'].isin(images_ids), :]
        p = workers.apply_async(single_core_to_PS_COCO, (df_to_process, out_dir, split_by_slide))
        processes.append(p)

    annotations_panoptic = []
    for p in processes:
        annotations_panoptic.extend(p.get())

    return annotations_panoptic
