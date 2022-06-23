from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.utils import basename


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
# Functions to
