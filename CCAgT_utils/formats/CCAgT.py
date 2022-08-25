from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing_extensions import Literal

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.base.errors import MoreThanOneIDbyItemError
from CCAgT_utils.base.geometries import clip_to_extent
from CCAgT_utils.base.utils import slide_from_filename
from CCAgT_utils.formats.annotation import bounds_to_BBox


class CCAgT(pd.DataFrame):
    _metadata = ['geometry']

    @property
    def _constructor(self) -> type[CCAgT]:
        return CCAgT


def load(
    filename: str,
    **kwargs: Any
) -> CCAgT:
    if not filename.endswith(('.parquet.gzip', '.parquet')):
        raise FileTypeError('The labels file is not a parquet file.')

    df = pd.read_parquet(filename, **kwargs)

    if 'geometry' in df.columns:
        # buffer(0) applied to fix invalid geomtries. From shapely issue #278
        df['geometry'] = df['geometry'].apply(lambda x: shapely.wkt.loads(x).buffer(0))

    return df


def save(
        ccagt_df: CCAgT,
        filename: str,
        compression: Literal['snappy', 'gzip', 'brotli'] | None = 'gzip',
        **kwargs: Any
) -> None:
    df_out = ccagt_df.copy()
    df_out['geometry'] = df_out['geometry'].apply(lambda x: x.wkt)
    df_out.to_parquet(filename, compression=compression, **kwargs)


# -----------------------------------------------------------------------
# Functions to generate or update a column (return a Serie)

def slides_ids(
    ccagt_df: CCAgT,
) -> pd.Series:
    return pd.Series(ccagt_df['image_name'].apply(lambda x: slide_from_filename(x)))


def geometries_type(
        ccagt_df: CCAgT,
) -> pd.Series:
    return pd.Series(ccagt_df['geometry'].apply(lambda x: x.geom_type))


def geometries_area(
        ccagt_df: CCAgT,
) -> pd.Series:
    return pd.Series(ccagt_df['geometry'].apply(lambda x: x.area))


def fit_geometries_to_boundary(
    ccagt_df: CCAgT,
    width: int,
    height: int,
) -> pd.Series:
    """This will fix the geometries that does't fit inside of a boundary
    box.

    Will check all geometries if the coords of the geometry fit inside
    the a boundary box. If does't fit, will compute the intersection
    between the polygon and the  boundary box (this will clip the polygon
    to fit inside the boundary box); If this intersection return a empty
    geometry, so will return a np.nan in the local of the geometry.

    Parameters
    ----------
    width : int
        Width of the boundary box
    height : int
        Height of the boundary box

    Returns
    -------
    pd.Series
        This will return the geometry column as a pandas Series, with
        the geometries fixed to fit in the images bounds.
    """

    boundary_box = box(0, 0, width, height)
    return pd.Series(ccagt_df['geometry'].apply(lambda x: clip_to_extent(x, boundary_box)))


def generate_ids(
    column: pd.Series,
) -> pd.Series:
    column = column.astype('category')
    return column.cat.codes + 1


# -----------------------------------------------------------------------
# Functions to modify the dataframe

def delete_by_area(
    ccagt_df: CCAgT,
    categories_infos: CategoriesInfos,
    categories_to_ignore: set[int] = set({}),
) -> CCAgT:
    if 'area' not in ccagt_df.columns:
        ccagt_df['area'] = geometries_area(ccagt_df)

    categories_at_df = ccagt_df['category_id'].unique()

    for cat_info in categories_infos:
        if cat_info.id in categories_to_ignore or cat_info.id not in categories_at_df:
            continue

        df_filtered = ccagt_df[ccagt_df['category_id'] == cat_info.id]
        length_before = df_filtered.shape[0]

        cleaned_by_area = df_filtered[df_filtered['area'] >= cat_info.minimal_area]

        length_after = cleaned_by_area.shape[0]
        dif = length_before - length_after

        if dif > 0:
            _df = ccagt_df[ccagt_df['category_id'] != cat_info.id]
            ccagt_df = CCAgT(pd.concat([_df, cleaned_by_area]))

            print(f'ATTENTION | {dif} items has been removed from category with id {cat_info.id}')

    return ccagt_df


def filter_by_categories(
    ccagt_df: CCAgT,
    categories_id: set[int],
) -> CCAgT:
    return CCAgT(ccagt_df[ccagt_df['category_id'].isin(categories_id)])


def union_geometries(
    ccagt_df: CCAgT,
    groups_by_image: dict[str, list[set[int]]],
    out_category_id: int | None = None,
) -> CCAgT:
    """Based on a dict for each image, it will join the geometries, so
    that each image can have a list of geometries groups. Each geometries
    group will be merged to a single geometry.

    Attention: This function will reset the indexes numbers.

    Parameters
    ----------
    groups_by_image : dict[str, list[set[int]]]
        The keys need to match with the names that the column has
        (`image_name`), and the list need to have groups (set) of indexes
        that must be merged into a single geometry

    out_category_id : int | None, optional
        If will desired to join multiple categories into one, and create
        a "new" category, specify what will be the category id for the
        outputs geometries, by default None. If this is None, the
        original category will be used.

    Returns
    -------
    pd.DataFrame
        Will return the self.df after the modifications. So the old
        geometries (before be merged) will be discarded and the new
        geometries will be add. Also, the DataFrame indexes will be
        reseted.

    Raises
    ------
    ValueError
        If the `out_category_id` is None and the group of
        geometries have more than one category, will raise it
    TypeError
        If the type of the geometry after the union isn't equals a
        `Polygon` will raise it
    """

    def __check_category_id(categories_ids: list[int]) -> int:
        if len(categories_ids) > 1:
            raise ValueError(f'The group of annotations <{group}> from {img_name} have more than one category\
                        need to set the out_category_id parameter!')
        else:
            return categories_ids[0]

    cat_id = out_category_id
    out = pd.DataFrame()
    for img_name, groups in groups_by_image.items():
        df_filtered = ccagt_df[ccagt_df['image_name'] == img_name]

        for group in groups:
            df_filtered_by_group = df_filtered[df_filtered.index.isin(group)]

            if df_filtered_by_group.empty:
                continue

            if out_category_id is None:
                cat_id = __check_category_id(df_filtered_by_group['category_id'].unique().tolist())

            geometries_to_join = df_filtered_by_group['geometry'].to_numpy()

            # To make sure it will intersect more than just touch bounds
            geometries_to_join = [g.buffer(1) for g in geometries_to_join]

            geo = unary_union(geometries_to_join).simplify(tolerance=0.3)

            df_filtered = df_filtered.drop(index=group, axis=0, errors='ignore')
            df_filtered = pd.concat([
                df_filtered, pd.DataFrame([{
                    'image_name': img_name,
                    'geometry': geo,
                    'category_id': cat_id,
                }]),
            ])

        out = pd.concat([out, df_filtered])

    # Drop the annotation of all images that stay at the parameter
    image_names_to_drop = list(groups_by_image.keys())
    df_without_news = ccagt_df[~ccagt_df['image_name'].isin(image_names_to_drop)]

    # Add the annotations that have been updates!
    ccagt_df = CCAgT(pd.concat([df_without_news, out], ignore_index=True))

    return ccagt_df


# -----------------------------------------------------------------------
# Utils functions

def find_intersecting_geometries(
    ccagt_df: CCAgT,
    geometry: Polygon,
    geo_idx: int,
) -> list[int] | np.nan:
    """Based on a geometry, search in the dataframe for others
    geometries that intersect with the geometry passed as parameter.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with all data of the same image of the geometry
        passed as parameter.
    geometry : Polygon
        A shapely polygon used to check with which others geometries has
        intersection.
    geo_idx : int
        The id (row.name) of the geometry

    Returns
    -------
    list[int]
        A list with the indexes of each geometry that intersects with the
        geometry passed as parameter
    """
    o = ccagt_df.apply(
        lambda row:
        np.nan if not row['geometry'].is_valid else
        np.nan if not geometry.intersects(row['geometry']) or row.name == geo_idx
        else row.name,
        axis=1,
    ).dropna()

    if o.shape[0] == 0:
        return np.nan
    else:
        return o.to_numpy(dtype=np.int64).tolist()


def has_intersecting_geometries(
    geo: Polygon,
    geometries: pd.Series,
) -> bool:
    for _, g in geometries.iteritems():
        if geo.intersects(g):
            return True
    return False


def find_overlapping_annotations(
    ccagt_df: CCAgT,
    categories_id: set[int],
    by_bbox: bool = False,
) -> dict[str, list[set[int]]]:
    df_out = filter_by_categories(ccagt_df, categories_id).copy()

    df_groupped = df_out.groupby('image_name')

    if by_bbox:
        df_out['geometry'] = df_out.apply(
            lambda row: bounds_to_BBox(
                row['geometry'].bounds,
                row['category_id'],
            ).to_polygon(), axis=1,
        )

    out = {}
    for img_name, df_gp in df_groupped:
        intersected_by = df_gp.apply(
            lambda row: find_intersecting_geometries(
                df_gp,
                row['geometry'],
                int(row.name),
            ), axis=1,
        ).dropna()

        df_gp['intersected_by'] = intersected_by
        if len(intersected_by) > 0:
            df_gp_exploded = df_gp['intersected_by'].reset_index().explode('intersected_by')
            df_gp_exploded = df_gp_exploded.dropna()

            graph_connections = df_gp_exploded[['index', 'intersected_by']].to_numpy().tolist()

            # Compute the geometries groups
            G = nx.Graph()
            G.add_edges_from(graph_connections)
            connected_items = list(nx.connected_components(G))

            out[img_name] = connected_items
    return out


def image_id_by_name(
    ccagt_df: CCAgT,
    image_name: str,
) -> int:
    v = ccagt_df.loc[ccagt_df['image_name'] == image_name, 'image_id'].unique()
    if len(v) > 1:
        raise MoreThanOneIDbyItemError(f'The image: {image_name} have {len(v)} IDs')
    else:
        return int(v[0])


# -----------------------------------------------------------------------
# Functions which generate a new/copy of the dataframe

def verify_if_intersects(
    ccagt_df: CCAgT,
    base_categories_id: set[int],
    target_categories_id: set[int] | None,
) -> pd.DataFrame:
    df_base = filter_by_categories(ccagt_df, base_categories_id)

    if target_categories_id is None:
        df_target = df_base.copy()
    else:
        df_target = filter_by_categories(ccagt_df, target_categories_id)

    df_base_groupped_by_image = df_base.groupby('image_name')
    out = pd.DataFrame()
    for img_name, df_base_by_image in df_base_groupped_by_image:
        df_target_geos = df_target.loc[df_target['image_name'] == img_name, 'geometry']

        df_base_by_image['has_intersecting'] = df_base_by_image.apply(
            lambda row: has_intersecting_geometries(row['geometry'], df_target_geos),
            axis=1,
        )

        out = pd.concat([out, df_base_by_image])

    return out
