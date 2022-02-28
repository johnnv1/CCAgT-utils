from __future__ import annotations

import multiprocessing
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.ops import unary_union

from CCAgT_utils.CCAgT import slide_from_filename
from CCAgT_utils.converters.COCO import COCO_OD
from CCAgT_utils.errors import MoreThanOneIDbyItemError
from CCAgT_utils.types.annotation import bounds_to_BBox
from CCAgT_utils.utils import get_traceback


class CCAgT_Annotations():
    IMAGE_WIDTH: int = 1600
    IMAGE_HEIGHT: int = 1200

    def __init__(self,
                 df: pd.DataFrame) -> None:

        self.__check_and_instance_dataframe(df)

    def __check_and_instance_dataframe(self,
                                       df: pd.DataFrame) -> None:
        need_to_have = {'image_name', 'geometry', 'category_id'}
        if isinstance(df, pd.DataFrame):
            if need_to_have.issubset(set(df.columns)):
                # TODO: check the type of each col : str, Shapely (Point | Polygon), int
                self.df = df
            else:
                raise KeyError(f'The dataframe must have the columns: {need_to_have}')
        else:
            raise TypeError('Need to be a pandas dataframe!')

    def get_slide_id(self) -> pd.Series:
        return self.df['image_name'].apply(lambda x: slide_from_filename(x))

    def geometries_type(self) -> pd.Series:
        return self.df['geometry'].apply(lambda x: x.geom_type)

    def satellite_point_to_polygon(self,
                                   satellite_geo: pd.Series,
                                   area_size: int = 90,
                                   resolution: int = 4,
                                   tolerance: float = 0.3) -> pd.Series:
        diameter = np.sqrt(area_size / np.pi)
        return satellite_geo.apply(lambda x: x.buffer(distance=diameter, resolution=resolution).simplify(tolerance=tolerance))

    def fit_geometries_to_image_boundary(self,
                                         width: int | None = None,
                                         height: int | None = None
                                         ) -> pd.Series:
        """This method will fix the geometries that does't fit inside
        of the image size.

        Will check all geometries if the coords of the geometry fit
        inside the image boundary. If does't fit, will compute the
        intersection between the polygon and the image bounds (this will
        clip the the polygon to fit inside the image); If this
        intersection return a empty geometry, so will return a np.nan
        in the local of the geometry.

        Parameters
        ----------
        width : int, optional
            By default is None, and the self.IMAGE_WIDTH will be used;
            Otherwise will considers the width passed as parameter.
        height : int, optional
            By default is None, and the self.IMAGE_HEIGHT will be used;
            Otherwise will considers the height passed as parameter.

        Returns
        -------
        pd.Series
            This will return the geometry collum as a pandas Series, with
            the geometries fixed to fit the images bounds.
        """
        w = self.IMAGE_WIDTH if width is None else width
        h = self.IMAGE_HEIGHT if height is None else height
        image_box = box(0, 0, w, h)

        def clip_to_extent(geometry: Polygon, image_box: box = image_box) -> Polygon | np.nan:
            """This will clip a geometry (Polygon) to fit inside of the
            image size. If the geometry is out of the image bounds
            will return np.nan.
            Returns
            -------
            Polygon | np.nan
                A polygon with garante that gave coords that fit inside
                the image size (self.IMAGE_WIDTH, self.IMAGE_HEIGHT).
                If the geometry is out of the image bounds
                will return np.nan
            """
            if not image_box.contains(geometry):
                geometry = geometry.intersection(image_box)
                if geometry.is_empty:
                    return np.nan
            return geometry

        return self.df['geometry'].apply(lambda x: clip_to_extent(x))

    def geometries_area(self) -> pd.Series:
        return self.df['geometry'].apply(lambda x: x.area)

    def generate_ids(self,
                     col: pd.Series) -> pd.Series:
        col = col.astype('category')
        return col.cat.codes + 1

    def delete_by_area(self,
                       minimal_area: dict[int, int],
                       ignore_categories: set[int] = set({})) -> pd.DataFrame:
        if 'area' not in self.df.columns:
            self.df['area'] = self.geometries_area()

        categories_at_df = self.df['category_id'].unique()

        for category_id, min_area in minimal_area.items():
            if category_id in ignore_categories or category_id not in categories_at_df:
                continue

            df_filtered = self.df[self.df['category_id'] == category_id]
            length_before = df_filtered.shape[0]

            cleaned_by_area = df_filtered[df_filtered['area'] >= min_area]

            length_after = cleaned_by_area.shape[0]
            dif = length_before - length_after

            if dif > 0:
                self.df = self.df[self.df['category_id'] != category_id].append(cleaned_by_area)

                print(f'ATTENTION | {dif} items has been removed from category with id {category_id}')

        return self.df

    @staticmethod
    def find_intersecting_geometries(geo: Polygon,
                                     geo_idx: int,
                                     df: pd.DataFrame) -> list[int] | np.nan:
        """Based on a geometry, search in the dataframe for others
        geometries that intersect with the geometry passed as parameter.

        Parameters
        ----------
        geo : Polygon
            A shapely polygon used to check with which others geometries
            has intersection.
        df : pd.DataFrame
            A dataframe with the data for the same tile/image of the
            geometry passed as parameter.

        Returns
        -------
        list[int]
            A list with the indexes of each geometry that intersects with
            the geometry passed as parameter
        """
        o = df.apply(lambda row:
                     np.nan if not row['geometry'].is_valid else
                     np.nan if not geo.intersects(row['geometry']) or row.name == geo_idx
                     else row.name,
                     axis=1).dropna()

        if o.shape[0] == 0:
            return np.nan
        else:
            return o.to_numpy(dtype=np.int64).tolist()

    def filter_by_category(self, categories_id: set[int]) -> pd.DataFrame:
        return self.df[self.df['category_id'].isin(categories_id)]

    @staticmethod
    def has_intersecting_geometries(geo: Polygon,
                                    geometries: pd.Series) -> bool:
        for _, g in geometries.iteritems():
            if geo.intersects(g):
                return True
        return False

    def verify_if_intersects(self,
                             base_categories_id: set[int],
                             target_categories_id: set[int] | None) -> pd.DataFrame:
        df_base = self.filter_by_category(base_categories_id)

        if target_categories_id is None:
            df_target = df_base.copy()
        else:
            df_target = self.filter_by_category(target_categories_id)

        df_base_groupped_by_image = df_base.groupby('image_name')
        out = pd.DataFrame()
        for img_name, df_base_by_image in df_base_groupped_by_image:
            df_target_geos = df_target.loc[df_target['image_name'] == img_name, 'geometry']

            df_base_by_image['has_intersecting'] = df_base_by_image.apply(
                lambda row: self.has_intersecting_geometries(row['geometry'], df_target_geos),
                axis=1
            )

            out = pd.concat([out, df_base_by_image])

        return out

    def find_overlapping_annotations(self,
                                     categories_id: set[int],
                                     by_bbox: bool = False) -> dict[str, list[set[int]]]:
        df = self.df[self.df['category_id'].isin(categories_id)].copy()

        df_groupped = df.groupby('image_name')

        if by_bbox:
            df['geometry'] = df.apply(lambda row: bounds_to_BBox(row['geometry'].bounds,
                                                                 row['category_id']).to_polygon(), axis=1)

        out = {}
        for img_name, df_gp in df_groupped:
            intersected_by = df_gp.apply(lambda row: self.find_intersecting_geometries(row['geometry'],
                                                                                       int(row.name),
                                                                                       df_gp), axis=1).dropna()

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

    def union_geometries(self,
                         groups_by_image: dict[str, list[set[int]]],
                         out_category_id: int | None = None
                         ) -> pd.DataFrame:
        """Based on a dict for each image, it will join the geometries,
        so that each image can have a list of geometries groups. Each
        geometries group will be merged to a single geometry.

        Attention: This function will reset the indexes numbers.

        Parameters
        ----------
        groups_by_image : dict[str, list[set[int]]]
            The keys need to match with the names that the column has
            (`image_name`), and the list need to have groups (set) of
            indexes that must be merged into a single geometry
        out_category_id : int | None, optional
            If will dessired to join multiple categories into one, and
            create a "new" category, specify what will be the category id
            for the outputs geometries, by default None.
            If this is None, the original category will be used.

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
            df_filtered = self.df[self.df['image_name'] == img_name]

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
                df_filtered = pd.concat([df_filtered, pd.DataFrame([{'image_name': img_name,
                                                                     'geometry': geo,
                                                                     'category_id': cat_id}])])

            out = pd.concat([out, df_filtered])

        # Drop the annotation of all images that stay at the parameter
        image_names_to_drop = list(groups_by_image.keys())
        df_without_news = self.df[~self.df['image_name'].isin(image_names_to_drop)]

        # Add the annotations that have been updates!
        self.df = df_without_news.append(out).reset_index(drop=True)

        return self.df

    def to_OD_COCO(self, decimals: int = 2) -> list[dict[str, Any]]:

        cols = self.df.columns
        if not all(c in cols for c in ['area', 'image_id']):
            raise KeyError('The dataframe need to have the columns `area`, `image_id`!')

        cpu_num = multiprocessing.cpu_count()

        # To ensure that will have sequential index
        self.df.reset_index(drop=True, inplace=True)
        self.df.index = self.df.index + 1

        # Split equals the annotations for the cpu quantity
        ann_ids_splitted = np.array_split(self.df.index.tolist(), cpu_num)
        print(f'Number of cores: {cpu_num}, annotations per core: {len(ann_ids_splitted[0])}')

        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for ann_ids in ann_ids_splitted:
            df_to_process = self.df.loc[self.df.index.isin(ann_ids), :]
            p = workers.apply_async(single_core_to_OD_COCO, (df_to_process, decimals))
            processes.append(p)

        annotations_coco = []
        for p in processes:
            annotations_coco.extend(p.get())

        return annotations_coco

    def image_id_by_name(self, image_name: str) -> int:
        v = self.df.loc[self.df['image_name'] == image_name, 'image_id'].unique()
        if len(v) > 1:
            raise MoreThanOneIDbyItemError(f'The image: {image_name} have {len(v)} IDs')
        else:
            return int(v[0])

    def to_parquet(self, filename: str, compression: str = 'gzip', **kwargs: Any) -> None:
        df_out = self.df.copy()
        df_out['geometry'] = df_out['geometry'].apply(lambda x: x.wkt)

        df_out.to_parquet(filename, compression=compression, **kwargs)

    # TODO: Split data into train validation and test

    # TODO: Describe / stats of the dataset or of the data splitted


@get_traceback
def single_core_to_OD_COCO(df: pd.DataFrame, decimals: int = 2) -> list[dict[str, Any]]:
    return df.apply(lambda row: {'id': row.name,
                                 'image_id': row['image_id'],
                                 'category_id': row['category_id'],
                                 'bbox': COCO_OD.bounds_to_coco_bb(row['geometry'].bounds, decimals),
                                 'segmentation': COCO_OD.geometry_to_coco_segment(row['geometry'], decimals),
                                 'area': np.round(row['area'], decimals),
                                 'iscrowd': 0},
                    axis=1).to_numpy().tolist()


def read_parquet(filename: str, **kwargs: Any) -> CCAgT_Annotations:
    df = pd.read_parquet(filename, **kwargs)
    df['geometry'] = df['geometry'].apply(lambda x: shapely.wkt.loads(x))

    return CCAgT_Annotations(df)
