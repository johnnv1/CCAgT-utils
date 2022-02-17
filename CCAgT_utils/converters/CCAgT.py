from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.geometry import Polygon

from CCAgT_utils.CCAgT import slide_from_filename


class CCAgT_Annotations():
    IMAGE_WIDTH: int = 1600
    IMAGE_HEIGHT: int = 1200

    def __init__(self, df: pd.DataFrame) -> None:

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

    def geometries_type(self) -> pd.Series:
        return self.df['geometry'].apply(lambda x: x.geom_type)

    def geometries_area(self) -> pd.Series:
        return self.df['geometry'].apply(lambda x: x.area)

    def generate_ids(self, col: pd.Series) -> pd.Series:
        col = col.astype('category')
        return col.cat.codes + 1

    def delete_by_area(self,
                       helper: Categories_Helper,
                       ignore_categories: set[int] = set({})) -> pd.DataFrame:
        if 'area' not in self.df.columns:
            self.df['area'] = self.geometries_area()

        categories_at_df = self.df['category_id'].unique()

        for category_id, min_area in helper.min_area_by_category_id.items():
            if category_id in ignore_categories or category_id not in categories_at_df:
                continue

            df_filtered = self.df[self.df['category_id'] == category_id]
            length_before = df_filtered.shape[0]

            cleaned_by_area = df_filtered[df_filtered['area'] >= min_area]

            length_after = cleaned_by_area.shape[0]
            dif = length_before - length_after

            if dif > 0:
                self.df = self.df[self.df['category_id'] != category_id].append(cleaned_by_area)

                print(f'ATTENTION | The category with id {category_id} have been removed {dif} items.')

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
                     np.nan if not geo.intersects(row['geometry']) or row.name == geo_idx
                     else row.name,
                     axis=1).dropna()

        if o.shape[0] == 0:
            return np.nan
        else:
            return o.to_numpy(dtype=np.int16).tolist()

    def find_overlapping_annotations(self,
                                     category_id: int) -> dict[str, list[set[int]]]:
        df = self.df[self.df['category_id'] == category_id]
        df_groupped = df.groupby('image_name')

        out = {}
        for img_name, df_gp in df_groupped:
            intersected_by = df_gp.apply(lambda row: self.find_intersecting_geometries(row['geometry'],
                                                                                       int(row.name),
                                                                                       df_gp), axis=1)

            df_gp['intersected_by'] = intersected_by
            if len(intersected_by) > 0:
                df_gp_exploded = df_gp['intersected_by'].reset_index().explode('intersected_by')
                df_gp_exploded = df_gp_exploded.dropna()

                graph_connections = df_gp_exploded[['index', 'intersected_by']].to_numpy().tolist()

                # Compute the geometries groups
                G = nx.Graph()
                G.add_edges_from(graph_connections)
                coneected_items = list(nx.connected_components(G))

            else:
                coneected_items = []

            out[img_name] = coneected_items
        return out

    def union_geometries(self,
                         groups_by_image: dict[str, list[set[int]]]) -> pd.DataFrame:

        raise NotImplementedError
        # Compute the union of the geometries that have intersection
        for img_name, groups in groups_by_image.items():
            for group in groups:
                # TODO: union geometries and remove original geometry
                pass

    # TODO: Split data into train validation and test

    # TODO: convert to COCO

    # TODO: Describe / stats of the dataset or of the data splitted

    # TODO: Join annotations of overlapped nuclei


class Categories_Helper():

    def __init__(self,
                 raw_helper: list[dict[str, Any]]) -> None:

        if not isinstance(raw_helper, list):
            raise ValueError('Expected a list of dictionary that represents raw helper data!')

        self.raw_helper = raw_helper

    @ property
    def min_area_by_category_id(self) -> dict[int, int]:
        return {int(x['id']): int(x['minimal_area']) for x in self.raw_helper}

    @ property
    def name_by_category_id(self) -> dict[int, str]:
        return {int(x['id']): str(x['name']) for x in self.raw_helper}
