from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.converters.CCAgT import CCAgT_Annotations
from CCAgT_utils.utils import basename


# Just remove the class, everything can be just functions
class LabelBox_Annotations():

    def __init__(self,
                 raw_labelbox: list[dict[str, Any]],
                 categories_map: list[dict[str, Any]] | None = None) -> None:

        if not isinstance(raw_labelbox, list):
            raise ValueError('Expected a list of dictionary that represents raw labelbox data!')

        expected_data = set({'ID', 'External ID', 'Skipped', 'Reviews', 'Label'})
        for it in raw_labelbox:
            if not all(i in it for i in expected_data):
                if 'Skipped' not in it:
                    raise KeyError(f'Not found expected values need to have `Skipped` or {expected_data}')
        self.raw_labelbox = raw_labelbox[:]

        self.categories_map = None
        if isinstance(categories_map, list):
            self.categories_map = categories_map[:]

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.raw_labelbox)

    def __check_or_instance_categories_map(self,
                                           categories_map: list[dict[str, Any]] | None) -> bool:
        if categories_map is None:
            if self.categories_map is None:
                raise ValueError('You need instantiate or pass as parameter the categories_map before!')
        elif isinstance(categories_map, list):
            if self.categories_map is not None:
                print('The categories map will be overwrite!')
            self.categories_map = categories_map

        if isinstance(self.categories_map, list):
            self.schematic_to_id = {x['labelbox_schemaId']: int(x['id']) for x in self.categories_map}
            return True
        else:
            raise ValueError('Some problems occur in the instantiation of the category map!')

    def __remove_duplicated_labels(self,
                                   df: pd.DataFrame) -> pd.DataFrame:
        duplicated_idx = df['image_name'].duplicated(keep=False)
        df_duplicated = df.loc[duplicated_idx, :]

        if df_duplicated.empty:
            return df

        def hasreview(reviews: list[dict[str, Any]]) -> bool:
            for x in reviews:
                if x['score'] == 1:
                    return True
            else:
                return False

        # Check the labels that has review
        df_duplicated['have_review'] = df_duplicated.apply(lambda row: hasreview(row['Reviews']), axis=1)

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

    def __explode_objects(self,
                          df: pd.DataFrame) -> pd.DataFrame:

        df['objects'] = df.apply(lambda row: row['Label']['objects'], axis=1)
        df = df.explode('objects')
        df = df.reset_index()

        df = df.drop(['index', 'Label'], axis=1)
        return df

    @staticmethod
    def labelbox_to_shapely(object: dict[str, Any]) -> Polygon | Point | np.nan:
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

    def __transform_geometry(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        df['geometry'] = df['objects'].apply(lambda obj: self.labelbox_to_shapely(obj))
        df_out = df.dropna(axis=0, subset=['geometry'])

        if df.shape != df_out.shape:
            print(f'Some NaN geometries have been deleted! Original shape = {df.shape} | out shape = {df_out.shape}')

        if df_out.empty:
            raise ValueError('Data without valid geometries! After transform the geometries the dataframe stay empty.')

        return df_out

    def __prepare_data(self,
                       df: pd.DataFrame) -> pd.DataFrame:
        # Drop ignored images at labelling process
        df = df.drop(df[df['Skipped']].index)

        # Drop irrelevant columns
        df = df.drop(['DataRow ID', 'Labeled Data', 'Created By', 'Project Name', 'Dataset Name', 'Created At', 'Updated At',
                      'Seconds to Label', 'Agreement', 'Benchmark Agreement', 'Benchmark ID', 'View Label',
                      'Has Open Issues', 'Skipped'], axis=1, errors='ignore')

        # Get image names
        df['image_name'] = df.apply(lambda row: basename(row['External ID']), axis=1)
        df = df.drop(['External ID'], axis=1)

        # Remove duplicated labels
        df = self.__remove_duplicated_labels(df)

        # Explode annotations to each row
        df = self.__explode_objects(df)

        # Transform labelbox annotation to a geometry
        df = self.__transform_geometry(df)

        # Map category IDs
        df['category_id'] = df.apply(lambda row: self.schematic_to_id[row['objects']['schemaId']], axis=1)

        df = df.drop(['ID', 'objects', 'Reviews'], axis=1)

        return df

    def to_CCAgT(self,
                 categories_map: list[dict[str, Any]] | None = None) -> CCAgT_Annotations:

        self.__check_or_instance_categories_map(categories_map)

        self.df = self.__prepare_data(self.raw_dataframe)

        CCAgT_anns = CCAgT_Annotations(self.df)
        return CCAgT_anns
