from __future__ import annotations

from typing import Any

import pandas as pd

from CCAgT_utils.converters.CCAgT import CCAgT_Annotations


class LabelBox_Annotations():

    def __init__(self, raw_labelbox: list[dict[str, Any]]) -> None:
        self.raw_labelbox = raw_labelbox

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.raw_labelbox)

    def __remove_duplicated_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO:
        raise NotImplementedError

    def __explode_objects(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO
        raise NotImplementedError

    def __transform_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: transform to geometry, drop NaN geometries, drop objects
        raise NotImplementedError

    def __prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO
        # Drop ignored images at labelling process
        df = df.drop(df[df['Skipped']].index)

        # Drop irrelevant columns
        df = df.drop(['DataRow ID', 'Labeled Data', 'Created By', 'Project Name', 'Created At', 'Updated At',
                      'Seconds to Label', 'External ID', 'Agreement', 'Benchmark Agreement', 'Benchmark ID', 'View Label',
                      'Has Open Issues'], axis=1)

        # Remove duplicated labels
        df = self.__remove_duplicated_labels(df)

        # Explode annotations to each row
        df = self.__explode_objects(df)

        # Transform labelbox annotation to a geometry
        df = self.__transform_geometry(df)

        # Map category IDs

        # Get image names

        # Generate int ids for images

        # Generate int ids for annotations

        # Transform satellite to polygon

    def to_CCAgT(self) -> CCAgT_Annotations:
        self.df = self.__prepare_data(self.raw_dataframe)

        # TODO: Convert the df to CCAgT dataframe format
        CCAgT_anns = CCAgT_Annotations(self.df)
        return CCAgT_anns
