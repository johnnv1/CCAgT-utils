from __future__ import annotations

import pandas as pd


class CCAgT_Annotations():
    def __init__(self, df: pd.DataFrame) -> None:

        self.__check_and_instance_dataframe(df)

    def __check_and_instance_dataframe(self, df: pd.DataFrame) -> None:
        need_to_have = {'image_name', 'geometry', 'category_id'}
        if isinstance(df, pd.DataFrame):
            if need_to_have.issubset(set(df.columns)):
                self.df = df
            else:
                raise KeyError(f'The dataframe must have the columns: {need_to_have}')
        else:
            raise TypeError('Need to be a pandas dataframe!')

    # TODO: Get slide ID

    # TODO: Check if all geometries are valid

    # TODO: Calculate the area of each geometry

    # TODO: Remove annotations by the minimal area

    # TODO: Split data into train validation and test

    # TODO: convert to COCO

    # TODO: Describe / stats of the dataset or of the data splitted

    # TODO: Join annotations of overlapped nuclei

    # TODO: Transform satellite to polygon

    # TODO: Generate int ids for images

    # TODO: Generate int ids for annotations
