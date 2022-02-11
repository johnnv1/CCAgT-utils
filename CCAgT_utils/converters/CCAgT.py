from __future__ import annotations

import pandas as pd


class CCAgT_Annotations():
    def __init__(self, df: pd.DataFrame | None = None) -> None:

        if df is not None:
            self.__check_and_instance_dataframe(df)

    def __check_and_instance_dataframe(self, df: pd.DataFrame) -> None:
        # TODO
        raise NotImplementedError

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
