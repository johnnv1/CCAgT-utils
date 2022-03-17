from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mlp_colors
import numpy as np
from PIL import Image

from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.errors import ShapeError
from CCAgT_utils.types.checkers import is_2d


@dataclass
class Mask:
    # FIXME: typing also can be a list[list[int]]
    categorical: np.ndarray

    def __post_init__(self) -> None:
        self.categorical: np.ndarray = np.array(self.categorical, dtype=np.uint8)

        if not is_2d(self.categorical.shape):
            raise ShapeError('Unexpected shape, categorical mask need to be a matrix (2D array)!')

    @property
    def height(self) -> int:
        return int(self.categorical.shape[0])

    @property
    def width(self) -> int:
        return int(self.categorical.shape[1])

    @property
    def unique_ids(self) -> set[int]:
        return set(np.unique(self.categorical))

    def colorized(self,
                  categories_infos: CategoriesInfos) -> np.ndarray:
        o = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for id in self.unique_ids:
            o[self.categorical == id] = categories_infos[id].color.rgb

        return o

    def cmap(self,
             categories_infos: CategoriesInfos
             ) -> mlp_colors.ListedColormap:
        o = [categories_infos[id].color.rgba_normalized for id in self.unique_ids]
        return mlp_colors.ListedColormap(o)

    def save(self,
             filename: str,
             categories_infos: CategoriesInfos | None = None,
             **kwargs: Any) -> None:

        if isinstance(categories_infos, CategoriesInfos):
            out = self.colorized(categories_infos)
        else:
            out = self.categorical

        Image.fromarray(out).save(filename, **kwargs)
