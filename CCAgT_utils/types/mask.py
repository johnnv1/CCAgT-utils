from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.colors as mlp_colors
import numpy as np

from CCAgT_utils.errors import ShapeError
from CCAgT_utils.types.checkers import is_2d
from CCAgT_utils.visualization import colors


@dataclass
class Mask:
    categorical: np.ndarray

    def __post_init__(self):
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
    def unique_ids(self) -> np.ndarray:
        return np.unique(self.categorical)

    def colorized(self,
                  get_color: Callable) -> np.ndarray:
        o = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for id in self.unique_ids:
            o[self.categorical == id] = get_color(id)

        return o

    def cmap(self, get_color: Callable) -> mlp_colors.ListedColormap:
        o = [colors.rgb_to_rgba(get_color(id), normalize=True) for id in self.unique_ids]
        return mlp_colors.ListedColormap(o)
