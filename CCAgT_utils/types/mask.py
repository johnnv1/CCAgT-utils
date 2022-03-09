from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mlp_colors
import numpy as np
from PIL import Image

from CCAgT_utils.errors import ShapeError
from CCAgT_utils.types.checkers import is_2d
from CCAgT_utils.visualization import colors


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
                  get_color: dict[int, list[int]]) -> np.ndarray:
        o = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for id in self.unique_ids:
            o[self.categorical == id] = get_color[id]

        return o

    def cmap(self, get_color: dict[int, list[int] | list[float]]) -> mlp_colors.ListedColormap:
        o = [colors.rgb_to_rgba(get_color[id], normalize=True) for id in self.unique_ids]
        return mlp_colors.ListedColormap(o)

    def save(self,
             filename: str,
             get_color: dict[int, list[int]] | None = None,
             **kwargs: Any) -> None:

        if isinstance(get_color, dict):
            out = self.colorized(get_color)
        else:
            out = self.categorical

        Image.fromarray(out).save(filename, **kwargs)
