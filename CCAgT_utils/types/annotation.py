from __future__ import annotations

import collections
from dataclasses import dataclass

from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.types.colors import Color


@dataclass
class Annotation:
    geometry: Polygon | MultiPolygon
    category_id: int
    iscrowd: int = 0
    color: Color | None = None

    @property
    def bbox(self) -> BBox:
        return bounds_to_BBox(self.geometry.bounds, self.category_id)

    @property
    def _geo_type(self) -> str:
        return self.geometry.geom_type

    @property
    def coco_bbox(self) -> list[float | int]:
        bbox = self.bbox
        return [bbox.x_init, bbox.y_init, bbox.width, bbox.height]

    def __iter__(self) -> Annotation:
        self._idx = 0

        if self._geo_type == 'MultiPolygon':
            self._geometries = list(self.geometry.geoms)
        elif self._geo_type == 'Polygon':
            self._geometries = [self.geometry]
        else:
            raise TypeError(f'Unexpected geometry type (`{self._geo_type}`) - expected `MultiPolygon` or `Polygon`')

        return self

    def __next__(self) -> list[Polygon]:
        if self._idx < len(self._geometries):
            out = self._geometries[self._idx]
            self._idx += 1
            return out
        else:
            raise StopIteration


@dataclass
class BBox:
    x_init: int
    y_init: int
    width: int
    height: int
    category_id: int

    @property
    def x_end(self) -> int:
        return self.x_init + self.width

    @property
    def y_end(self) -> int:
        return self.y_init + self.height

    @property
    def upper_left_point(self) -> tuple[int, int]:
        return (self.x_init, self.y_init)

    @property
    def upper_right_point(self) -> tuple[int, int]:
        return (self.x_end, self.y_init)

    @property
    def bottom_right_point(self) -> tuple[int, int]:
        return (self.x_end, self.y_end)

    @property
    def bottom_left_point(self) -> tuple[int, int]:
        return (self.x_init, self.y_end)

    @property
    def coords(self) -> list[tuple[int, int]]:
        return [self.upper_left_point,
                self.upper_right_point,
                self.bottom_right_point,
                self.bottom_left_point]

    @property
    def xy(self) -> tuple[list[int], list[int]]:
        _x, _y = zip(*self.coords)
        return (list(_x), list(_y))

    @property
    def slice_x(self) -> slice:
        return slice(self.x_init, self.x_end)

    @property
    def slice_y(self) -> slice:
        return slice(self.y_init, self.y_end)

    def center_point(self) -> tuple[int, int]:
        return (self.x_init + self.width // 2,
                self.y_init + self.height // 2)

    def area(self) -> int | float:
        return self.width * self.height

    def to_polygon(self) -> Polygon:
        return Polygon(self.coords)

    def fit_inside(self, bounds: tuple[int, int, int, int]) -> None:
        min_x, min_y, max_x, max_y = bounds
        self.x_init = min_x if self.x_init < min_x else self.x_init
        self.y_init = min_y if self.y_init < min_y else self.y_init
        self.width = max_x - self.x_init if self.x_end > max_x else self.width
        self.height = max_y - self.y_init if self.y_end > max_y else self.height

    def add_padding(self, padding: int | float = 0, bounds: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:

        if padding != 0:
            if isinstance(padding, int):
                self.width += padding * 2
                self.height += padding * 2
                self.x_init -= padding
                self.y_init -= padding
            elif isinstance(padding, float):
                self.x_init = int(self.x_init - (self.width * padding))
                self.y_init = int(self.y_init - (self.height * padding))
                self.width = int(self.width * (1 + padding * 2))
                self.height = int(self.height * (1 + padding * 2))
            else:
                raise TypeError('Unexpected value for the padding! Use int or float values')

            self.fit_inside(bounds)


def bounds_to_BBox(bounds: tuple[float], category_id: int) -> BBox:
    b = tuple(int(i) for i in bounds)
    min_x, min_y, max_x, max_y = b
    return BBox(min_x, min_y, max_x - min_x, max_y - min_y, category_id)


def count_BBox_categories(items: list[BBox],
                          categories_infos: CategoriesInfos) -> dict[str, int]:
    c: dict[str, int] = collections.defaultdict(int)
    for bbox in items:
        cat_name = categories_infos[bbox.category_id].name
        c[cat_name] += 1

    return c
