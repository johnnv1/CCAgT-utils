from __future__ import annotations

import collections
from dataclasses import dataclass

from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon


@dataclass
class Annotation:
    geometry: Polygon | MultiPolygon
    category_id: int

    @property
    def bbox(self) -> BBox:
        return bounds_to_BBox(self.geometry.bounds, self.category_id)

    @property
    def _geo_type(self) -> str:
        return self.geometry.geom_type

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

    def center_point(self) -> tuple[int, int]:
        return (self.x_init + self.width // 2,
                self.y_init + self.height // 2)

    def area(self) -> int | float:
        return self.width * self.height

    def to_polygon(self) -> Polygon:
        return Polygon(self.coords)


def bounds_to_BBox(bounds: tuple[float], category_id: int) -> BBox:
    b = tuple(int(i) for i in bounds)
    min_x, min_y, max_x, max_y = b
    return BBox(min_x, min_y, max_x - min_x, max_y - min_y, category_id)


def count_BBox_categories(items: list[BBox],
                          get_categories_name: dict[int, str]) -> dict[str, int]:
    c: dict[str, int] = collections.defaultdict(int)
    for bbox in items:
        cat_name = get_categories_name[bbox.category_id]
        c[cat_name] += 1

    return c
