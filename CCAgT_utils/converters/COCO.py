from __future__ import annotations

from itertools import chain

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon


class COCO_OD():

    @staticmethod
    def bounds_to_coco_bb(bounds: tuple[float],
                          decimals: int = 2) -> list[float]:
        # bounds is in  (minx, miny, maxx, maxy)
        # bb  of coco is in [min(x), min(y), max(x)-min(x), max(y)-min(y)]
        b = tuple(np.round(x, decimals) for x in bounds)
        min_x, min_y, max_x, max_y = b
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    @staticmethod
    def geometry_to_coco_segment(geo: Polygon | MultiPolygon,
                                 decimals: int = 2) -> list[list[float]]:
        # polygon of shapely is a class
        # polygon or segmentation at coco is a list of [[x0, y0, x1, y1 ...]]

        def coco_pol(geometry: Polygon, decimals: int = decimals) -> list[float]:
            list_of_points = list(zip(*geometry.exterior.coords.xy))
            geometry = list(chain(*list_of_points))
            geometry = list(np.around(np.array(geometry), decimals))
            return geometry

        if geo.geom_type == 'Polygon':
            return [coco_pol(geo)]
        elif geo.geom_type == 'MultiPolygon':
            o = []
            for g in geo.geoms:
                o.append(coco_pol(g))
            return o
        else:
            raise TypeError(f'Geometry shape is not a polygon or MultiPolygon. This is a {geo.geom_type}.')
