from __future__ import annotations

from itertools import chain

import numpy as np
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
    def polygon_to_coco_segment(pol: Polygon,
                                decimals: int = 2) -> list[list[float]]:
        # polygon of shapely is a class
        # polygon or segmentation at coco is a list of [x0, y0, x1, y1 ...]
        list_of_points = list(zip(*pol.exterior.coords.xy))
        pol = list(chain(*list_of_points))
        pol = list(np.around(np.array(pol), decimals))
        return [pol]
