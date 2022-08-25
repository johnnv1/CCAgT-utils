from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.geometry import Polygon


def point_to_polygon(
    points_geometries: pd.Series,
    area_size: int = 90,
    resolution: int = 4,
    tolerance: float = 0.3,
) -> pd.Series:
    diameter = np.sqrt(area_size / np.pi)
    return pd.Series(
        points_geometries.apply(
            lambda x: x.buffer(
                distance=diameter,
                resolution=resolution,
            ).simplify(
                tolerance=tolerance,
            ),
        ),
    )


def clip_to_extent(geometry: Polygon, boundary: box) -> Polygon | np.nan:
    """This will clip a geometry (Polygon) to fit inside of the
    a box size. If the geometry is out of the bounds
    will return np.nan.

    Parameters
    ----------
    geometry : Polygon
        A polygon to be clipped
    boundary : box
        A boundary box to be used as extent

    Returns
    -------
    Polygon | np.nan
        A polygon with garante that gave coords that fit inside
        the boundary box. If the geometry is out of the image bounds
        will return np.nan
    """

    if not boundary.contains(geometry):
        geometry = geometry.intersection(boundary)
        if geometry.is_empty:
            return np.nan
    return geometry
