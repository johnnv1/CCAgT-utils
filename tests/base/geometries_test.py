from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box
from shapely.geometry import Polygon

from CCAgT_utils.base.geometries import clip_to_extent
from CCAgT_utils.base.geometries import point_to_polygon


@pytest.mark.parametrize(
    'area_size,resolution,tolerance', [
        (70, 4, 0.3),
        (120, 4, 0.3),
        (97, 4, 1),
    ],
)
def test_point_to_polygon(
    ccagt_df_multi,
    satellite_ex,
    area_size,
    resolution,
    tolerance,
):
    df = ccagt_df_multi
    sat_series = df.loc[
        (df['category_id'] == 3) & (
            df['geo_type'] == 'Point'
        ), 'geometry',
    ]

    area_size = 70
    resolution = 4
    tolerance = 0.3
    sat_series_pol = point_to_polygon(
        sat_series,
        area_size=area_size,
        resolution=resolution,
        tolerance=tolerance,
    )

    diameter = np.sqrt(area_size / np.pi)
    pol_sat = satellite_ex.buffer(
        diameter,
        resolution,
    ).simplify(tolerance)
    assert sat_series_pol.to_numpy()[0].equals(pol_sat)


@pytest.mark.parametrize(
    'pol,bb,expected', [
        (
            Polygon([(0, 0), (0, 10), (10, 10), (10, 15), (15, 15)]),
            box(0, 0, 10, 10),
            Polygon([(0, 0), (0, 10), (10, 10)]),
        ),
        (
            Polygon([(0, 0), (0, 10), (10, 10), (10, 15), (15, 15)]),
            box(50, 50, 100, 100),
            np.nan,
        ),
        (
            Polygon([(0, 0), (0, 10), (10, 10), (10, 15), (15, 15)]),
            box(0, 0, 100, 100),
            Polygon([(0, 0), (0, 10), (10, 10), (10, 15), (15, 15)]),
        ),
    ],
)
def test_clip_to_extent(pol, bb, expected):
    out = clip_to_extent(pol, bb)
    if isinstance(expected, Polygon):
        assert expected.equals(out)
    else:
        assert np.isnan(expected) and np.isnan(out)
