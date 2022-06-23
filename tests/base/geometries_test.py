from __future__ import annotations

import numpy as np
import pytest

from CCAgT_utils.base.geometries import point_to_polygon


@pytest.mark.parametrize(
    'area_size,resolution,tolerance', [
        (70, 4, 0.3),
        (120, 4, 0.3),
        (97, 4, 1),
    ],
)
def test_point_to_polygon(ccagt_df_multi, satellite_ex, area_size, resolution, tolerance):
    df = ccagt_df_multi
    sat_series = df.loc[(df['category_id'] == 3) & (df['geo_type'] == 'Point'), 'geometry']

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
    pol_sat = satellite_ex.buffer(diameter, resolution).simplify(tolerance)
    assert sat_series_pol.to_numpy()[0].equals(pol_sat)
