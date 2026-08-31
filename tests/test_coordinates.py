"""Geodetic conversions and the Golden-Morrison rehearsal box."""

import numpy as np

from monarc.common.coordinates import (
    GOLDEN_MORRISON_AOI,
    box_from_center,
    geodetic_to_ecef,
    geodetic_to_enu,
    geodetic_to_utm,
    utm_zone,
)


def test_golden_morrison_is_ten_km_rehearsal_not_county_product():
    aoi = GOLDEN_MORRISON_AOI
    assert aoi.product_boundary == "colorado-state"
    assert aoi.role == "rehearsal_slice"
    assert aoi.size_km_east == 10.0
    assert aoi.size_km_north == 10.0
    assert abs(aoi.center_lat - 39.725) < 1e-9
    assert abs(aoi.center_lon - 105.220) < 1e-9 or abs(aoi.center_lon + 105.220) < 1e-9
    enu_ne = geodetic_to_enu(aoi.north, aoi.east, 0.0, aoi.center_lat, aoi.center_lon)
    span_e = 2.0 * abs(float(enu_ne[0]))
    span_n = 2.0 * abs(float(enu_ne[1]))
    assert 9000 < span_e < 11000
    assert 9000 < span_n < 11000


def test_aoi_intersection():
    aoi = GOLDEN_MORRISON_AOI
    assert aoi.intersects(aoi.bbox)
    assert not aoi.intersects([-110.0, 30.0, -109.0, 31.0])


def test_utm_zone_golden():
    zone, hemi = utm_zone(-105.220, 39.725)
    assert zone == 13
    assert hemi == "N"
    e, n, z, h = geodetic_to_utm(39.725, -105.220)
    assert z == 13
    assert h == "N"
    assert 400000 < e < 600000
    assert 4_000_000 < n < 5_000_000


def test_ecef_finite():
    p = geodetic_to_ecef(39.725, -105.220, 1800.0)
    assert p.shape == (3,)
    assert np.linalg.norm(p) > 6e6
