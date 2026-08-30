"""WGS84, UTM, and local ENU conversions plus the Golden-Morrison rehearsal AOI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Product rehearsal slice. v1 coverage remains Colorado-the-state (docs/cost.md).
GOLDEN_MORRISON_CENTER_LAT = 39.725
GOLDEN_MORRISON_CENTER_LON = -105.220
GOLDEN_MORRISON_SIZE_KM = 10.0


@dataclass(frozen=True)
class AoiBox:
    """Axis-aligned WGS84 box. Coordinates are west, south, east, north."""

    west: float
    south: float
    east: float
    north: float
    center_lat: float
    center_lon: float
    size_km_north: float
    size_km_east: float
    name: str = ""
    role: str = ""
    product_boundary: str = "colorado-state"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def intersects(self, other_bbox: Sequence[float]) -> bool:
        w, s, e, n = [float(x) for x in other_bbox]
        return not (e < self.west or w > self.east or n < self.south or s > self.north)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "product_boundary": self.product_boundary,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "size_km": [self.size_km_east, self.size_km_north],
            "bbox_wgs84": list(self.bbox),
        }


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray = 0.0) -> np.ndarray:
    """Convert geodetic WGS84 to ECEF meters. Returns (..., 3)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(h_m, dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + h) * cos_lat * np.cos(lon)
    y = (n + h) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Row-orthonormal ECEF->ENU rotation at a reference geodetic point."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    h_m: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    origin_h: float = 0.0,
) -> np.ndarray:
    """Local east-north-up meters relative to an origin."""
    p = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    p0 = geodetic_to_ecef(origin_lat, origin_lon, origin_h).reshape(3)
    R = ecef_to_enu_matrix(origin_lat, origin_lon)
    return (p - p0) @ R.T


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Approximate WGS84 meters per degree of latitude and longitude."""
    lat = np.deg2rad(lat_deg)
    m_per_deg_lat = (np.pi / 180.0) * WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * np.sin(lat) ** 2) ** 1.5
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    m_per_deg_lon = (np.pi / 180.0) * n * np.cos(lat)
    return float(m_per_deg_lat), float(m_per_deg_lon)


def box_from_center(
    lat: float,
    lon: float,
    size_km: float = GOLDEN_MORRISON_SIZE_KM,
    *,
    name: str = "",
    role: str = "",
) -> AoiBox:
    """Build a square geodetic box of ``size_km`` on each side around a center."""
    m_lat, m_lon = meters_per_degree(lat)
    half_m = 0.5 * size_km * 1000.0
    dlat = half_m / m_lat
    dlon = half_m / m_lon
    return AoiBox(
        west=lon - dlon,
        south=lat - dlat,
        east=lon + dlon,
        north=lat + dlat,
        center_lat=lat,
        center_lon=lon,
        size_km_north=size_km,
        size_km_east=size_km,
        name=name,
        role=role,
    )


def utm_zone(lon_deg: float, lat_deg: float) -> tuple[int, str]:
    """UTM zone number and hemisphere letter (N/S) from WGS84."""
    zone = int((float(lon_deg) + 180.0) // 6) + 1
    hemi = "N" if lat_deg >= 0.0 else "S"
    return zone, hemi


def geodetic_to_utm(lat_deg: float, lon_deg: float) -> tuple[float, float, int, str]:
    """WGS84 to UTM easting/northing (meters) using the standard series expansion."""
    zone, hemi = utm_zone(lon_deg, lat_deg)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lon0 = np.deg2rad((zone - 1) * 6 - 180 + 3)
    k0 = 0.9996
    e2 = WGS84_E2
    ep2 = e2 / (1.0 - e2)
    n = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    t = np.tan(lat)
    c = ep2 * np.cos(lat) ** 2
    a = np.cos(lat) * (lon - lon0)
    e4 = e2 * e2
    e6 = e4 * e2
    m = WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * np.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * np.sin(4.0 * lat)
        - (35.0 * e6 / 3072.0) * np.sin(6.0 * lat)
    )
    easting = (
        k0
        * n
        * (
            a
            + (1.0 - t**2 + c) * a**3 / 6.0
            + (5.0 - 18.0 * t**2 + t**4 + 72.0 * c - 58.0 * ep2) * a**5 / 120.0
        )
        + 500000.0
    )
    northing = k0 * (
        m
        + n
        * t
        * (
            a**2 / 2.0
            + (5.0 - t**2 + 9.0 * c + 4.0 * c**2) * a**4 / 24.0
            + (61.0 - 58.0 * t**2 + t**4 + 600.0 * c - 330.0 * ep2) * a**6 / 720.0
        )
    )
    if hemi == "S":
        northing += 10000000.0
    return float(easting), float(northing), zone, hemi


GOLDEN_MORRISON_AOI = box_from_center(
    GOLDEN_MORRISON_CENTER_LAT,
    GOLDEN_MORRISON_CENTER_LON,
    GOLDEN_MORRISON_SIZE_KM,
    name="golden-morrison-rehearsal",
    role="rehearsal_slice",
)
