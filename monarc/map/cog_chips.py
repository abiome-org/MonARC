"""Plan and materialize range-read COG chips. Never copies a full GeoTIFF to R2."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from monarc.common.coordinates import AoiBox, geodetic_to_enu, geodetic_to_utm, meters_per_degree
from monarc.data.pc_sas import apply_sas_token, is_azure_blob_href, token_expired, unsigned_href

JsonObj = dict[str, Any]
WindowReader = Callable[[str, int, int, int, int], np.ndarray]

DEFAULT_CHIP_SIZE_PX = 224
DEFAULT_CHIP_GRID = 8
DEFAULT_MAX_CHIPS = 64


class ChipExtractError(RuntimeError):
    """Raised when a chip window cannot be planned or materialised."""


def parse_affine(transform: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    """STAC ``proj:transform`` as affine (a, b, c, d, e, f)."""
    vals = [float(x) for x in transform]
    if len(vals) >= 9:
        return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
    if len(vals) == 6:
        return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
    raise ChipExtractError(f"proj:transform length {len(vals)} is not 6 or 9")


def _invert_north_up(a: float, c: float, e: float, f: float, x: float, y: float) -> tuple[float, float]:
    if a == 0.0 or e == 0.0:
        raise ChipExtractError("non-invertible affine")
    col = (x - c) / a
    row = (y - f) / e
    return col, row


def covering_item(items: Sequence[JsonObj], lon: float, lat: float) -> JsonObj | None:
    for item in items:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        w, s, e, n = [float(x) for x in bbox]
        if w <= lon <= e and s <= lat <= n:
            return item
    return None


def pixel_window_for_point(
    item: JsonObj,
    lat: float,
    lon: float,
    size_px: int,
) -> dict[str, int] | None:
    """Pixel window centred on ``(lat, lon)`` using STAC affine metadata."""
    props = item.get("properties") or {}
    transform = props.get("proj:transform") or item.get("proj:transform")
    shape = props.get("proj:shape") or item.get("proj:shape")
    epsg = props.get("proj:epsg") or item.get("proj:epsg")
    if transform is None or shape is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    a, _b, c, _d, e, f = parse_affine(transform)
    easting, northing, zone, hemi = geodetic_to_utm(lat, lon)
    if epsg in (26913, 32613) and (zone != 13 or hemi != "N"):
        return None
    col_c, row_c = _invert_north_up(a, c, e, f, easting, northing)
    half = int(size_px) // 2
    col0 = int(round(col_c)) - half
    row0 = int(round(row_c)) - half
    col0 = max(0, min(col0, max(0, width - size_px)))
    row0 = max(0, min(row0, max(0, height - size_px)))
    return {
        "col_off": col0,
        "row_off": row0,
        "width": int(size_px),
        "height": int(size_px),
        "utm_zone": int(zone),
        "utm_easting": float(easting),
        "utm_northing": float(northing),
    }


def _item_href(item: JsonObj) -> str:
    if item.get("catalog_href"):
        return unsigned_href(str(item["catalog_href"]))
    if item.get("signed_href"):
        return unsigned_href(str(item["signed_href"]))
    assets = item.get("assets") or {}
    image = assets.get("image") or assets.get("visual") or assets.get("data") or {}
    return unsigned_href(str(image.get("href") or ""))


def _item_id(item: JsonObj) -> str:
    return str(item.get("id") or "")


def plan_chip_windows(
    aoi: AoiBox,
    items: Sequence[JsonObj],
    *,
    size_px: int = DEFAULT_CHIP_SIZE_PX,
    grid: int = DEFAULT_CHIP_GRID,
    max_chips: int = DEFAULT_MAX_CHIPS,
    overlap_frac: float = 0.0,
    gsd_m: float = 0.3,
) -> list[JsonObj]:
    """Sample a regular grid of chip windows over the AOI. One vintage of items."""
    if size_px <= 0 or grid <= 0 or max_chips <= 0:
        raise ChipExtractError("chip size, grid, and max_chips must be positive")
    if not 0.0 <= float(overlap_frac) < 1.0:
        raise ChipExtractError("overlap_frac must be in [0, 1)")
    if not np.isfinite(gsd_m) or float(gsd_m) <= 0.0:
        raise ChipExtractError("gsd_m must be positive")
    windows: list[JsonObj] = []
    chip_size_m = float(size_px) * float(gsd_m)
    stride_m = chip_size_m * (1.0 - float(overlap_frac))
    if overlap_frac > 0.0:
        m_lat, m_lon = meters_per_degree(aoi.center_lat)
        lat_step, lon_step = stride_m / m_lat, stride_m / m_lon
        latitudes = np.arange(aoi.south + 0.5 * chip_size_m / m_lat,
                              aoi.north, lat_step)
        longitudes = np.arange(aoi.west + 0.5 * chip_size_m / m_lon,
                               aoi.east, lon_step)
    else:
        n = int(grid)
        latitudes = [aoi.south + (aoi.north - aoi.south) * (i + 0.5) / n for i in range(n)]
        longitudes = [aoi.west + (aoi.east - aoi.west) * (j + 0.5) / n for j in range(n)]
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            if not aoi.intersects([lon, lat, lon, lat]):
                continue
            item = covering_item(items, lon, lat)
            if item is None:
                continue
            href = _item_href(item)
            pix = pixel_window_for_point(item, lat, lon, size_px) or {}
            xyz = geodetic_to_enu(lat, lon, 0.0, aoi.center_lat, aoi.center_lon)
            rec: JsonObj = {
                "id": f"chip-{i:02d}-{j:02d}",
                "item_id": _item_id(item),
                "catalog_href": href,
                "signed_href": item.get("signed_href"),
                "sas": item.get("sas"),
                "sas_expiry": item.get("sas_expiry"),
                "lat": float(lat),
                "lon": float(lon),
                "xyz": [float(xyz[0]), float(xyz[1]), float("nan")],
                "size_px": int(size_px),
                "range_read": True,
                "copy_full_geotiff": False,
                "chip_size_m": chip_size_m,
                "stride_m": stride_m,
                "overlap_frac": float(overlap_frac),
            }
            rec.update(pix)
            windows.append(rec)
            if len(windows) >= int(max_chips):
                return windows
    return windows


def chip_plan_block(
    windows: Sequence[JsonObj],
    *,
    size_px: int,
    grid: int,
    max_chips: int,
    overlap_frac: float = 0.0,
    gsd_m: float = 0.3,
) -> JsonObj:
    chip_size_m = float(size_px) * float(gsd_m)
    return {
        "size_px": int(size_px),
        "grid": int(grid),
        "max_chips": int(max_chips),
        "gsd_m": float(gsd_m),
        "chip_size_m": chip_size_m,
        "stride_m": chip_size_m * (1.0 - float(overlap_frac)),
        "overlap_frac": float(overlap_frac),
        "n_windows": len(windows),
        "range_read": True,
        "copy_full_geotiff": False,
        "r2_rasters": False,
        "windows": list(windows),
    }


def local_image_read_window(href: str, col: int, row: int, width: int, height: int) -> np.ndarray:
    """Read a window from a local raster with Pillow. Not for remote JPEG COGs."""
    path = Path(urlparse(href).path if href.startswith("file:") else href)
    if not path.is_file():
        raise ChipExtractError(f"local chip source missing: {path}")
    img = Image.open(path).convert("RGB")
    box = (int(col), int(row), int(col) + int(width), int(row) + int(height))
    crop = img.crop(box)
    if crop.size != (int(width), int(height)):
        crop = crop.resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def rasterio_read_window(href: str, col: int, row: int, width: int, height: int) -> np.ndarray:
    """Range-read a COG window. Requires rasterio (optional ``monarc[ingest]`` extra)."""
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ChipExtractError(
            "range-read of remote COGs needs rasterio; pip install 'monarc[ingest]'"
        ) from exc
    with rasterio.open(href) as dataset:
        data = dataset.read(window=Window(int(col), int(row), int(width), int(height)))
    if data.ndim != 3:
        raise ChipExtractError(f"unexpected raster rank {data.ndim}")
    rgb = np.transpose(data[:3], (1, 2, 0))
    if rgb.dtype != np.uint8:
        finite = np.clip(rgb.astype(np.float32), 0, 255)
        rgb = finite.astype(np.uint8)
    return np.ascontiguousarray(rgb)


def resolve_window_reader(href: str, reader: WindowReader | None = None) -> WindowReader:
    if reader is not None:
        return reader
    parsed = urlparse(href)
    if parsed.scheme in {"", "file"} or Path(href).is_file():
        return local_image_read_window
    return rasterio_read_window


def window_read_href(window: JsonObj) -> str:
    signed = window.get("signed_href")
    if signed:
        return str(signed)
    href = str(window.get("catalog_href") or "")
    token = (window.get("sas") or {}).get("token")
    if token and is_azure_blob_href(href):
        return apply_sas_token(href, str(token))
    return href


def materialize_chip_windows(
    windows: Sequence[JsonObj],
    out_dir: str | Path,
    *,
    reader: WindowReader | None = None,
    refresh: Callable[[JsonObj], JsonObj] | None = None,
) -> dict[str, Any]:
    """Write PNG chips + xyz sidecars. Does not write GeoTIFF and does not use R2."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for window in windows:
        rec = dict(window)
        if refresh is not None and token_expired(rec.get("sas_expiry")):
            rec = refresh(rec)
        href = window_read_href(rec)
        if not href:
            raise ChipExtractError(f"chip {rec.get('id')!r} has no HREF")
        col = int(rec.get("col_off", 0))
        row = int(rec.get("row_off", 0))
        width = int(rec.get("width") or rec.get("size_px") or DEFAULT_CHIP_SIZE_PX)
        height = int(rec.get("height") or rec.get("size_px") or DEFAULT_CHIP_SIZE_PX)
        use = resolve_window_reader(href, reader)
        rgb = use(href, col, row, width, height)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ChipExtractError(f"chip {rec.get('id')!r} is not RGB")
        name = str(rec.get("id") or f"chip-{len(written):04d}")
        png = out / f"{name}.png"
        Image.fromarray(rgb[:, :, :3]).save(png)
        xyz = rec.get("xyz") or [float("nan"), float("nan"), float("nan")]
        (out / f"{name}.xyz.json").write_text(
            json.dumps(
                {
                    "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "lat": rec.get("lat"),
                    "lon": rec.get("lon"),
                    "item_id": rec.get("item_id"),
                    "range_read": True,
                }
            )
            + "\n"
        )
        written.append(png.name)
    meta = {
        "n_chips": len(written),
        "out_dir": str(out),
        "files": written,
        "rasters_copied": False,
        "full_geotiff": False,
        "r2_rasters": False,
        "format": "png",
    }
    if windows:
        for key in ("overlap_frac", "stride_m", "chip_size_m"):
            if key in windows[0]:
                meta[key] = windows[0][key]
    (out / "chips_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def materialize_chips_from_manifest(
    manifest: JsonObj | str | Path,
    out_dir: str | Path,
    *,
    reader: WindowReader | None = None,
    refresh: Callable[[JsonObj], JsonObj] | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    windows = ((manifest.get("chip_extract") or {}).get("windows")) or []
    if not windows:
        raise ChipExtractError("manifest has no chip_extract.windows")
    return materialize_chip_windows(windows, out_dir, reader=reader, refresh=refresh)
