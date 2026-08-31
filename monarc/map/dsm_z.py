"""Fill coarse chip-center ENU heights from range-read 3DEP rasters."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from monarc.common.coordinates import geodetic_to_enu
from monarc.data.pc_sas import apply_sas_token, is_azure_blob_href

JsonObj = dict[str, Any]
ElevationSampler = Callable[[str, float, float], float]


def sample_elevation(href: str, lon: float, lat: float) -> float:
    """Sample band 1 at a WGS84 point; return NaN outside coverage or at nodata."""
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError(
            "3DEP sampling needs rasterio; pip install 'monarc[ingest]'"
        ) from exc

    try:
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                return float("nan")
            xs, ys = transform("EPSG:4326", dataset.crs, [float(lon)], [float(lat)])
            x, y = xs[0], ys[0]
            if not (dataset.bounds.left <= x <= dataset.bounds.right):
                return float("nan")
            if not (dataset.bounds.bottom <= y <= dataset.bounds.top):
                return float("nan")
            value = next(dataset.sample([(x, y)], indexes=1, masked=True))[0]
            if np.ma.is_masked(value):
                return float("nan")
            height = float(value)
            if not np.isfinite(height):
                return float("nan")
            if dataset.nodata is not None and np.isclose(height, float(dataset.nodata)):
                return float("nan")
            return height
    except (OSError, StopIteration, ValueError):
        return float("nan")


def _contains(record: JsonObj, lon: float, lat: float) -> bool:
    bbox = record.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    west, south, east, north = (float(value) for value in bbox)
    return west <= lon <= east and south <= lat <= north


def _record_href(record: JsonObj) -> str | None:
    href = record.get("download_url") or record.get("signed_href") or record.get("catalog_href")
    if not href:
        return None
    href = str(href)
    token = (record.get("sas") or {}).get("token")
    if not record.get("signed_href") and token and is_azure_blob_href(href):
        return apply_sas_token(href, str(token))
    return href


def records_from_manifest(manifest: JsonObj) -> list[JsonObj]:
    """Return TNM public HTTPS records first, then PC seamless fallbacks."""
    from monarc.data.aflora_ingest import public_threedep_href

    records: list[JsonObj] = []
    for record in (manifest.get("threedep") or {}).get("items") or []:
        href = public_threedep_href(record.get("download_url"))
        if record.get("access") == "public-https" and href:
            public_record = dict(record)
            public_record["download_url"] = href
            records.append(public_record)
    records.extend(dict(record) for record in (manifest.get("threedep_pc") or {}).get("items") or [])
    return records


def fill_chip_xyz_z(
    xyz: np.ndarray,
    lats: Sequence[float],
    lons: Sequence[float],
    origin_lat: float,
    origin_lon: float,
    records: Sequence[JsonObj],
    *,
    sampler: ElevationSampler = sample_elevation,
) -> np.ndarray:
    """Return xyz with ENU up filled where a covering 3DEP sample succeeds.

    Existing XY is preserved so a previously established spatial split is unchanged.
    """
    filled = np.asarray(xyz, dtype=np.float64).copy()
    if filled.ndim != 2 or filled.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if len(lats) != filled.shape[0] or len(lons) != filled.shape[0]:
        raise ValueError("lat/lon rows must match xyz")
    for index, (lat, lon) in enumerate(zip(lats, lons, strict=True)):
        lat_f, lon_f = float(lat), float(lon)
        for record in records:
            if not _contains(record, lon_f, lat_f):
                continue
            href = _record_href(record)
            if not href:
                continue
            height = float(sampler(href, lon_f, lat_f))
            if not np.isfinite(height):
                continue
            enu = geodetic_to_enu(lat_f, lon_f, height, origin_lat, origin_lon)
            filled[index, 2] = float(enu[2])
            break
    return filled


def _load_json(path: Path) -> JsonObj:
    return json.loads(path.read_text()) if path.is_file() else {}


def _chip_key(value: str) -> str:
    path = Path(value)
    return path.stem if path.suffix else path.name


def _lat_lon_rows(
    extract: Path, chips: Path | None, manifest: JsonObj, n: int
) -> tuple[list[str], list[float], list[float], dict[str, Path]]:
    ids_path = extract / "ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.is_file() else []
    if not ids:
        ids = [window["id"] for window in (manifest.get("chip_extract") or {}).get("windows") or []]
    if len(ids) != n:
        raise ValueError(f"chip ids {len(ids)} != xyz rows {n}")
    windows = {
        _chip_key(str(window.get("id", ""))): window
        for window in (manifest.get("chip_extract") or {}).get("windows") or []
    }
    sidecars: dict[str, Path] = {}
    lats: list[float] = []
    lons: list[float] = []
    for raw_id in ids:
        key = _chip_key(str(raw_id))
        payload: JsonObj = {}
        if chips is not None:
            sidecar = chips / f"{key}.xyz.json"
            if sidecar.is_file():
                payload = _load_json(sidecar)
                sidecars[key] = sidecar
        window = windows.get(key, {})
        lat = payload.get("lat", window.get("lat"))
        lon = payload.get("lon", window.get("lon"))
        if lat is None or lon is None:
            raise ValueError(f"chip {raw_id!r} has no lat/lon sidecar or manifest window")
        lats.append(float(lat))
        lons.append(float(lon))
    return [str(value) for value in ids], lats, lons, sidecars


def _update_meta(directory: Path, xyz: np.ndarray) -> None:
    path = directory / "meta.json"
    meta = _load_json(path)
    meta["xyz_finite"] = bool(np.isfinite(xyz).all())
    meta["xyz_kind"] = "coarse-chip-center"
    meta["xyz_is_chip_center"] = True
    meta["rasters_copied"] = False
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def fill_xyz_dirs(
    extract_dir: str | Path,
    *,
    manifest_path: str | Path,
    fsq_dir: str | Path | None = None,
    chips_dir: str | Path | None = None,
    href: str | None = None,
    offline: bool = False,
    sampler: ElevationSampler = sample_elevation,
) -> JsonObj:
    """Fill extract/FSQ arrays and matching sidecars from chip-center samples."""
    extract = Path(extract_dir)
    fsq = Path(fsq_dir) if fsq_dir is not None else None
    chips = Path(chips_dir) if chips_dir is not None else None
    manifest = _load_json(Path(manifest_path))
    xyz = np.asarray(np.load(extract / "xyz.npy"), dtype=np.float64).reshape(-1, 3)
    ids, lats, lons, sidecars = _lat_lon_rows(extract, chips, manifest, xyz.shape[0])
    aoi = manifest.get("aoi") or {}
    origin_lat = aoi.get("center_lat")
    origin_lon = aoi.get("center_lon")
    if origin_lat is None or origin_lon is None:
        raise ValueError("manifest aoi must contain center_lat and center_lon")
    records = records_from_manifest(manifest)
    if href is not None:
        records = [{"bbox": [-180.0, -90.0, 180.0, 90.0], "download_url": href}]
    if offline:
        remote = [r for r in records if urlparse(_record_href(r) or "").scheme in {"http", "https"}]
        if remote:
            raise ValueError("--offline requires --href to be a local raster")
    before = np.isfinite(xyz[:, 2])
    filled = fill_chip_xyz_z(
        xyz, lats, lons, float(origin_lat), float(origin_lon), records, sampler=sampler
    )
    np.save(extract / "xyz.npy", filled)
    _update_meta(extract, filled)
    if fsq is not None:
        fsq_xyz = np.asarray(np.load(fsq / "xyz.npy"), dtype=np.float64).reshape(-1, 3)
        if fsq_xyz.shape != filled.shape:
            raise ValueError("FSQ xyz shape does not match extract xyz")
        fsq_xyz[:, 2] = filled[:, 2]
        np.save(fsq / "xyz.npy", fsq_xyz)
        _update_meta(fsq, fsq_xyz)
    for index, raw_id in enumerate(ids):
        key = _chip_key(raw_id)
        sidecar = sidecars.get(key)
        if sidecar is None:
            continue
        payload = _load_json(sidecar)
        payload["xyz"] = [float(value) for value in filled[index]]
        sidecar.write_text(json.dumps(payload, sort_keys=True) + "\n")
    after = np.isfinite(filled[:, 2])
    return {
        "n_chips": int(filled.shape[0]),
        "n_z_filled": int(np.count_nonzero(after & ~before)),
        "n_z_finite": int(np.count_nonzero(after)),
        "n_z_nan": int(np.count_nonzero(~after)),
        "xyz_kind": "coarse-chip-center",
        "xyz_is_chip_center": True,
        "rasters_copied": False,
        "network": not offline and any(
            urlparse(_record_href(record) or "").scheme in {"http", "https"} for record in records
        ),
    }
