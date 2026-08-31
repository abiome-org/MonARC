"""Fill coarse chip-center ENU heights from range-read 3DEP rasters."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from monarc.common.coordinates import geodetic_to_enu, meters_per_degree
from monarc.data.pc_sas import apply_sas_token, is_azure_blob_href

JsonObj = dict[str, Any]
ElevationSampler = Callable[[str, float, float], float]
BatchElevationSampler = Callable[[str, Sequence[float], Sequence[float]], np.ndarray]


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


def sample_elevations(
    href: str, lons: Sequence[float], lats: Sequence[float]
) -> np.ndarray:
    """Sample many WGS84 points while opening the source COG only once.

    Compact clusters (one DINO chip) are window-read; sparse/large spans fall
    back to ``dataset.sample``.
    """
    if len(lons) != len(lats):
        raise ValueError("lat/lon sample counts differ")
    result = np.full(len(lons), np.nan, dtype=np.float64)
    if result.size == 0:
        return result
    try:
        import rasterio
        from rasterio.transform import rowcol
        from rasterio.warp import transform
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "3DEP sampling needs rasterio; pip install 'monarc[ingest]'"
        ) from exc
    try:
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                return result
            xs, ys = transform(
                "EPSG:4326", dataset.crs,
                [float(value) for value in lons], [float(value) for value in lats],
            )
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            inside = (
                (xs >= dataset.bounds.left)
                & (xs <= dataset.bounds.right)
                & (ys >= dataset.bounds.bottom)
                & (ys <= dataset.bounds.top)
            )
            if not np.any(inside):
                return result
            rows, cols = rowcol(dataset.transform, xs, ys)
            rows = np.asarray(rows, dtype=np.float64)
            cols = np.asarray(cols, dtype=np.float64)
            inside_idx = np.flatnonzero(inside)
            rmin = int(np.floor(rows[inside_idx].min()))
            rmax = int(np.ceil(rows[inside_idx].max()))
            cmin = int(np.floor(cols[inside_idx].min()))
            cmax = int(np.ceil(cols[inside_idx].max()))
            rmin = max(rmin, 0)
            cmin = max(cmin, 0)
            rmax = min(rmax, int(dataset.height) - 1)
            cmax = min(cmax, int(dataset.width) - 1)
            height = rmax - rmin + 1
            width = cmax - cmin + 1
            max_window = 2048
            use_window = height > 0 and width > 0 and height * width <= max_window * max_window
            if use_window:
                band = dataset.read(1, window=Window(cmin, rmin, width, height), masked=True)
                for index in inside_idx.tolist():
                    rr = int(round(rows[index])) - rmin
                    cc = int(round(cols[index])) - cmin
                    if rr < 0 or cc < 0 or rr >= height or cc >= width:
                        continue
                    value = band[rr, cc]
                    if np.ma.is_masked(value):
                        continue
                    height_m = float(value)
                    if not np.isfinite(height_m):
                        continue
                    if dataset.nodata is not None and np.isclose(height_m, float(dataset.nodata)):
                        continue
                    result[index] = height_m
            else:
                points = [(float(xs[i]), float(ys[i])) for i in inside_idx.tolist()]
                for index, sample in zip(
                    inside_idx.tolist(),
                    dataset.sample(points, indexes=1, masked=True),
                    strict=True,
                ):
                    value = sample[0]
                    if np.ma.is_masked(value):
                        continue
                    height_m = float(value)
                    if not np.isfinite(height_m):
                        continue
                    if dataset.nodata is not None and np.isclose(height_m, float(dataset.nodata)):
                        continue
                    result[index] = height_m
    except (OSError, StopIteration, ValueError):
        pass
    return result


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


def _patch_lon_lat(
    lats: Sequence[float],
    lons: Sequence[float],
    gsd: Sequence[float],
    height: int,
    width: int,
    patch_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return patch-center latitude/longitude arrays shaped [N, H, W]."""
    rows, cols = np.indices((height, width), dtype=np.float64)
    px = (cols + 0.5) * patch_size
    py = (rows + 0.5) * patch_size
    size_x = width * patch_size
    size_y = height * patch_size
    patch_lats = np.empty((len(lats), height, width), dtype=np.float64)
    patch_lons = np.empty_like(patch_lats)
    for index, (lat, lon, gsd_m) in enumerate(zip(lats, lons, gsd, strict=True)):
        m_lat, m_lon = meters_per_degree(float(lat))
        patch_lats[index] = float(lat) + (size_y / 2.0 - py) * float(gsd_m) / m_lat
        patch_lons[index] = float(lon) + (px - size_x / 2.0) * float(gsd_m) / m_lon
    return patch_lats, patch_lons


def fill_patch_xyz(
    patch_lats: np.ndarray,
    patch_lons: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    records: Sequence[JsonObj],
    *,
    sampler: BatchElevationSampler = sample_elevations,
) -> np.ndarray:
    """Sample per-patch terrain and return metric ENU xyz, retaining NaN failures.

    Each chip is sampled as one compact batch so HTTPS COG range-reads stay
    window-sized rather than one request per DINO cell.
    """
    if patch_lats.shape != patch_lons.shape or patch_lats.ndim != 3:
        raise ValueError("patch lat/lon must have matching [N, H, W] shapes")
    n_chips, height, width = patch_lats.shape
    out = np.full((n_chips, height, width, 3), np.nan, dtype=np.float64)
    for chip in range(n_chips):
        flat_lat = patch_lats[chip].reshape(-1)
        flat_lon = patch_lons[chip].reshape(-1)
        heights = np.full(flat_lat.shape, np.nan, dtype=np.float64)
        for record in records:
            href = _record_href(record)
            if not href:
                continue
            unresolved = ~np.isfinite(heights)
            indices = np.array(
                [
                    i
                    for i in np.flatnonzero(unresolved).tolist()
                    if _contains(record, float(flat_lon[i]), float(flat_lat[i]))
                ],
                dtype=np.int64,
            )
            if not indices.size:
                continue
            sampled = np.asarray(
                sampler(href, flat_lon[indices].tolist(), flat_lat[indices].tolist()),
                dtype=np.float64,
            )
            if sampled.shape != (indices.size,):
                raise ValueError("batch elevation sampler returned the wrong shape")
            valid = np.isfinite(sampled)
            heights[indices[valid]] = sampled[valid]
            if np.isfinite(heights).all():
                break
        enu = geodetic_to_enu(flat_lat, flat_lon, heights, origin_lat, origin_lon)
        enu[~np.isfinite(heights)] = np.nan
        out[chip] = enu.reshape(height, width, 3)
    return out


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
    patches: bool = False,
    gsd_m: float = 0.3,
    sampler: ElevationSampler = sample_elevation,
    batch_sampler: BatchElevationSampler = sample_elevations,
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
    report = {
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
    if patches:
        features = np.load(extract / "features.npy", mmap_mode="r")
        if features.ndim != 4:
            raise ValueError("per-patch xyz needs features shaped [N, C, H, W]")
        if features.shape[0] != filled.shape[0]:
            raise ValueError("feature and xyz chip counts differ")
        meta = _load_json(extract / "meta.json")
        patch_size = float(meta.get("patch_size", 14))
        windows = {
            _chip_key(str(window.get("id", ""))): window
            for window in (manifest.get("chip_extract") or {}).get("windows") or []
        }
        naip_items = {
            str(item.get("id")): item for item in (manifest.get("naip") or {}).get("items") or []
            if item.get("id") is not None
        }
        gsds = []
        for raw_id in ids:
            window = windows.get(_chip_key(raw_id), {})
            item = naip_items.get(str(window.get("item_id")), {})
            item_properties = item.get("properties") or {}
            value = window.get("gsd_m", window.get("gsd"))
            if value is None:
                value = item.get("gsd", item_properties.get("gsd", gsd_m))
            value = gsd_m if value is None else value
            if float(value) <= 0:
                raise ValueError("GSD must be positive")
            gsds.append(float(value))
        patch_lats, patch_lons = _patch_lon_lat(
            lats, lons, gsds, int(features.shape[2]), int(features.shape[3]), patch_size
        )
        patch_xyz = fill_patch_xyz(
            patch_lats, patch_lons, float(origin_lat), float(origin_lon), records,
            sampler=batch_sampler,
        )
        np.save(extract / "patch_xyz.npy", patch_xyz)
        if fsq is not None:
            np.save(fsq / "patch_xyz.npy", patch_xyz)
        for directory in (extract, fsq):
            if directory is None:
                continue
            patch_meta = _load_json(directory / "meta.json")
            patch_meta["xyz_kind"] = "per-patch-3dep"
            patch_meta["xyz_is_chip_center"] = False
            patch_meta["patch_xyz_finite"] = bool(np.isfinite(patch_xyz).all())
            patch_meta["rasters_copied"] = False
            (directory / "meta.json").write_text(
                json.dumps(patch_meta, indent=2, sort_keys=True) + "\n"
            )
        report.update(
            xyz_kind="per-patch-3dep",
            xyz_is_chip_center=False,
            n_patch_samples=int(np.prod(patch_xyz.shape[:-1])),
            n_patch_xyz_finite=int(np.count_nonzero(np.isfinite(patch_xyz).all(axis=-1))),
        )
    return report
