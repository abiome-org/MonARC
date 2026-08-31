"""CPU-only chip-center 3DEP sampling tests using a local GeoTIFF."""

import json

import numpy as np
import pytest

from monarc.cli import main
from monarc.common.coordinates import geodetic_to_enu
from monarc.map.dsm_z import (
    fill_chip_xyz_z,
    fill_xyz_dirs,
    records_from_manifest,
    sample_elevation,
)

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin


def _write_dem(path):
    values = np.array([[100.0, 110.0], [-9999.0, 130.0]], dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-105.02, 40.02, 0.01, 0.01),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)
    return path


def _write_sloped_dem(path):
    height = width = 120
    rows, cols = np.indices((height, width), dtype=np.float32)
    values = 200.0 + rows + 2.0 * cols
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_origin(-105.011, 40.011, 0.00002, 0.00002),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)
    return path


def _manifest(path, dem):
    payload = {
        "aoi": {"center_lat": 40.01, "center_lon": -105.01},
        "threedep": {
            "items": [
                {
                    "bbox": [-105.02, 40.0, -105.0, 40.02],
                    "download_url": str(dem),
                    "access": "public-https",
                }
            ]
        },
        "threedep_pc": {"items": []},
        "chip_extract": {
            "windows": [
                {"id": "chip-00-00", "lat": 40.015, "lon": -105.015},
                {"id": "chip-00-01", "lat": 40.005, "lon": -105.015},
                {"id": "chip-00-02", "lat": 39.99, "lon": -105.015},
            ]
        },
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _cache_dirs(tmp_path, dem):
    extract, fsq, chips = tmp_path / "extract", tmp_path / "fsq", tmp_path / "chips"
    extract.mkdir()
    fsq.mkdir()
    chips.mkdir()
    xyz = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan], [5.0, 6.0, np.nan]])
    np.save(extract / "xyz.npy", xyz)
    np.save(fsq / "xyz.npy", xyz)
    ids = ["chip-00-00.png", "chip-00-01.png", "chip-00-02.png"]
    (extract / "ids.json").write_text(json.dumps(ids) + "\n")
    (extract / "meta.json").write_text(json.dumps({"has_dsm": False}) + "\n")
    (fsq / "meta.json").write_text(json.dumps({"has_dsm": False}) + "\n")
    points = [(40.015, -105.015), (40.005, -105.015), (39.99, -105.015)]
    for chip_id, (lat, lon), row in zip(ids, points, xyz, strict=True):
        name = chip_id.removesuffix(".png")
        (chips / f"{name}.xyz.json").write_text(
            json.dumps({"xyz": row.tolist(), "lat": lat, "lon": lon}) + "\n"
        )
    manifest = _manifest(tmp_path / "manifest.json", dem)
    return extract, fsq, chips, manifest


def test_sample_elevation_known_nodata_and_outside(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif")
    assert sample_elevation(str(dem), -105.015, 40.015) == pytest.approx(100.0)
    assert np.isnan(sample_elevation(str(dem), -105.015, 40.005))
    assert np.isnan(sample_elevation(str(dem), -105.015, 39.99))


def test_manifest_tnm_s3_reference_becomes_public_https():
    records = records_from_manifest(
        {
            "threedep": {
                "items": [
                    {
                        "bbox": [-106.0, 39.0, -104.0, 41.0],
                        "download_url": "s3://prd-tnm/StagedProducts/Elevation/1m/fixture.tif",
                        "access": "public-https",
                    }
                ]
            }
        }
    )
    assert records[0]["download_url"] == (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/fixture.tif"
    )


def test_fill_helper_preserves_xy_and_leaves_misses_nan():
    xyz = np.array([[10.0, 20.0, np.nan], [30.0, 40.0, np.nan]])
    records = [{"bbox": [-106.0, 39.0, -104.0, 41.0], "download_url": "fixture"}]
    filled = fill_chip_xyz_z(
        xyz,
        [40.0, 40.1],
        [-105.0, -105.1],
        40.0,
        -105.0,
        records,
        sampler=lambda _href, lon, _lat: 250.0 if lon == -105.0 else float("nan"),
    )
    assert np.array_equal(filled[:, :2], xyz[:, :2])
    expected = geodetic_to_enu(40.0, -105.0, 250.0, 40.0, -105.0)[2]
    assert filled[0, 2] == pytest.approx(expected)
    assert np.isnan(filled[1, 2])


def test_fill_dirs_updates_extract_fsq_sidecars_and_meta(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif")
    extract, fsq, chips, manifest = _cache_dirs(tmp_path, dem)
    report = fill_xyz_dirs(
        extract,
        fsq_dir=fsq,
        chips_dir=chips,
        manifest_path=manifest,
        href=str(dem),
        offline=True,
    )
    extract_xyz = np.load(extract / "xyz.npy")
    assert report["n_z_filled"] == 1
    assert report["rasters_copied"] is False
    assert np.array_equal(extract_xyz[:, :2], np.array([[1, 2], [3, 4], [5, 6]]))
    assert np.allclose(np.load(fsq / "xyz.npy"), extract_xyz, equal_nan=True)
    assert np.isfinite(extract_xyz[0, 2])
    assert np.isnan(extract_xyz[1:, 2]).all()
    sidecar = json.loads((chips / "chip-00-00.xyz.json").read_text())
    assert sidecar["xyz"][2] == pytest.approx(extract_xyz[0, 2])
    for directory in (extract, fsq):
        meta = json.loads((directory / "meta.json").read_text())
        assert meta["has_dsm"] is False
        assert meta["xyz_finite"] is False
        assert meta["xyz_kind"] == "coarse-chip-center"
        assert meta["rasters_copied"] is False


def test_cli_fill_dsm_z_writes_updated_xyz(tmp_path, capsys):
    dem = _write_dem(tmp_path / "dem.tif")
    extract, fsq, chips, manifest = _cache_dirs(tmp_path, dem)
    assert main(
        [
            "fill-dsm-z",
            "--extract", str(extract),
            "--fsq", str(fsq),
            "--chips", str(chips),
            "--manifest", str(manifest),
            "--href", str(dem),
            "--offline",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_z_filled"] == 1
    assert np.isfinite(np.load(extract / "xyz.npy")[0, 2])


def test_cli_fill_dsm_z_writes_varying_per_patch_xyz(tmp_path, capsys):
    dem = _write_sloped_dem(tmp_path / "slope.tif")
    extract, fsq, chips, manifest = _cache_dirs(tmp_path, dem)
    # Keep this fixture inside the sloped DEM and provide the DINO grid contract.
    payload = json.loads(manifest.read_text())
    for window in payload["chip_extract"]["windows"]:
        window.update(lat=40.01, lon=-105.01, gsd_m=0.6)
    manifest.write_text(json.dumps(payload) + "\n")
    for sidecar in chips.glob("*.xyz.json"):
        row = json.loads(sidecar.read_text())
        row.update(lat=40.01, lon=-105.01)
        sidecar.write_text(json.dumps(row) + "\n")
    np.save(extract / "features.npy", np.zeros((3, 2, 4, 4), dtype=np.float32))
    meta = json.loads((extract / "meta.json").read_text())
    meta["patch_size"] = 14
    (extract / "meta.json").write_text(json.dumps(meta) + "\n")

    assert main([
        "fill-dsm-z", "--extract", str(extract), "--fsq", str(fsq),
        "--chips", str(chips), "--manifest", str(manifest), "--href", str(dem),
        "--offline", "--patches",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    patch_xyz = np.load(extract / "patch_xyz.npy")
    assert patch_xyz.shape == (3, 4, 4, 3)
    assert np.isfinite(patch_xyz).all()
    assert np.unique(patch_xyz[0, ..., 0]).size > 1
    assert np.unique(patch_xyz[0, ..., 1]).size > 1
    assert np.unique(patch_xyz[0, ..., 2]).size > 1
    assert np.allclose(np.load(fsq / "patch_xyz.npy"), patch_xyz)
    assert report["xyz_kind"] == "per-patch-3dep"
    assert report["xyz_is_chip_center"] is False
    assert report["rasters_copied"] is False
    assert json.loads((extract / "meta.json").read_text())["has_dsm"] is False
