"""AOI ingest against offline fixtures. No live STAC/TNM/SAS in tests."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from monarc.cli import main
from monarc.common.coordinates import GOLDEN_MORRISON_AOI, box_from_center, geodetic_to_utm
from monarc.data.aflora_ingest import (
    DEFAULT_SOURCE,
    SOURCE_COLORADO_PUBLIC,
    SOURCE_NAIP_VISUALIZATION,
    SOURCE_PLANETARY_COMPUTER,
    build_aoi_manifest,
    ingest_aoi_to_path,
    load_offline_inventory,
    parse_s3_list_xml,
    public_threedep_href,
    visualization_uri_from_stac_item,
)
from monarc.data.pc_sas import apply_sas_token, unsigned_href
from monarc.map.cog_chips import (
    materialize_chip_windows, materialize_chips_from_manifest,
    plan_aligned_overlap_windows, plan_chip_windows,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "inventory"


def _boom_get(_url):
    raise AssertionError("HTTP GET must not run in fixture mode")


def _boom_post(_url, _body):
    raise AssertionError("HTTP POST must not run in fixture mode")


def _boom_get_text(_url):
    raise AssertionError("HTTP GET text must not run in fixture mode")


def test_default_source_is_planetary_computer():
    assert DEFAULT_SOURCE == SOURCE_PLANETARY_COMPUTER


def test_visualization_uri_rewrite_does_not_keep_naip_source():
    item = {
        "properties": {"naip:state": "co", "naip:year": "2023", "gsd": 0.3},
        "assets": {
            "image": {
                "href": "s3://naip-source/co/2023/30cm/rgbir/39105/m_fixture_aa_sw_13_030_19700101.tif"
            }
        },
    }
    uri = visualization_uri_from_stac_item(item)
    assert uri is not None
    assert uri.startswith("s3://naip-visualization/")
    assert "naip-source" not in uri
    assert uri.endswith("m_fixture_aa_sw_13_030_19700101.tif")


def test_prd_tnm_s3_rewrites_to_public_https():
    href = public_threedep_href("s3://prd-tnm/StagedProducts/Elevation/1m/fixture.tif")
    assert href == "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/fixture.tif"


def test_offline_pc_manifest_signs_and_intersects_only(tmp_path):
    naip_items, tnm_items = load_offline_inventory(FIXTURE_DIR)
    token = json.loads((FIXTURE_DIR / "sas_token.json").read_text())["token"]
    expiry = json.loads((FIXTURE_DIR / "sas_token.json").read_text())["msft:expiry"]
    threedep_pc = json.loads((FIXTURE_DIR / "3dep_pc.json").read_text())["features"]
    manifest = build_aoi_manifest(
        GOLDEN_MORRISON_AOI,
        naip_items=naip_items,
        threedep_items=tnm_items,
        threedep_pc_items=threedep_pc,
        sas_token=token,
        sas_expiry=expiry,
        queried_at="1970-01-01T00:00:00Z",
        http_get=_boom_get,
        http_post=_boom_post,
        chip_grid=4,
        max_chips=16,
    )
    assert manifest["source"] == SOURCE_PLANETARY_COMPUTER
    assert manifest["credentials"]["aws_required"] is False
    assert manifest["credentials"]["reads_dot_aws"] is False
    naip_ids = {it["id"] for it in manifest["naip"]["items"]}
    tnm_ids = {it["source_id"] for it in manifest["threedep"]["items"]}
    assert "fixture-naip-intersect-a" in naip_ids
    assert "fixture-naip-intersect-b" in naip_ids
    assert "fixture-naip-outside" not in naip_ids
    assert "fixture-naip-old-vintage" not in naip_ids
    assert "fixture-3dep-a" in tnm_ids
    assert "fixture-3dep-far" not in tnm_ids
    assert manifest["persist"]["rasters"] is False
    assert manifest["persist"]["naip_source"] is False
    assert manifest["persist"]["r2"] is False
    assert manifest["persist"]["full_geotiff"] is False
    assert manifest["aoi"]["product_boundary"] == "colorado-state"
    assert manifest["naip_visualization"]["skipped"] is True
    for item in manifest["naip"]["items"]:
        assert item["catalog_href"].startswith("https://naipeuwest.blob.core.windows.net/")
        assert "sig=fixture" in item["signed_href"]
        assert item["sas"]["token_url"].endswith("/token/naip")
        assert not str(item["catalog_href"]).startswith("s3://naip-visualization/")
    for item in manifest["threedep"]["items"]:
        assert item["download_url"].startswith("https://")
        assert item["requester_pays"] is False
        assert item["rejected_prd_tnm_requester_pays"] is False
    assert manifest["chip_extract"]["range_read"] is True
    assert manifest["chip_extract"]["copy_full_geotiff"] is False
    assert manifest["chip_extract"]["n_windows"] > 0
    assert all(w.get("range_read") is True for w in manifest["chip_extract"]["windows"])


def test_aws_visualization_source_is_explicit(tmp_path):
    naip_items, tnm_items = load_offline_inventory(FIXTURE_DIR)
    manifest = build_aoi_manifest(
        GOLDEN_MORRISON_AOI,
        source=SOURCE_NAIP_VISUALIZATION,
        naip_items=naip_items,
        threedep_items=tnm_items,
        queried_at="1970-01-01T00:00:00Z",
        http_get=_boom_get,
        http_post=_boom_post,
    )
    assert manifest["source"] == SOURCE_NAIP_VISUALIZATION
    assert manifest["credentials"]["aws_required"] is True
    assert manifest["credentials"]["reads_dot_aws"] is False
    vis = {it["visualization_uri"] for it in manifest["naip_visualization"]["items"]}
    assert all(v.startswith("s3://naip-visualization/") for v in vis if v)
    assert manifest["naip_visualization"]["n_items"] == 2


def test_colorado_public_https_list_offline(tmp_path):
    xml = (FIXTURE_DIR / "colorado_list.xml").read_text()
    keys, token, truncated = parse_s3_list_xml(xml)
    assert token is None
    assert truncated is False
    assert len(keys) == 3
    _, tnm_items = load_offline_inventory(FIXTURE_DIR)
    manifest = build_aoi_manifest(
        GOLDEN_MORRISON_AOI,
        source=SOURCE_COLORADO_PUBLIC,
        threedep_items=tnm_items,
        colorado_keys=keys,
        queried_at="1970-01-01T00:00:00Z",
        http_get=_boom_get,
        http_get_text=_boom_get_text,
        http_post=_boom_post,
    )
    assert manifest["source"] == SOURCE_COLORADO_PUBLIC
    assert manifest["credentials"]["aws_required"] is False
    assert manifest["colorado_public_imagery"]["unsigned"] is True
    assert manifest["colorado_public_imagery"]["no_sign_request"] is True
    hrefs = [it["catalog_href"] for it in manifest["naip"]["items"]]
    assert hrefs
    assert all(h.startswith("https://colorado-public-imagery.s3.amazonaws.com/") for h in hrefs)
    assert all("3610201" not in h for h in hrefs)
    assert manifest["naip"]["n_items"] == 2


def test_cli_offline_ingest(tmp_path):
    out = tmp_path / "manifest.json"
    path = ingest_aoi_to_path(out, offline=FIXTURE_DIR)
    payload = json.loads(path.read_text())
    assert payload["source"] == SOURCE_PLANETARY_COMPUTER
    assert payload["naip"]["n_items"] == 2
    assert payload["threedep"]["n_items"] == 2
    assert payload["credentials"]["aws_required"] is False


def test_offline_ingest_records_overlap_stride_metadata(tmp_path):
    out = tmp_path / "overlap-manifest.json"
    path = ingest_aoi_to_path(out, offline=FIXTURE_DIR, overlap_frac=0.5,
                              gsd_m=0.3, chip_size=224, max_chips=8)
    plan = json.loads(path.read_text())["chip_extract"]
    assert plan["overlap_frac"] == 0.5
    assert plan["stride_m"] < plan["chip_size_m"]
    assert plan["n_windows"] <= plan["max_chips"] == 8
    assert plan["range_read"] is True
    assert plan["copy_full_geotiff"] is False
    assert plan["r2_rasters"] is False


def test_cli_offline_chips(tmp_path):
    out = tmp_path / "manifest.json"
    chips = tmp_path / "chips"

    def fake_reader(_href, _col, _row, width, height):
        return np.full((height, width, 3), 90, dtype=np.uint8)

    path = ingest_aoi_to_path(
        out,
        offline=FIXTURE_DIR,
        chips_dir=chips,
        chip_grid=2,
        max_chips=4,
        window_reader=fake_reader,
    )
    payload = json.loads(path.read_text())
    assert payload["chip_extract"]["n_windows"] <= 4
    pngs = list(chips.glob("*.png"))
    assert pngs
    assert not list(chips.glob("*.tif"))
    assert (chips / "chips_meta.json").is_file()
    meta = json.loads((chips / "chips_meta.json").read_text())
    assert meta["rasters_copied"] is False
    assert meta["full_geotiff"] is False


def test_overlap_windows_range_read_new_pixels_from_larger_tile(tmp_path):
    aoi = box_from_center(39.725, -105.220, 0.18)
    east, north, _zone, _hemi = geodetic_to_utm(aoi.center_lat, aoi.center_lon)
    size = 32
    item = {
        "id": "local-pattern-tile", "bbox": list(aoi.bbox), "catalog_href": "fixture://tile",
        "properties": {"proj:epsg": 26913, "proj:shape": [800, 800],
                       "proj:transform": [0.3, 0.0, east - 120.0, 0.0, -0.3, north + 120.0]},
    }
    windows = plan_chip_windows(aoi, [item], size_px=size, max_chips=8,
                                overlap_frac=0.5, gsd_m=0.3)
    assert len(windows) >= 2
    adjacent = next((a, b) for a, b in zip(windows, windows[1:])
                    if a["row_off"] == b["row_off"])
    first, second = adjacent
    stride_px = second["col_off"] - first["col_off"]
    assert 0 < stride_px < size

    yy, xx = np.mgrid[:800, :800]
    tile = np.stack((xx % 251, yy % 251, (xx + yy) % 251), axis=-1).astype(np.uint8)
    reads = []
    def reader(_href, col, row, width, height):
        reads.append((col, row, width, height))
        return tile[row:row + height, col:col + width]
    materialize_chip_windows([first, second], tmp_path / "chips", reader=reader)
    assert reads == [(first["col_off"], first["row_off"], size, size),
                     (second["col_off"], second["row_off"], size, size)]
    # The second range read includes tile pixels beyond the first standalone PNG.
    second_png = np.asarray(Image.open(tmp_path / "chips" / f"{second['id']}.png"))
    assert np.array_equal(second_png[:, -1], tile[second["row_off"]:second["row_off"] + size,
                                                   second["col_off"] + size - 1])
    assert second["col_off"] + size - 1 > first["col_off"] + size - 1


def test_aligned_overlap_windows_anchor_to_sparse_source_grid():
    aoi = box_from_center(39.725, -105.220, 4.0)
    east, north, _zone, _hemi = geodetic_to_utm(aoi.center_lat, aoi.center_lon)
    item = {
        "id": "source-item", "bbox": list(aoi.bbox), "catalog_href": "fixture://tile",
        "properties": {"proj:epsg": 26913, "proj:shape": [20000, 20000],
                       "proj:transform": [0.3, 0.0, east - 3000.0, 0.0, -0.3, north + 3000.0]},
    }
    source = plan_chip_windows(aoi, [item], size_px=64, grid=2, max_chips=4, gsd_m=0.3)
    windows = plan_aligned_overlap_windows(
        source, [item], aoi, size_px=64, overlap_frac=0.5, gsd_m=0.3, max_chips=6)
    assert 0 < len(windows) <= 6
    assert all(w["item_id"] == "source-item" for w in windows)
    by_source = {}
    for window in windows:
        by_source.setdefault(window["aligned_to"], []).append(window)
        origin = next(s for s in source if s["id"] == window["aligned_to"])
        delta = np.abs(np.asarray(window["xyz"][:2]) - np.asarray(origin["xyz"][:2]))
        assert np.all(delta < window["chip_size_m"])
        assert window["stride_m"] < window["size_px"] * 0.3
    pair = next(group for group in by_source.values() if len(group) == 2)
    assert np.linalg.norm(np.asarray(pair[0]["xyz"][:2]) - np.asarray(pair[1]["xyz"][:2])) <= pair[0]["chip_size_m"]


def test_cli_offline_align_to_keeps_source_aoi(tmp_path):
    source = tmp_path / "source.json"
    ingest_aoi_to_path(source, offline=FIXTURE_DIR, chip_grid=2, max_chips=4)
    out = tmp_path / "aligned.json"
    code = main(["ingest-aoi", "--align-to", str(source), "--offline", str(FIXTURE_DIR),
                 "--center", "0,0", "--max-chips", "4", "--out", str(out)])
    assert code == 0
    source_payload, aligned = json.loads(source.read_text()), json.loads(out.read_text())
    assert aligned["aoi"] == source_payload["aoi"]
    assert aligned["chip_extract"]["aligned_to"] == str(source)
    assert aligned["chip_extract"]["overlap_frac"] == 0.5
    assert aligned["chip_extract"]["n_windows"] > 0
    assert all(w.get("aligned_to") for w in aligned["chip_extract"]["windows"])


def test_materialize_only_from_manifest(tmp_path):
    out = tmp_path / "manifest.json"
    ingest_aoi_to_path(out, offline=FIXTURE_DIR, chip_grid=2, max_chips=2)
    chips = tmp_path / "chips"

    def fake_reader(_href, _col, _row, width, height):
        return np.zeros((height, width, 3), dtype=np.uint8)

    materialize_chips_from_manifest(out, chips, reader=fake_reader)
    assert list(chips.glob("*.png"))
    assert list(chips.glob("*.xyz.json"))


def test_live_search_is_not_used_when_items_injected():
    naip_items, tnm_items = load_offline_inventory(FIXTURE_DIR)
    manifest = build_aoi_manifest(
        GOLDEN_MORRISON_AOI,
        naip_items=naip_items,
        threedep_items=tnm_items,
        http_get=_boom_get,
        http_post=_boom_post,
    )
    assert manifest["naip"]["n_items"] == 2


def test_sas_apply_and_strip():
    href = "https://naipeuwest.blob.core.windows.net/naip/v002/co/x.tif"
    signed = apply_sas_token(href, "se=2099&sig=abc")
    assert signed.endswith("?se=2099&sig=abc")
    assert unsigned_href(signed) == href
    assert apply_sas_token(signed, "se=other&sig=x") == signed


def test_cli_source_flag(tmp_path, capsys):
    out = tmp_path / "m.json"
    code = main(
        [
            "ingest-aoi",
            "--out",
            str(out),
            "--offline",
            str(FIXTURE_DIR),
            "--source",
            "naip-visualization",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["source"] == "naip-visualization"
    assert payload["credentials"]["aws_required"] is True
