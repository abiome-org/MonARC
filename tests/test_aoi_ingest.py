"""AOI ingest against offline fixtures. No live STAC/TNM in tests."""

import json
from pathlib import Path

from monarc.common.coordinates import GOLDEN_MORRISON_AOI
from monarc.data.aflora_ingest import (
    build_aoi_manifest,
    ingest_aoi_to_path,
    load_offline_inventory,
    visualization_uri_from_stac_item,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "inventory"


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


def test_offline_manifest_intersects_only(tmp_path):
    naip_items, tnm_items = load_offline_inventory(FIXTURE_DIR)
    manifest = build_aoi_manifest(
        GOLDEN_MORRISON_AOI,
        naip_items=naip_items,
        threedep_items=tnm_items,
        queried_at="1970-01-01T00:00:00Z",
    )
    naip_ids = {it["id"] for it in manifest["naip_visualization"]["items"]}
    tnm_ids = {it["source_id"] for it in manifest["threedep"]["items"]}
    assert "fixture-naip-intersect-a" in naip_ids
    assert "fixture-naip-intersect-b" in naip_ids
    assert "fixture-naip-outside" not in naip_ids
    assert "fixture-3dep-a" in tnm_ids
    assert "fixture-3dep-far" not in tnm_ids
    assert manifest["persist"]["rasters"] is False
    assert manifest["persist"]["naip_source"] is False
    assert manifest["persist"]["r2"] is False
    assert manifest["aoi"]["product_boundary"] == "colorado-state"
    vis = {it["visualization_uri"] for it in manifest["naip_visualization"]["items"]}
    assert all(v.startswith("s3://naip-visualization/") for v in vis if v)


def test_cli_offline_ingest(tmp_path):
    out = tmp_path / "manifest.json"
    path = ingest_aoi_to_path(out, offline=FIXTURE_DIR)
    payload = json.loads(path.read_text())
    assert payload["naip_visualization"]["n_items"] == 2
    assert payload["threedep"]["n_items"] == 2


def test_live_search_is_not_used_when_items_injected():
    def boom_get(_url):
        raise AssertionError("HTTP GET must not run in fixture mode")

    def boom_post(_url, _body):
        raise AssertionError("HTTP POST must not run in fixture mode")

    naip_items, tnm_items = load_offline_inventory(FIXTURE_DIR)
    manifest = build_aoi_manifest(
        GOLDEN_MORRISON_AOI,
        naip_items=naip_items,
        threedep_items=tnm_items,
        http_get=boom_get,
        http_post=boom_post,
    )
    assert manifest["naip_visualization"]["n_items"] == 2
