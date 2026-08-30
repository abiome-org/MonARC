"""AOI ingest: Golden-Morrison rehearsal box x NAIP visualization STAC x 3DEP inventory.

Queries catalogs at launch time. Tile IDs are not hardcoded. Rasters are not
downloaded or written; the helper emits a JSON manifest of intersecting assets.
``naip-source`` URIs are recorded only as rejected sources.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monarc.common.coordinates import GOLDEN_MORRISON_AOI, AoiBox, box_from_center

NAIP_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
NAIP_STAC_COLLECTION = "naip"
NAIP_VISUALIZATION_BUCKET = "s3://naip-visualization"
TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_3DEP_1M = "Digital Elevation Model (DEM) 1 meter"

JsonObj = dict[str, Any]
HttpGet = Callable[[str], JsonObj]
HttpPost = Callable[[str, JsonObj], JsonObj]


class InventoryError(RuntimeError):
    """Raised when a catalog query fails in live mode."""


def golden_morrison_aoi() -> AoiBox:
    return GOLDEN_MORRISON_AOI


def parse_center(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("center must be 'lat,lon'")
    return float(parts[0]), float(parts[1])


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_http_get(url: str, timeout: float = 30.0) -> JsonObj:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "monarc-aoi-ingest"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise InventoryError(f"GET failed for {url}: {exc}") from exc


def default_http_post(url: str, body: JsonObj, timeout: float = 30.0) -> JsonObj:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "monarc-aoi-ingest",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise InventoryError(f"POST failed for {url}: {exc}") from exc


def visualization_uri_from_stac_item(item: JsonObj) -> str | None:
    """Rewrite a NAIP STAC item to an ``s3://naip-visualization/`` object URI.

    Derives state/year/GSD/quad/filename from the item; does not consult a
    hardcoded tile list. ``naip-source`` hrefs are rewritten when path parts
    are present, never kept as the persist URI.
    """
    assets = item.get("assets") or {}
    image = assets.get("image") or assets.get("visual") or {}
    href = str(image.get("href") or "")
    if href.startswith(NAIP_VISUALIZATION_BUCKET + "/"):
        return href
    props = item.get("properties") or {}
    state = str(props.get("naip:state") or "").lower()
    year = str(props.get("naip:year") or "")
    gsd = props.get("gsd")
    path = urllib.parse.urlparse(href).path.strip("/")
    parts = [p for p in path.split("/") if p]
    filename = parts[-1] if parts else ""
    quad = parts[-2] if len(parts) >= 2 else ""
    if href.startswith("s3://naip-source/"):
        href = ""
    if not state or not year or not filename or not quad:
        return None
    if gsd is None:
        return None
    gsd_cm = int(round(float(gsd) * 100.0))
    if gsd_cm <= 0:
        return None
    return f"{NAIP_VISUALIZATION_BUCKET}/{state}/{year}/{gsd_cm}cm/rgb/{quad}/{filename}"


def _item_bbox(item: JsonObj) -> list[float] | None:
    bbox = item.get("bbox")
    if bbox and len(bbox) == 4:
        return [float(x) for x in bbox]
    box = item.get("boundingBox") or {}
    if {"minX", "minY", "maxX", "maxY"} <= set(box):
        return [float(box["minX"]), float(box["minY"]), float(box["maxX"]), float(box["maxY"])]
    return None


def search_naip_stac(
    aoi: AoiBox,
    *,
    http_post: HttpPost | None = None,
    url: str = NAIP_STAC_SEARCH_URL,
    limit: int = 200,
    collection: str = NAIP_STAC_COLLECTION,
) -> list[JsonObj]:
    body = {
        "collections": [collection],
        "bbox": list(aoi.bbox),
        "limit": int(limit),
    }
    poster = http_post or default_http_post
    data = poster(url, body)
    features = data.get("features") or data.get("items") or []
    kept = []
    for item in features:
        bbox = _item_bbox(item)
        if bbox is None or not aoi.intersects(bbox):
            continue
        kept.append(item)
    return kept


def search_3dep_inventory(
    aoi: AoiBox,
    *,
    http_get: HttpGet | None = None,
    url: str = TNM_PRODUCTS_URL,
    dataset: str = TNM_3DEP_1M,
    max_items: int = 200,
) -> list[JsonObj]:
    west, south, east, north = aoi.bbox
    query = urllib.parse.urlencode(
        {
            "bbox": f"{west},{south},{east},{north}",
            "datasets": dataset,
            "prodFormats": "GeoTIFF",
            "outputFormat": "JSON",
            "max": str(int(max_items)),
        }
    )
    getter = http_get or default_http_get
    data = getter(f"{url}?{query}")
    items = data.get("items") or []
    kept = []
    for item in items:
        bbox = _item_bbox(item)
        if bbox is None or not aoi.intersects(bbox):
            continue
        kept.append(item)
    return kept


def _naip_records(aoi: AoiBox, items: list[JsonObj]) -> list[JsonObj]:
    records = []
    for item in items:
        vis = visualization_uri_from_stac_item(item)
        href = ((item.get("assets") or {}).get("image") or {}).get("href")
        rejected_source = bool(href) and str(href).startswith("s3://naip-source/")
        records.append(
            {
                "id": item.get("id"),
                "bbox": _item_bbox(item),
                "datetime": (item.get("properties") or {}).get("datetime"),
                "gsd": (item.get("properties") or {}).get("gsd"),
                "naip:state": (item.get("properties") or {}).get("naip:state"),
                "naip:year": (item.get("properties") or {}).get("naip:year"),
                "catalog_href": href,
                "visualization_uri": vis,
                "rejected_naip_source": rejected_source,
            }
        )
    return records


def _threedep_records(items: list[JsonObj]) -> list[JsonObj]:
    records = []
    for item in items:
        urls = item.get("urls") or {}
        download = item.get("downloadURL") or urls.get("TIFF")
        records.append(
            {
                "title": item.get("title"),
                "bbox": _item_bbox(item),
                "publication_date": item.get("publicationDate"),
                "format": item.get("format"),
                "download_url": download,
                "source_id": item.get("sourceId"),
                "extent": item.get("extent"),
            }
        )
    return records


def build_aoi_manifest(
    aoi: AoiBox | None = None,
    *,
    naip_items: list[JsonObj] | None = None,
    threedep_items: list[JsonObj] | None = None,
    http_get: HttpGet | None = None,
    http_post: HttpPost | None = None,
    queried_at: str | None = None,
) -> dict:
    """Intersect the AOI with NAIP visualization STAC and 3DEP inventory."""
    aoi = aoi or golden_morrison_aoi()
    if naip_items is None:
        naip_items = search_naip_stac(aoi, http_post=http_post)
    else:
        naip_items = [it for it in naip_items if (b := _item_bbox(it)) and aoi.intersects(b)]
    if threedep_items is None:
        threedep_items = search_3dep_inventory(aoi, http_get=http_get)
    else:
        threedep_items = [it for it in threedep_items if (b := _item_bbox(it)) and aoi.intersects(b)]
    return {
        "aoi": aoi.as_dict(),
        "queried_at": queried_at or _utcnow(),
        "naip_visualization": {
            "catalog": NAIP_STAC_SEARCH_URL,
            "collection": NAIP_STAC_COLLECTION,
            "persist_bucket": NAIP_VISUALIZATION_BUCKET,
            "n_items": len(naip_items),
            "items": _naip_records(aoi, naip_items),
        },
        "threedep": {
            "inventory": TNM_PRODUCTS_URL,
            "datasets": [TNM_3DEP_1M],
            "n_items": len(threedep_items),
            "items": _threedep_records(threedep_items),
        },
        "persist": {
            "rasters": False,
            "r2": False,
            "naip_source": False,
            "artifacts": ["fsq_codes", "xyz", "compact_metadata"],
        },
        "coverage_note": (
            "This AOI is the $150 rehearsal / first slice inside Colorado. "
            "v1 product coverage remains Colorado-the-state."
        ),
    }


def load_offline_inventory(directory: str | Path) -> tuple[list[JsonObj], list[JsonObj]]:
    directory = Path(directory)
    naip_path = directory / "naip_stac.json"
    tnm_path = directory / "tnm_products.json"
    naip = json.loads(naip_path.read_text())
    tnm = json.loads(tnm_path.read_text())
    naip_items = naip.get("features") or naip.get("items") or []
    tnm_items = tnm.get("items") or tnm
    if isinstance(tnm_items, dict):
        tnm_items = tnm_items.get("items") or []
    return naip_items, tnm_items


def write_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def ingest_aoi_to_path(
    out: str | Path,
    *,
    center: str | None = None,
    size_km: float = 10.0,
    offline: str | Path | None = None,
) -> Path:
    if center:
        lat, lon = parse_center(center)
        aoi = box_from_center(
            lat,
            lon,
            size_km,
            name="custom-aoi",
            role="rehearsal_slice" if size_km <= 10.0 else "custom",
        )
    else:
        aoi = golden_morrison_aoi()
    if offline is not None:
        naip_items, tnm_items = load_offline_inventory(offline)
        manifest = build_aoi_manifest(aoi, naip_items=naip_items, threedep_items=tnm_items)
    else:
        manifest = build_aoi_manifest(aoi)
    return write_manifest(manifest, out)
