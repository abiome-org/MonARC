"""First-slice AOI ingest without AWS billed credentials.

Default source is Microsoft Planetary Computer STAC (collection ``naip``)
with anonymous SAS. Rasters are not copied; the helper writes a JSON
manifest of signed-or-refreshable HREFs and a chip-window plan. Optional
unsigned ``colorado-public-imagery`` listing is a fallback. The AWS
``naip-visualization`` rewrite stays behind an explicit source flag and
never reads AWS shared credentials.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib import error as urllib_error
from urllib import request as urllib_request

from monarc.common.coordinates import (
    GOLDEN_MORRISON_AOI,
    AoiBox,
    box_from_center,
    usgs_75_quads_for_bbox,
    usgs_degree_cell,
    usgs_quarter_bbox,
)
from monarc.data.pc_sas import (
    PC_SAS_SIGN_URL,
    apply_sas_token,
    fetch_sas_token,
    is_azure_blob_href,
    sas_refresh_block,
    sas_token_url,
    unsigned_href,
)
from monarc.map.cog_chips import (
    DEFAULT_CHIP_GRID,
    DEFAULT_CHIP_SIZE_PX,
    DEFAULT_MAX_CHIPS,
    WindowReader,
    chip_plan_block,
    materialize_chip_windows,
    plan_chip_windows,
)

NAIP_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
NAIP_STAC_COLLECTION = "naip"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_VISUALIZATION_BUCKET = "s3://naip-visualization"
TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_3DEP_1M = "Digital Elevation Model (DEM) 1 meter"
TNM_NED_13 = "National Elevation Dataset (NED) 1/3 arc-second"
PC_3DEP_COLLECTION = "3dep-seamless"
COLORADO_PUBLIC_S3 = "s3://colorado-public-imagery"
COLORADO_PUBLIC_HTTPS = "https://colorado-public-imagery.s3.amazonaws.com"
PRD_TNM_S3 = "s3://prd-tnm"
PRD_TNM_HTTPS = "https://prd-tnm.s3.amazonaws.com"

SOURCE_PLANETARY_COMPUTER = "planetary-computer"
SOURCE_COLORADO_PUBLIC = "colorado-public-imagery"
SOURCE_NAIP_VISUALIZATION = "naip-visualization"
DEFAULT_SOURCE = SOURCE_PLANETARY_COMPUTER
INGEST_SOURCES = (
    SOURCE_PLANETARY_COMPUTER,
    SOURCE_COLORADO_PUBLIC,
    SOURCE_NAIP_VISUALIZATION,
)

NAIP_STEM_RE = re.compile(
    r"m_(\d{2})(\d{3})(\d{2})_([nNsS][eEwW])_",
)
S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"

JsonObj = dict[str, Any]
HttpGet = Callable[[str], JsonObj]
HttpGetText = Callable[[str], str]
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
    req = urllib_request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "monarc-aoi-ingest"}
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise InventoryError(f"GET failed for {url}: {exc}") from exc


def default_http_get_text(url: str, timeout: float = 30.0) -> str:
    req = urllib_request.Request(url, headers={"User-Agent": "monarc-aoi-ingest"})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib_error.URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise InventoryError(f"GET failed for {url}: {exc}") from exc


def default_http_post(url: str, body: JsonObj, timeout: float = 30.0) -> JsonObj:
    payload = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
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
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise InventoryError(f"POST failed for {url}: {exc}") from exc


def visualization_uri_from_stac_item(item: JsonObj) -> str | None:
    """Rewrite a NAIP STAC item to an ``s3://naip-visualization/`` object URI.

    Used only for ``--source naip-visualization``. Derives path parts from the
    item; does not consult a hardcoded tile list. ``naip-source`` hrefs are
    rewritten when path parts are present, never kept as the persist URI.
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
    path = urlparse(href).path.strip("/")
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


def _asset_image_href(item: JsonObj) -> str:
    assets = item.get("assets") or {}
    image = assets.get("image") or assets.get("visual") or assets.get("data") or {}
    return str(image.get("href") or item.get("catalog_href") or "")


def _latest_naip_year(items: list[JsonObj]) -> str | None:
    years = []
    for item in items:
        year = (item.get("properties") or {}).get("naip:year")
        if year is None:
            continue
        years.append(str(year))
    if not years:
        return None
    return max(years)


def keep_one_naip_vintage(items: list[JsonObj]) -> list[JsonObj]:
    year = _latest_naip_year(items)
    if year is None:
        return items
    kept = []
    for item in items:
        item_year = (item.get("properties") or {}).get("naip:year")
        if item_year is None or str(item_year) == year:
            kept.append(item)
    return kept


def search_naip_stac(
    aoi: AoiBox,
    *,
    http_post: HttpPost | None = None,
    http_get: HttpGet | None = None,
    url: str = NAIP_STAC_SEARCH_URL,
    limit: int = 200,
    collection: str = NAIP_STAC_COLLECTION,
    max_pages: int = 8,
) -> list[JsonObj]:
    body = {
        "collections": [collection],
        "bbox": list(aoi.bbox),
        "limit": int(limit),
    }
    poster = http_post or default_http_post
    getter = http_get or default_http_get
    data = poster(url, body)
    features: list[JsonObj] = []
    pages = 0
    while data is not None and pages < int(max_pages):
        pages += 1
        batch = data.get("features") or data.get("items") or []
        features.extend(batch)
        next_href = None
        for link in data.get("links") or []:
            if link.get("rel") == "next" and link.get("href"):
                next_href = str(link["href"])
                break
        if not next_href:
            break
        data = getter(next_href)
    kept = []
    for item in features:
        bbox = _item_bbox(item)
        if bbox is None or not aoi.intersects(bbox):
            continue
        kept.append(item)
    return keep_one_naip_vintage(kept)


def search_3dep_inventory(
    aoi: AoiBox,
    *,
    http_get: HttpGet | None = None,
    url: str = TNM_PRODUCTS_URL,
    dataset: str = TNM_3DEP_1M,
    max_items: int = 200,
) -> list[JsonObj]:
    west, south, east, north = aoi.bbox
    query = urlencode(
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


def search_3dep_seamless_stac(
    aoi: AoiBox,
    *,
    http_post: HttpPost | None = None,
    url: str = NAIP_STAC_SEARCH_URL,
    limit: int = 20,
) -> list[JsonObj]:
    return search_naip_stac(
        aoi,
        http_post=http_post,
        url=url,
        limit=limit,
        collection=PC_3DEP_COLLECTION,
        max_pages=2,
    )


def is_requester_pays_prd_tnm(url: str | None) -> bool:
    if not url:
        return False
    raw = str(url).strip()
    if raw.startswith(PRD_TNM_S3):
        return True
    return False


def public_threedep_href(url: str | None) -> str | None:
    """Prefer anonymous HTTPS. Rewrite ``s3://prd-tnm`` to public HTTPS GET."""
    if not url:
        return None
    raw = str(url).strip()
    if raw.startswith(PRD_TNM_S3 + "/"):
        key = raw[len(PRD_TNM_S3) + 1 :]
        return f"{PRD_TNM_HTTPS}/{key}"
    if raw.startswith("s3://"):
        return None
    return raw


def _naip_pc_record(
    item: JsonObj,
    *,
    token: str | None = None,
    expiry: str | None = None,
) -> JsonObj:
    href = unsigned_href(_asset_image_href(item))
    rejected_source = href.startswith("s3://naip-source/")
    props = item.get("properties") or {}
    signed = None
    if token and href and is_azure_blob_href(href) and not rejected_source:
        signed = apply_sas_token(href, token)
    rec: JsonObj = {
        "id": item.get("id"),
        "bbox": _item_bbox(item),
        "datetime": props.get("datetime"),
        "gsd": props.get("gsd"),
        "naip:state": props.get("naip:state"),
        "naip:year": props.get("naip:year"),
        "proj:epsg": props.get("proj:epsg"),
        "proj:shape": props.get("proj:shape"),
        "proj:transform": props.get("proj:transform"),
        "catalog_href": href or None,
        "signed_href": signed,
        "sas_expiry": expiry if signed else None,
        "sas": sas_refresh_block(NAIP_STAC_COLLECTION) if is_azure_blob_href(href) else None,
        "rejected_naip_source": rejected_source,
    }
    return rec


def _naip_visualization_record(item: JsonObj) -> JsonObj:
    vis = visualization_uri_from_stac_item(item)
    href = _asset_image_href(item)
    rejected_source = bool(href) and str(href).startswith("s3://naip-source/")
    props = item.get("properties") or {}
    return {
        "id": item.get("id"),
        "bbox": _item_bbox(item),
        "datetime": props.get("datetime"),
        "gsd": props.get("gsd"),
        "naip:state": props.get("naip:state"),
        "naip:year": props.get("naip:year"),
        "proj:epsg": props.get("proj:epsg"),
        "proj:shape": props.get("proj:shape"),
        "proj:transform": props.get("proj:transform"),
        "catalog_href": href or None,
        "visualization_uri": vis,
        "signed_href": None,
        "rejected_naip_source": rejected_source,
    }


def _threedep_records(items: list[JsonObj]) -> list[JsonObj]:
    records = []
    for item in items:
        urls = item.get("urls") or {}
        download = item.get("downloadURL") or urls.get("TIFF")
        public = public_threedep_href(download)
        rejected = is_requester_pays_prd_tnm(download) and public is None
        records.append(
            {
                "title": item.get("title"),
                "bbox": _item_bbox(item),
                "publication_date": item.get("publicationDate"),
                "format": item.get("format"),
                "download_url": public,
                "source_id": item.get("sourceId"),
                "extent": item.get("extent"),
                "requester_pays": bool(rejected),
                "rejected_prd_tnm_requester_pays": bool(rejected),
                "access": "rejected-requester-pays" if rejected else "public-https",
            }
        )
    return records


def _threedep_pc_records(
    items: list[JsonObj],
    *,
    token: str | None = None,
    expiry: str | None = None,
) -> list[JsonObj]:
    records = []
    for item in items:
        href = unsigned_href(_asset_image_href(item))
        signed = None
        if token and href and is_azure_blob_href(href):
            signed = apply_sas_token(href, token)
        props = item.get("properties") or {}
        records.append(
            {
                "id": item.get("id"),
                "bbox": _item_bbox(item),
                "gsd": props.get("gsd"),
                "catalog_href": href or None,
                "signed_href": signed,
                "sas_expiry": expiry if signed else None,
                "sas": sas_refresh_block(PC_3DEP_COLLECTION) if is_azure_blob_href(href) else None,
            }
        )
    return records


def parse_naip_filename(name: str) -> dict[str, Any] | None:
    stem = Path(name).name
    match = NAIP_STEM_RE.search(stem)
    if match is None:
        return None
    north = int(match.group(1))
    west = int(match.group(2))
    qq = int(match.group(3))
    corner = match.group(4).lower()
    return {
        "north_deg": north,
        "west_abs_deg": west,
        "qq": qq,
        "corner": corner,
        "bbox": list(usgs_quarter_bbox(north, west, qq, corner)),
    }


def _s3_text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text


def parse_s3_list_xml(xml_text: str) -> tuple[list[str], str | None, bool]:
    """Parse a ListBucketResult. Returns (keys, continuation_token, truncated)."""
    root = ET.fromstring(xml_text)
    ns = {"s3": S3_NS} if root.tag.startswith("{") else {}
    def find(tag: str) -> ET.Element | None:
        if ns:
            return root.find(f"s3:{tag}", ns)
        return root.find(tag)

    def findall(tag: str) -> list[ET.Element]:
        if ns:
            return list(root.findall(f"s3:{tag}", ns))
        return list(root.findall(tag))

    keys = []
    for contents in findall("Contents"):
        key_el = contents.find(f"{{{S3_NS}}}Key") if ns else contents.find("Key")
        key = _s3_text(key_el)
        if key and not key.endswith("/"):
            keys.append(key)
    truncated_el = find("IsTruncated")
    truncated = _s3_text(truncated_el).lower() == "true"
    token_el = find("NextContinuationToken")
    token = _s3_text(token_el) or None
    return keys, token, truncated


def colorado_public_list_url(prefix: str, continuation: str | None = None, max_keys: int = 1000) -> str:
    query = {
        "list-type": "2",
        "prefix": prefix,
        "max-keys": str(int(max_keys)),
    }
    if continuation:
        query["continuation-token"] = continuation
    return f"{COLORADO_PUBLIC_HTTPS}/?{urlencode(query)}"


def colorado_public_object_href(key: str) -> str:
    return f"{COLORADO_PUBLIC_HTTPS}/{key.lstrip('/')}"


def list_colorado_public_cogs(
    aoi: AoiBox,
    *,
    http_get_text: HttpGetText | None = None,
    keys: list[str] | None = None,
    year: int | None = None,
    max_pages: int = 8,
) -> list[JsonObj]:
    """Unsigned HTTPS list of COG keys intersecting the AOI. No AWS signature."""
    if keys is None:
        getter = http_get_text or default_http_get_text
        north, west = usgs_degree_cell(aoi.center_lat, aoi.center_lon)
        vintage = int(year) if year is not None else 2021
        prefix = f"NAIP/NAIP{vintage}/cogs/m_{north}{west:03d}"
        collected: list[str] = []
        continuation = None
        for _ in range(int(max_pages)):
            xml_text = getter(colorado_public_list_url(prefix, continuation))
            page_keys, continuation, truncated = parse_s3_list_xml(xml_text)
            collected.extend(page_keys)
            if not truncated or not continuation:
                break
        keys = collected
    wanted = {cell for cell in usgs_75_quads_for_bbox(aoi.bbox)}
    records = []
    for key in keys:
        parsed = parse_naip_filename(key)
        if parsed is None:
            continue
        cell = (parsed["north_deg"], parsed["west_abs_deg"], parsed["qq"])
        if cell not in wanted:
            continue
        if not aoi.intersects(parsed["bbox"]):
            continue
        href = key if key.startswith("http") else colorado_public_object_href(key)
        records.append(
            {
                "id": Path(key).stem,
                "bbox": parsed["bbox"],
                "catalog_href": href,
                "signed_href": href,
                "gsd": 0.6,
                "naip:state": "co",
                "naip:year": str(year) if year is not None else None,
                "access": "unsigned-https",
                "s3_uri": f"{COLORADO_PUBLIC_S3}/{key.lstrip('/')}" if not key.startswith("http") else None,
            }
        )
    return records


def _sign_collection(
    collection: str,
    *,
    sas_token: str | None,
    sas_expiry: str | None,
    http_get: HttpGet | None,
    do_sign: bool,
) -> tuple[str | None, str | None]:
    if not do_sign:
        return sas_token, sas_expiry
    if sas_token and sas_expiry:
        return sas_token, sas_expiry
    if http_get is None:
        return sas_token, sas_expiry
    return fetch_sas_token(collection, http_get=http_get)


def _naip_items_for_source(
    aoi: AoiBox,
    source: str,
    *,
    naip_items: list[JsonObj] | None,
    colorado_keys: list[str] | None,
    http_get: HttpGet | None,
    http_post: HttpPost | None,
    http_get_text: HttpGetText | None,
) -> list[JsonObj]:
    if source == SOURCE_COLORADO_PUBLIC:
        if naip_items is not None:
            return [it for it in naip_items if (b := _item_bbox(it)) and aoi.intersects(b)]
        return list_colorado_public_cogs(
            aoi, http_get_text=http_get_text, keys=colorado_keys
        )
    if naip_items is None:
        return search_naip_stac(aoi, http_post=http_post, http_get=http_get)
    return [it for it in naip_items if (b := _item_bbox(it)) and aoi.intersects(b)]


def build_aoi_manifest(
    aoi: AoiBox | None = None,
    *,
    source: str = DEFAULT_SOURCE,
    naip_items: list[JsonObj] | None = None,
    threedep_items: list[JsonObj] | None = None,
    threedep_pc_items: list[JsonObj] | None = None,
    colorado_keys: list[str] | None = None,
    http_get: HttpGet | None = None,
    http_get_text: HttpGetText | None = None,
    http_post: HttpPost | None = None,
    queried_at: str | None = None,
    sas_token: str | None = None,
    sas_expiry: str | None = None,
    sas_token_3dep: str | None = None,
    sas_expiry_3dep: str | None = None,
    sign: bool | None = None,
    chip_size: int = DEFAULT_CHIP_SIZE_PX,
    chip_grid: int = DEFAULT_CHIP_GRID,
    max_chips: int = DEFAULT_MAX_CHIPS,
) -> dict:
    """Intersect the AOI with NAIP and public 3DEP. Default source needs no AWS."""
    if source not in INGEST_SOURCES:
        raise ValueError(f"unknown ingest source {source!r}")
    aoi = aoi or golden_morrison_aoi()
    live = naip_items is None and source != SOURCE_COLORADO_PUBLIC
    live_co = naip_items is None and colorado_keys is None and source == SOURCE_COLORADO_PUBLIC
    do_sign = bool(sign) if sign is not None else bool(
        sas_token or live or (source == SOURCE_PLANETARY_COMPUTER and sas_token)
    )
    if source == SOURCE_PLANETARY_COMPUTER and sign is None:
        do_sign = sas_token is not None or live

    naip_items = _naip_items_for_source(
        aoi,
        source,
        naip_items=naip_items,
        colorado_keys=colorado_keys,
        http_get=http_get,
        http_post=http_post,
        http_get_text=http_get_text,
    )
    if source != SOURCE_COLORADO_PUBLIC:
        naip_items = keep_one_naip_vintage(naip_items)

    if threedep_items is None:
        if live or live_co:
            threedep_items = search_3dep_inventory(aoi, http_get=http_get)
        else:
            threedep_items = []
    else:
        threedep_items = [it for it in threedep_items if (b := _item_bbox(it)) and aoi.intersects(b)]

    if threedep_pc_items is None and source == SOURCE_PLANETARY_COMPUTER and live:
        threedep_pc_items = search_3dep_seamless_stac(aoi, http_post=http_post)
    elif threedep_pc_items is None:
        threedep_pc_items = []
    else:
        threedep_pc_items = [
            it for it in threedep_pc_items if (b := _item_bbox(it)) and aoi.intersects(b)
        ]

    token = expiry = None
    sign_http = (http_get or default_http_get) if (do_sign and live) else None
    if source == SOURCE_PLANETARY_COMPUTER:
        token, expiry = _sign_collection(
            NAIP_STAC_COLLECTION,
            sas_token=sas_token,
            sas_expiry=sas_expiry,
            http_get=sign_http,
            do_sign=do_sign,
        )
        naip_records = [_naip_pc_record(it, token=token, expiry=expiry) for it in naip_items]
    elif source == SOURCE_NAIP_VISUALIZATION:
        naip_records = [_naip_visualization_record(it) for it in naip_items]
    else:
        naip_records = []
        for it in naip_items:
            if it.get("catalog_href") or it.get("assets"):
                rec = {
                    "id": it.get("id"),
                    "bbox": _item_bbox(it) or it.get("bbox"),
                    "datetime": (it.get("properties") or {}).get("datetime"),
                    "gsd": it.get("gsd") or (it.get("properties") or {}).get("gsd"),
                    "naip:state": it.get("naip:state") or (it.get("properties") or {}).get("naip:state"),
                    "naip:year": it.get("naip:year") or (it.get("properties") or {}).get("naip:year"),
                    "catalog_href": it.get("catalog_href") or _asset_image_href(it),
                    "signed_href": it.get("signed_href") or it.get("catalog_href") or _asset_image_href(it),
                    "access": it.get("access") or "unsigned-https",
                    "s3_uri": it.get("s3_uri"),
                    "rejected_naip_source": False,
                }
                naip_records.append(rec)
            else:
                naip_records.append(it)

    token_3dep = expiry_3dep = None
    if source == SOURCE_PLANETARY_COMPUTER and threedep_pc_items:
        token_3dep, expiry_3dep = _sign_collection(
            PC_3DEP_COLLECTION,
            sas_token=sas_token_3dep,
            sas_expiry=sas_expiry_3dep,
            http_get=sign_http,
            do_sign=do_sign,
        )

    chip_items = []
    for rec in naip_records:
        chip_items.append(
            {
                "id": rec.get("id"),
                "bbox": rec.get("bbox"),
                "catalog_href": rec.get("catalog_href") or rec.get("visualization_uri"),
                "signed_href": rec.get("signed_href") or rec.get("visualization_uri"),
                "sas": rec.get("sas"),
                "sas_expiry": rec.get("sas_expiry"),
                "properties": {
                    "proj:epsg": rec.get("proj:epsg"),
                    "proj:shape": rec.get("proj:shape"),
                    "proj:transform": rec.get("proj:transform"),
                    "gsd": rec.get("gsd"),
                },
                "proj:epsg": rec.get("proj:epsg"),
                "proj:shape": rec.get("proj:shape"),
                "proj:transform": rec.get("proj:transform"),
            }
        )
    windows = plan_chip_windows(
        aoi, chip_items, size_px=chip_size, grid=chip_grid, max_chips=max_chips
    )

    aws_required = source == SOURCE_NAIP_VISUALIZATION
    persist = {
        "rasters": False,
        "r2": False,
        "r2_rasters": False,
        "full_geotiff": False,
        "naip_source": False,
        "artifacts": ["fsq_codes", "xyz", "compact_metadata"],
    }
    credentials = {
        "aws_required": aws_required,
        "reads_dot_aws": False,
        "anonymous": not aws_required,
        "sas_token_url": sas_token_url(NAIP_STAC_COLLECTION)
        if source == SOURCE_PLANETARY_COMPUTER
        else None,
        "sas_sign_url": PC_SAS_SIGN_URL if source == SOURCE_PLANETARY_COMPUTER else None,
    }
    naip_block = {
        "source": source,
        "catalog": NAIP_STAC_SEARCH_URL
        if source != SOURCE_COLORADO_PUBLIC
        else COLORADO_PUBLIC_HTTPS,
        "collection": NAIP_STAC_COLLECTION if source != SOURCE_COLORADO_PUBLIC else None,
        "n_items": len(naip_records),
        "items": naip_records,
        "one_vintage": True,
    }
    manifest = {
        "aoi": aoi.as_dict(),
        "queried_at": queried_at or _utcnow(),
        "source": source,
        "credentials": credentials,
        "naip": naip_block,
        "threedep": {
            "inventory": TNM_PRODUCTS_URL,
            "datasets": [TNM_3DEP_1M],
            "n_items": len(threedep_items),
            "items": _threedep_records(threedep_items),
            "requester_pays_prd_tnm": False,
        },
        "threedep_pc": {
            "catalog": NAIP_STAC_SEARCH_URL,
            "collection": PC_3DEP_COLLECTION,
            "n_items": len(threedep_pc_items),
            "items": _threedep_pc_records(
                threedep_pc_items, token=token_3dep, expiry=expiry_3dep
            ),
        },
        "chip_extract": chip_plan_block(
            windows, size_px=chip_size, grid=chip_grid, max_chips=max_chips
        ),
        "persist": persist,
        "coverage_note": (
            "This AOI is the $150 rehearsal / first slice inside Colorado. "
            "v1 product coverage remains Colorado-the-state. "
            "Cloudflare R2 stores codes+index only; do not copy GeoTIFFs to R2."
        ),
    }
    if source == SOURCE_NAIP_VISUALIZATION:
        manifest["naip_visualization"] = {
            "catalog": NAIP_STAC_SEARCH_URL,
            "collection": NAIP_STAC_COLLECTION,
            "persist_bucket": NAIP_VISUALIZATION_BUCKET,
            "n_items": len(naip_records),
            "items": naip_records,
        }
    else:
        manifest["naip_visualization"] = {
            "catalog": NAIP_STAC_SEARCH_URL,
            "collection": NAIP_STAC_COLLECTION,
            "persist_bucket": None,
            "n_items": 0,
            "items": [],
            "skipped": True,
            "reason": "naip-visualization requires an explicit --source flag and a billed AWS path",
        }
    if source == SOURCE_COLORADO_PUBLIC:
        manifest["colorado_public_imagery"] = {
            "bucket": COLORADO_PUBLIC_S3,
            "https": COLORADO_PUBLIC_HTTPS,
            "unsigned": True,
            "no_sign_request": True,
            "n_items": len(naip_records),
        }
    return manifest


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


def load_offline_sas(directory: str | Path) -> tuple[str | None, str | None]:
    path = Path(directory) / "sas_token.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text())
    return payload.get("token"), payload.get("msft:expiry")


def load_offline_colorado_keys(directory: str | Path) -> list[str] | None:
    path = Path(directory) / "colorado_list.xml"
    if not path.is_file():
        return None
    keys, _token, _trunc = parse_s3_list_xml(path.read_text())
    return keys


def load_offline_3dep_pc(directory: str | Path) -> list[JsonObj]:
    path = Path(directory) / "3dep_pc.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    return payload.get("features") or payload.get("items") or []


def write_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def aoi_from_args(center: str | None, size_km: float) -> AoiBox:
    if center:
        lat, lon = parse_center(center)
        return box_from_center(
            lat,
            lon,
            size_km,
            name="custom-aoi",
            role="rehearsal_slice" if size_km <= 10.0 else "custom",
        )
    return golden_morrison_aoi()


def refresh_naip_signatures(
    records: list[JsonObj],
    *,
    http_get: HttpGet,
    collection: str = NAIP_STAC_COLLECTION,
) -> list[JsonObj]:
    token, expiry = fetch_sas_token(collection, http_get=http_get)
    out = []
    for rec in records:
        href = rec.get("catalog_href")
        if not href or not is_azure_blob_href(str(href)):
            out.append(rec)
            continue
        updated = dict(rec)
        updated["signed_href"] = apply_sas_token(str(href), token)
        updated["sas_expiry"] = expiry
        updated["sas"] = sas_refresh_block(collection)
        out.append(updated)
    return out


def ingest_aoi_to_path(
    out: str | Path,
    *,
    center: str | None = None,
    size_km: float = 10.0,
    offline: str | Path | None = None,
    source: str = DEFAULT_SOURCE,
    chips_dir: str | Path | None = None,
    chip_size: int = DEFAULT_CHIP_SIZE_PX,
    chip_grid: int = DEFAULT_CHIP_GRID,
    max_chips: int = DEFAULT_MAX_CHIPS,
    window_reader: WindowReader | None = None,
    materialize_only: bool = False,
    http_get: HttpGet | None = None,
    http_post: HttpPost | None = None,
    http_get_text: HttpGetText | None = None,
) -> Path:
    out_path = Path(out)
    if materialize_only:
        manifest = json.loads(out_path.read_text())
        if chips_dir is None:
            raise ValueError("--chips is required with --materialize-only")
        windows = (manifest.get("chip_extract") or {}).get("windows") or []
        if source == SOURCE_PLANETARY_COMPUTER and http_get is not None:
            windows = refresh_naip_signatures(windows, http_get=http_get)
            manifest["chip_extract"]["windows"] = windows
            write_manifest(manifest, out_path)
        materialize_chip_windows(windows, chips_dir, reader=window_reader)
        return out_path

    aoi = aoi_from_args(center, size_km)
    if offline is not None:
        naip_items, tnm_items = load_offline_inventory(offline)
        token, expiry = load_offline_sas(offline)
        colorado_keys = load_offline_colorado_keys(offline)
        threedep_pc = load_offline_3dep_pc(offline)
        if source == SOURCE_COLORADO_PUBLIC and colorado_keys is not None:
            naip_arg = None
        else:
            naip_arg = naip_items
        manifest = build_aoi_manifest(
            aoi,
            source=source,
            naip_items=naip_arg,
            threedep_items=tnm_items,
            threedep_pc_items=threedep_pc or None,
            colorado_keys=colorado_keys,
            sas_token=token,
            sas_expiry=expiry,
            sign=token is not None,
            chip_size=chip_size,
            chip_grid=chip_grid,
            max_chips=max_chips,
            http_get=http_get,
            http_post=http_post,
            http_get_text=http_get_text,
        )
    else:
        manifest = build_aoi_manifest(
            aoi,
            source=source,
            chip_size=chip_size,
            chip_grid=chip_grid,
            max_chips=max_chips,
            http_get=http_get,
            http_post=http_post,
            http_get_text=http_get_text,
        )
    path = write_manifest(manifest, out_path)
    if chips_dir is not None:
        windows = (manifest.get("chip_extract") or {}).get("windows") or []
        materialize_chip_windows(windows, chips_dir, reader=window_reader)
    return path
