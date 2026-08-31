"""Colorado-track chip retrieval: bag-of-codes plus frozen DINO on a spatial holdout.

Loads local ``extract`` + ``fsq`` arrays. No network. Query chips come from
a geographically disjoint box (high side of the longest east/north span),
not a random sample of neighboring chips. Rank-1 xyz error is the retrieved
gallery chip versus the query chip. Recall@K is a hit when the spatially
nearest gallery chip is in the top K.

Bag-of-codes is the FSQ baseline. When ``features.npy`` is present, the same
split is also scored with frozen DINO pooled cosine (mean over the feature
grid) and flattened-grid cosine. Top-level Recall@K / xyz error remain the
bag-of-codes numbers.

Numbers are this chip set only. They are not University-1652 Recall@1 and
not Colorado GPS-denied flight ATE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from monarc.localization.global_retrieve import (
    BAG_OF_CODES_DESCRIPTOR,
    DINO_GRID_DESCRIPTOR,
    DINO_POOLED_DESCRIPTOR,
    FEATURE_POOL_FLATTEN,
    FEATURE_POOL_MEAN,
    CodeRetriever,
    FeatureRetriever,
)

TINY_N_CHIPS = 128
TINY_N_QUERY = 32
AXIS_EAST = 0
AXIS_NORTH = 1
AXIS_NAMES = ("east", "north")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _as_chip_codes(codes: np.ndarray) -> np.ndarray:
    arr = np.asarray(codes, dtype=np.int64)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"codes must be [N], [N, T], or [N, H, W], got {arr.shape}")


def _as_xyz(xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"xyz must be [N, 3], got {arr.shape}")
    return arr


def _as_chip_features(features: np.ndarray, n_chips: int) -> np.ndarray:
    arr = np.asarray(features)
    if arr.shape[0] != n_chips:
        raise ValueError(f"features chip count {arr.shape[0]} != {n_chips}")
    if arr.ndim not in (2, 3, 4):
        raise ValueError(
            f"features must be [N, D], [N, C, T], or [N, C, H, W], got {arr.shape}"
        )
    return arr


def infer_codebook_size(codes: np.ndarray, meta: dict[str, Any] | None = None) -> int:
    meta = meta or {}
    observed = int(np.max(codes)) + 1 if codes.size else 1
    declared = meta.get("codebook_size")
    if declared is None:
        return max(observed, 1)
    size = int(declared)
    if size < observed:
        raise ValueError(f"meta codebook_size {size} < max(code)+1 {observed}")
    return size


def load_retrieve_inputs(
    extract_dir: str | Path,
    fsq_dir: str | Path,
) -> dict[str, Any]:
    """Load ``features.npy`` (extract) + ``codes.npy`` / ``xyz.npy`` (fsq)."""
    extract_dir = Path(extract_dir)
    fsq_dir = Path(fsq_dir)
    features_path = extract_dir / "features.npy"
    codes_path = fsq_dir / "codes.npy"
    if not features_path.is_file():
        raise FileNotFoundError(f"missing extract features: {features_path}")
    if not codes_path.is_file():
        raise FileNotFoundError(f"missing fsq codes: {codes_path}")
    features = np.load(features_path)
    codes = _as_chip_codes(np.load(codes_path))
    xyz_fsq = fsq_dir / "xyz.npy"
    xyz_extract = extract_dir / "xyz.npy"
    if xyz_fsq.is_file() and xyz_extract.is_file():
        xyz_a = _as_xyz(np.load(xyz_fsq))
        xyz_b = _as_xyz(np.load(xyz_extract))
        if xyz_a.shape != xyz_b.shape or not np.allclose(xyz_a, xyz_b, equal_nan=True):
            raise ValueError("extract xyz.npy and fsq xyz.npy do not match")
        xyz = xyz_a
        xyz_source = "fsq"
    elif xyz_fsq.is_file():
        xyz = _as_xyz(np.load(xyz_fsq))
        xyz_source = "fsq"
    elif xyz_extract.is_file():
        xyz = _as_xyz(np.load(xyz_extract))
        xyz_source = "extract"
    else:
        raise FileNotFoundError(
            f"missing xyz.npy in {fsq_dir} and {extract_dir}"
        )
    n_feat = int(features.shape[0])
    if codes.shape[0] != n_feat or xyz.shape[0] != n_feat:
        raise ValueError(
            f"chip count mismatch: features {n_feat}, codes {codes.shape[0]}, xyz {xyz.shape[0]}"
        )
    ids_payload = json.loads((extract_dir / "ids.json").read_text()) if (extract_dir / "ids.json").is_file() else []
    if isinstance(ids_payload, list) and len(ids_payload) == n_feat:
        ids = [str(x) for x in ids_payload]
    else:
        ids = [f"chip-{i:04d}" for i in range(n_feat)]
    meta = _read_json(fsq_dir / "meta.json")
    extract_meta = _read_json(extract_dir / "meta.json")
    codebook_size = infer_codebook_size(codes, meta)
    return {
        "features": features,
        "codes": codes,
        "xyz": xyz,
        "ids": ids,
        "codebook_size": codebook_size,
        "meta": meta,
        "extract_meta": extract_meta,
        "xyz_source": xyz_source,
        "extract_dir": str(extract_dir),
        "fsq_dir": str(fsq_dir),
    }


def choose_split_axis(xyz: np.ndarray, axis: str | int | None = "auto") -> int:
    if axis in (None, "auto"):
        span = np.nanmax(xyz[:, :2], axis=0) - np.nanmin(xyz[:, :2], axis=0)
        if not np.isfinite(span).any():
            raise ValueError("xyz east/north are not finite")
        if span[AXIS_EAST] >= span[AXIS_NORTH]:
            return AXIS_EAST
        return AXIS_NORTH
    if axis in ("east", 0, "0"):
        return AXIS_EAST
    if axis in ("north", 1, "1"):
        return AXIS_NORTH
    raise ValueError(f"axis must be auto, east, or north, got {axis!r}")


def spatial_holdout_indices(
    xyz: np.ndarray,
    *,
    query_fraction: float = 0.25,
    axis: str | int | None = "auto",
) -> dict[str, Any]:
    """Hold out a geographic box on the high side of one ENU axis.

    Chips are not sampled at random. Ties on the cut share the query side so
    the held-out set is a closed half-space, not scattered neighbors.
    """
    xyz = _as_xyz(xyz)
    if xyz.shape[0] < 2:
        raise ValueError("need at least 2 chips for a spatial split")
    if not np.isfinite(xyz[:, :2]).all():
        raise ValueError("spatial holdout needs finite east/north xyz")
    frac = float(query_fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError("query_fraction must be in (0, 1)")
    axis_i = choose_split_axis(xyz, axis)
    values = xyz[:, axis_i]
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError(
            f"xyz has no extent on {AXIS_NAMES[axis_i]}; cannot form a spatial box"
        )
    n_cut = int(round(unique.size * frac))
    n_cut = min(max(1, n_cut), unique.size - 1)
    threshold = float(unique[-n_cut])
    query_mask = values >= threshold
    query_idx = np.flatnonzero(query_mask).astype(np.int64)
    gallery_idx = np.flatnonzero(~query_mask).astype(np.int64)
    if query_idx.size == 0 or gallery_idx.size == 0:
        raise ValueError("spatial split produced an empty gallery or query set")
    query_vals = values[query_idx]
    gallery_vals = values[gallery_idx]
    disjoint_box = bool(np.min(query_vals) >= np.max(gallery_vals))
    return {
        "axis": AXIS_NAMES[axis_i],
        "axis_index": axis_i,
        "query_fraction": frac,
        "threshold": threshold,
        "query_idx": query_idx,
        "gallery_idx": gallery_idx,
        "kind": "spatial-box",
        "disjoint_box": disjoint_box,
    }


def chip_distance(a: np.ndarray, b: np.ndarray, *, use_3d: bool) -> float:
    delta = np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)
    if use_3d:
        return float(np.linalg.norm(delta))
    return float(np.linalg.norm(delta[:2]))


def nearest_gallery_indices(
    query_xyz: np.ndarray,
    gallery_xyz: np.ndarray,
    *,
    use_3d: bool,
) -> np.ndarray:
    if use_3d:
        delta = gallery_xyz - query_xyz.reshape(1, 3)
        dist = np.linalg.norm(delta, axis=1)
    else:
        delta = gallery_xyz[:, :2] - query_xyz.reshape(3)[:2]
        dist = np.linalg.norm(delta, axis=1)
    mind = float(np.min(dist))
    return np.flatnonzero(np.isclose(dist, mind, rtol=0.0, atol=1e-6)).astype(np.int64)


def percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _tiny_reason(n_chips: int, n_query: int, n_gallery: int) -> str | None:
    reasons = []
    if n_chips < TINY_N_CHIPS:
        reasons.append(f"n_chips={n_chips} < {TINY_N_CHIPS}")
    if n_query < TINY_N_QUERY:
        reasons.append(f"n_query={n_query} < {TINY_N_QUERY}")
    if n_gallery < 5:
        reasons.append(f"n_gallery={n_gallery} < 5 (Recall@5 uses k=min(5, n_gallery))")
    if not reasons:
        return None
    return "; ".join(reasons)


def _score_ranked_queries(
    ranked_per_query: list[list[tuple[str, float]]],
    query_idx: np.ndarray,
    ids: list[str],
    xyz: np.ndarray,
    gallery_xyz: np.ndarray,
    gallery_ids: list[str],
    gt_ids_per_query: list[list[str]],
    oracle_err: list[float],
    ks: tuple[int, ...],
    use_3d: bool,
) -> dict[str, Any]:
    n_query = int(query_idx.size)
    hits = {int(k): 0 for k in ks}
    rank1_err: list[float] = []
    per_query: list[dict[str, Any]] = []
    for qi, ranked, gt_ids, oracle in zip(
        query_idx.tolist(), ranked_per_query, gt_ids_per_query, oracle_err, strict=True
    ):
        qi = int(qi)
        ranked_ids = [doc for doc, _score in ranked]
        rank1_id = ranked_ids[0] if ranked_ids else None
        if rank1_id is None:
            err = float("nan")
        else:
            gpos = gallery_ids.index(rank1_id)
            err = chip_distance(xyz[qi], gallery_xyz[gpos], use_3d=use_3d)
        rank1_err.append(err)
        row_hits: dict[str, bool] = {}
        for k in ks:
            hit = bool(set(ranked_ids[: int(k)]) & set(gt_ids))
            row_hits[f"hit_at_{int(k)}"] = hit
            if hit:
                hits[int(k)] += 1
        per_query.append(
            {
                "query_id": ids[qi],
                "rank1_id": rank1_id,
                "rank1_score": float(ranked[0][1]) if ranked else None,
                "xyz_error_m": err,
                "oracle_xyz_m": float(oracle),
                "gt_ids": gt_ids,
                **row_hits,
            }
        )
    recall = {
        f"recall_at_{int(k)}": (hits[int(k)] / n_query) if n_query else float("nan")
        for k in ks
    }
    return {
        **recall,
        "median_xyz_error_m": percentile(np.asarray(rank1_err), 50.0),
        "p90_xyz_error_m": percentile(np.asarray(rank1_err), 90.0),
        "queries": per_query,
    }


def _bag_rankings(
    codes: np.ndarray,
    query_idx: np.ndarray,
    gallery_idx: np.ndarray,
    ids: list[str],
    codebook_size: int,
    k_cap: int,
) -> list[list[tuple[str, float]]]:
    retriever = CodeRetriever(codebook_size=int(codebook_size))
    for gi in gallery_idx.tolist():
        retriever.add(ids[int(gi)], codes[int(gi)])
    return [retriever.query(codes[int(qi)], k=k_cap) for qi in query_idx.tolist()]


def _feature_rankings(
    features: np.ndarray,
    query_idx: np.ndarray,
    gallery_idx: np.ndarray,
    ids: list[str],
    k_cap: int,
    pool: str,
) -> list[list[tuple[str, float]]]:
    gallery_ids = [ids[int(i)] for i in gallery_idx.tolist()]
    retriever = FeatureRetriever.from_batch(gallery_ids, features[gallery_idx], pool=pool)
    return [retriever.query(features[int(qi)], k=k_cap) for qi in query_idx.tolist()]


def evaluate_chip_retrieve(
    codes: np.ndarray,
    xyz: np.ndarray,
    *,
    codebook_size: int,
    ids: list[str] | None = None,
    query_fraction: float = 0.25,
    axis: str | int | None = "auto",
    ks: tuple[int, ...] = (1, 5),
    features: np.ndarray | None = None,
) -> dict[str, Any]:
    """Chip retrieve on a spatial holdout. CPU, no network.

    Top-level Recall@K and xyz error are bag-of-codes. When ``features`` is
    set, the same split is also scored with frozen DINO cosine (pooled and
    flattened grid) under ``modes``.
    """
    codes = _as_chip_codes(codes)
    xyz = _as_xyz(xyz)
    n = int(codes.shape[0])
    if xyz.shape[0] != n:
        raise ValueError("codes and xyz chip counts differ")
    if ids is None or len(ids) != n:
        ids = [f"chip-{i:04d}" for i in range(n)]
    feat = None if features is None else _as_chip_features(features, n)
    split = spatial_holdout_indices(xyz, query_fraction=query_fraction, axis=axis)
    query_idx = split["query_idx"]
    gallery_idx = split["gallery_idx"]
    use_3d = bool(np.isfinite(xyz[:, 2]).all())
    max_k = max(int(k) for k in ks)
    k_cap = min(max_k, int(gallery_idx.size))
    gallery_xyz = xyz[gallery_idx]
    gallery_ids = [ids[int(i)] for i in gallery_idx.tolist()]
    oracle_err: list[float] = []
    gt_ids_per_query: list[list[str]] = []
    for qi in query_idx.tolist():
        qi = int(qi)
        nearest_local = nearest_gallery_indices(xyz[qi], gallery_xyz, use_3d=use_3d)
        gt_ids = [gallery_ids[int(j)] for j in nearest_local.tolist()]
        oracle = min(
            chip_distance(xyz[qi], gallery_xyz[int(j)], use_3d=use_3d) for j in nearest_local.tolist()
        )
        oracle_err.append(float(oracle))
        gt_ids_per_query.append(gt_ids)
    score_kw: dict[str, Any] = {
        "query_idx": query_idx,
        "ids": ids,
        "xyz": xyz,
        "gallery_xyz": gallery_xyz,
        "gallery_ids": gallery_ids,
        "gt_ids_per_query": gt_ids_per_query,
        "oracle_err": oracle_err,
        "ks": ks,
        "use_3d": use_3d,
    }
    bag = _score_ranked_queries(
        _bag_rankings(codes, query_idx, gallery_idx, ids, codebook_size, k_cap),
        **score_kw,
    )
    modes: dict[str, Any] = {
        BAG_OF_CODES_DESCRIPTOR: {
            "descriptor": BAG_OF_CODES_DESCRIPTOR,
            "features_used": False,
            **bag,
        }
    }
    descriptors = [BAG_OF_CODES_DESCRIPTOR]
    if feat is not None:
        feat_shape = [int(x) for x in feat.shape]
        for descriptor, pool in (
            (DINO_POOLED_DESCRIPTOR, FEATURE_POOL_MEAN),
            (DINO_GRID_DESCRIPTOR, FEATURE_POOL_FLATTEN),
        ):
            scored = _score_ranked_queries(
                _feature_rankings(feat, query_idx, gallery_idx, ids, k_cap, pool),
                **score_kw,
            )
            modes[descriptor] = {
                "descriptor": descriptor,
                "features_used": True,
                "pool": pool,
                "feature_shape": feat_shape,
                **scored,
            }
            descriptors.append(descriptor)
    n_query = int(query_idx.size)
    n_gallery = int(gallery_idx.size)
    tiny_reason = _tiny_reason(n, n_query, n_gallery)
    unique_codes = int(np.unique(codes).size)
    features_used = feat is not None
    if features_used:
        retrieve_note = (
            "Bag-of-codes is the FSQ baseline. Frozen DINO pooled cosine and "
            "flattened-grid cosine are scored from extract features.npy on the "
            "same spatial split."
        )
        protocol = "chip retrieve on spatial holdout (bag-of-codes + frozen DINO)"
    else:
        retrieve_note = (
            "Retrieval is bag-of-codes; frozen DINO features were not provided."
        )
        protocol = "bag-of-codes chip retrieve on spatial holdout"
    note = (
        "Numbers are from this chip set and this spatial split only. "
        "They are not University-1652 Recall@1, not OrthoLoC translation, "
        "and not Colorado GPS-denied flight ATE. "
        f"{retrieve_note}"
    )
    if tiny_reason:
        note = (
            f"Split is tiny ({tiny_reason}). Do not treat these figures as "
            f"a Colorado-state or flight result. {note}"
        )
    return {
        "track": "colorado-retrieval",
        "protocol": protocol,
        "not": [
            "university1652",
            "ortholoc",
            "colorado-flight-ate",
            "hunter",
            "vla",
        ],
        "n_chips": n,
        "n_query": n_query,
        "n_gallery": n_gallery,
        "n_tokens": int(codes.size),
        "unique_codes_in_eval": unique_codes,
        "codebook_size": int(codebook_size),
        "descriptor": BAG_OF_CODES_DESCRIPTOR,
        "descriptors": descriptors,
        "features_used": features_used,
        "network": False,
        "xyz_error_kind": "euclidean-3d" if use_3d else "horizontal-xy",
        "split": {
            "kind": split["kind"],
            "axis": split["axis"],
            "query_fraction": split["query_fraction"],
            "threshold": split["threshold"],
            "disjoint_box": split["disjoint_box"],
            "tiny": tiny_reason is not None,
            "tiny_reason": tiny_reason,
        },
        **{key: bag[key] for key in bag if key.startswith("recall_at_")},
        "median_xyz_error_m": bag["median_xyz_error_m"],
        "p90_xyz_error_m": bag["p90_xyz_error_m"],
        "median_oracle_xyz_m": percentile(np.asarray(oracle_err), 50.0),
        "p90_oracle_xyz_m": percentile(np.asarray(oracle_err), 90.0),
        "k_cap": k_cap,
        "queries": bag["queries"],
        "modes": modes,
        "note": note,
    }


def evaluate_retrieve_dirs(
    extract_dir: str | Path,
    fsq_dir: str | Path,
    *,
    query_fraction: float = 0.25,
    axis: str | int | None = "auto",
    out: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_retrieve_inputs(extract_dir, fsq_dir)
    report = evaluate_chip_retrieve(
        payload["codes"],
        payload["xyz"],
        codebook_size=payload["codebook_size"],
        ids=payload["ids"],
        query_fraction=query_fraction,
        axis=axis,
        features=payload["features"],
    )
    report["extract_dir"] = payload["extract_dir"]
    report["fsq_dir"] = payload["fsq_dir"]
    report["xyz_source"] = payload["xyz_source"]
    report["backbone_mode"] = payload["extract_meta"].get("backbone_mode")
    report["fsq_levels"] = payload["meta"].get("fsq_levels")
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["out"] = str(out_path)
    return report


def write_retrieve_fixture(
    extract_dir: str | Path,
    fsq_dir: str | Path,
    *,
    n_east: int = 4,
    n_north: int = 2,
    grid: int = 4,
    codebook_size: int = 32,
    spacing_m: float = 10.0,
    match_nearest: bool = True,
    match_features_nearest: bool = True,
) -> dict[str, Any]:
    """Tiny local codes/xyz/features for CPU tests.

    Codes copy the west neighbor when ``match_nearest`` is true, else a far
    chip. Features encode chip east/north so pooled cosine prefers the spatial
    neighbor; ``match_features_nearest=False`` copies a far chip's grid.
    """
    extract_dir = Path(extract_dir)
    fsq_dir = Path(fsq_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    fsq_dir.mkdir(parents=True, exist_ok=True)
    n_east = int(n_east)
    n_north = int(n_north)
    n = n_east * n_north
    xyz = np.zeros((n, 3), dtype=np.float64)
    ids = []
    east_of = []
    north_of = []
    for r in range(n_north):
        for c in range(n_east):
            i = r * n_east + c
            xyz[i] = (c * spacing_m, r * spacing_m, 80.0)
            ids.append(f"chip-{r:02d}-{c:02d}")
            east_of.append(c)
            north_of.append(r)
    codes = np.zeros((n, grid, grid), dtype=np.int64)
    for i, (e, nrt) in enumerate(zip(east_of, north_of)):
        codes[i] = (e + nrt * n_east) % codebook_size
    if match_nearest:
        for i, e in enumerate(east_of):
            if e == n_east - 1:
                src = None
                for j, (e2, n2) in enumerate(zip(east_of, north_of)):
                    if e2 == e - 1 and n2 == north_of[i]:
                        src = j
                        break
                if src is not None:
                    codes[i] = codes[src]
    else:
        for i, e in enumerate(east_of):
            if e == n_east - 1:
                far = None
                for j, (e2, n2) in enumerate(zip(east_of, north_of)):
                    if e2 == 0 and n2 != north_of[i]:
                        far = j
                        break
                if far is None:
                    far = 0
                codes[i] = codes[far]
    features = np.zeros((n, 4, 2, 2), dtype=np.float16)
    for i, (e, nrt) in enumerate(zip(east_of, north_of)):
        features[i, 0, :, :] = float(e + 1)
        features[i, 1, :, :] = float(nrt + 1)
        features[i, 2, :, :] = 1.0
        features[i, 3, 0, 0] = float(e)
        features[i, 3, 0, 1] = float(nrt)
        features[i, 3, 1, 0] = float(e + nrt)
        features[i, 3, 1, 1] = float(e - nrt)
    if not match_features_nearest:
        for i, e in enumerate(east_of):
            if e == n_east - 1:
                far = None
                for j, (e2, n2) in enumerate(zip(east_of, north_of)):
                    if e2 == 0 and n2 != north_of[i]:
                        far = j
                        break
                if far is None:
                    far = 0
                features[i] = features[far]
    np.save(extract_dir / "features.npy", features)
    np.save(extract_dir / "xyz.npy", xyz)
    np.save(fsq_dir / "codes.npy", codes)
    np.save(fsq_dir / "xyz.npy", xyz)
    (extract_dir / "ids.json").write_text(json.dumps(ids, indent=2) + "\n")
    (extract_dir / "meta.json").write_text(
        json.dumps({"n_chips": n, "backbone_mode": "stub", "rasters_copied": False}, indent=2)
        + "\n"
    )
    (fsq_dir / "meta.json").write_text(
        json.dumps(
            {
                "codebook_size": int(codebook_size),
                "fsq_levels": [8, 5, 5, 5],
                "n_chips": n,
                "unique_codes": int(np.unique(codes).size),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"n_chips": n, "ids": ids, "codebook_size": codebook_size}
