"""Local frozen-DINO grid matching inside retrieved Colorado chip candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from monarc.localization.dpnp import PnPResult, solve_pnp_lm
from monarc.localization.eval_retrieve import (
    _bag_rankings,
    _feature_rankings,
    chip_distance,
    load_retrieve_inputs,
    nearest_gallery_indices,
    percentile,
    spatial_holdout_indices,
)
from monarc.localization.global_retrieve import (
    BAG_OF_CODES_DESCRIPTOR,
    DINO_GRID_DESCRIPTOR,
    DINO_POOLED_DESCRIPTOR,
    FEATURE_POOL_FLATTEN,
    FEATURE_POOL_MEAN,
)
from monarc.localization.matcher import Correspondences

RETRIEVE_MODES = (BAG_OF_CODES_DESCRIPTOR, DINO_POOLED_DESCRIPTOR, DINO_GRID_DESCRIPTOR)


def _feature_grid(features: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return L2-normalized patch descriptors [T, D] and grid dimensions."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 3:
        channels, height, width = arr.shape
        tokens = arr.reshape(channels, -1).T
    elif arr.ndim == 2:
        channels, count = arr.shape
        height, width = 1, count
        tokens = arr.T
    else:
        raise ValueError(f"local matching needs [C, T] or [C, H, W], got {arr.shape}")
    tokens = np.where(np.isfinite(tokens), tokens, 0.0)
    norm = np.linalg.norm(tokens, axis=1, keepdims=True)
    return tokens / np.maximum(norm, 1e-8), int(height), int(width)


def _patch_uv(height: int, width: int, patch_size: float) -> np.ndarray:
    rows, cols = np.indices((height, width), dtype=np.float64)
    return np.stack(
        [(cols.reshape(-1) + 0.5) * patch_size, (rows.reshape(-1) + 0.5) * patch_size],
        axis=1,
    )


def match_dino_grids(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    min_cosine: float = 0.8,
    patch_size: float = 14.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual-nearest patch matches, returned as query indices and cosines."""
    query, qh, qw = _feature_grid(query_features)
    gallery, _gh, _gw = _feature_grid(gallery_features)
    similarity = query @ gallery.T
    q_to_g = np.argmax(similarity, axis=1)
    g_to_q = np.argmax(similarity, axis=0)
    q_idx = np.arange(query.shape[0], dtype=np.int64)
    mutual = g_to_q[q_to_g] == q_idx
    scores = similarity[q_idx, q_to_g]
    keep = mutual & np.isfinite(scores) & (scores >= float(min_cosine))
    kept = q_idx[keep]
    # Compute uv here so shape/patch geometry is validated alongside matching.
    _patch_uv(qh, qw, patch_size)[kept]
    return kept, scores[keep]


def _rankings(
    mode: str,
    codes: np.ndarray,
    features: np.ndarray,
    query_idx: np.ndarray,
    gallery_idx: np.ndarray,
    ids: list[str],
    codebook_size: int,
    k: int,
) -> list[list[tuple[str, float]]]:
    if mode == BAG_OF_CODES_DESCRIPTOR:
        return _bag_rankings(codes, query_idx, gallery_idx, ids, codebook_size, k)
    pool = FEATURE_POOL_MEAN if mode == DINO_POOLED_DESCRIPTOR else FEATURE_POOL_FLATTEN
    if mode not in RETRIEVE_MODES:
        raise ValueError(f"retrieve_mode must be one of {RETRIEVE_MODES}, got {mode!r}")
    return _feature_rankings(features, query_idx, gallery_idx, ids, k, pool)


def _failed_pnp(n: int) -> PnPResult:
    return PnPResult(np.eye(4), np.zeros((0,), dtype=np.int64), float("inf"), False, n)


def evaluate_match_pnp(
    codes: np.ndarray,
    features: np.ndarray,
    xyz: np.ndarray,
    *,
    codebook_size: int,
    ids: list[str] | None = None,
    query_fraction: float = 0.25,
    axis: str | int | None = "auto",
    top_k: int = 5,
    retrieve_mode: str = BAG_OF_CODES_DESCRIPTOR,
    min_cosine: float = 0.8,
    patch_size: float = 14.0,
) -> dict[str, Any]:
    """Retrieve candidates, locally match DINO grids, then try PnP/LM."""
    codes = np.asarray(codes, dtype=np.int64)
    features = np.asarray(features)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = int(xyz.shape[0])
    if codes.shape[0] != n or features.shape[0] != n:
        raise ValueError("codes, features, and xyz chip counts differ")
    if ids is None or len(ids) != n:
        ids = [f"chip-{i:04d}" for i in range(n)]
    split = spatial_holdout_indices(xyz, query_fraction=query_fraction, axis=axis)
    query_idx, gallery_idx = split["query_idx"], split["gallery_idx"]
    k = min(max(1, int(top_k)), int(gallery_idx.size))
    retrieve_k = min(max(k, 5), int(gallery_idx.size))
    rankings = _rankings(
        retrieve_mode, codes, features, query_idx, gallery_idx, ids, codebook_size, retrieve_k
    )
    id_to_idx = {doc_id: i for i, doc_id in enumerate(ids)}
    gallery_xyz = xyz[gallery_idx]
    gallery_ids = [ids[int(i)] for i in gallery_idx]
    hits = {1: 0, 5: 0}
    rank1_errors: list[float] = []
    refined_errors: list[float] = []
    queries: list[dict[str, Any]] = []

    for qi_raw, ranked in zip(query_idx.tolist(), rankings, strict=True):
        qi = int(qi_raw)
        retrieve_ids = [doc_id for doc_id, _ in ranked]
        ranked_ids = retrieve_ids[:k]
        nearest = nearest_gallery_indices(xyz[qi], gallery_xyz, use_3d=False)
        gt_ids = {gallery_ids[int(i)] for i in nearest.tolist()}
        for recall_k in hits:
            if gt_ids.intersection(retrieve_ids[: min(recall_k, len(retrieve_ids))]):
                hits[recall_k] += 1
        rank1_xyz = xyz[id_to_idx[ranked_ids[0]]]
        rank1_error = chip_distance(xyz[qi], rank1_xyz, use_3d=False)
        rank1_errors.append(rank1_error)

        candidate_rows: list[dict[str, Any]] = []
        all_uv: list[np.ndarray] = []
        all_xyz: list[np.ndarray] = []
        all_codes: list[int] = []
        all_qidx: list[int] = []
        _q_tokens, qh, qw = _feature_grid(features[qi])
        uv_grid = _patch_uv(qh, qw, patch_size)
        for candidate_id in ranked_ids:
            gi = id_to_idx[candidate_id]
            matched_q, scores = match_dino_grids(
                features[qi], features[gi], min_cosine=min_cosine, patch_size=patch_size
            )
            candidate_rows.append(
                {
                    "id": candidate_id,
                    "match_inlier_count": int(matched_q.size),
                    "mean_match_cosine": float(scores.mean()) if scores.size else None,
                }
            )
            for token_i in matched_q.tolist():
                all_uv.append(uv_grid[int(token_i)])
                all_xyz.append(xyz[gi])
                all_codes.append(gi)
                all_qidx.append(int(token_i))

        if candidate_rows:
            best = max(
                candidate_rows,
                key=lambda row: (
                    row["match_inlier_count"],
                    -1.0 if row["mean_match_cosine"] is None else row["mean_match_cosine"],
                ),
            )
            refined_xyz = xyz[id_to_idx[best["id"]]].copy()
            match_inliers = int(best["match_inlier_count"])
        else:
            best = None
            refined_xyz = rank1_xyz.copy()
            match_inliers = 0
        refined_error = chip_distance(xyz[qi], refined_xyz, use_3d=False)
        refined_errors.append(refined_error)

        if all_uv:
            corr = Correspondences(all_uv, all_xyz, all_codes, all_qidx)
            finite = np.isfinite(corr.xyz).all(axis=1)
            corr_finite = Correspondences(
                corr.uv[finite], corr.xyz[finite], corr.codes[finite], corr.query_index[finite]
            )
            width_px = max(qw * patch_size, 1.0)
            height_px = max(qh * patch_size, 1.0)
            focal = max(width_px, height_px)
            K = np.array(
                [
                    [focal, 0.0, width_px / 2],
                    [0.0, focal, height_px / 2],
                    [0.0, 0.0, 1.0],
                ]
            )
            pnp = solve_pnp_lm(corr_finite, K) if len(corr_finite) else _failed_pnp(0)
        else:
            pnp = _failed_pnp(0)
        queries.append(
            {
                "query_id": ids[qi],
                "top_k_ids": ranked_ids,
                "rank1_id": ranked_ids[0],
                "rank1_xy_m": rank1_xyz[:2].tolist(),
                "rank1_xy_error_m": rank1_error,
                "candidate_matches": candidate_rows,
                "match_inlier_count": match_inliers,
                "refined_xy_m": refined_xyz[:2].tolist(),
                "xy_error_m": refined_error,
                "pnp_success": bool(pnp.success),
                "pnp_inlier_count": int(pnp.inliers.size),
                "pose_T_cw": pnp.T_cw.tolist() if pnp.success else None,
                "xy_estimate_kind": "matched-chip-center-horizontal-fallback",
            }
        )

    n_query = int(query_idx.size)
    tiny_reasons = []
    if n < 128:
        tiny_reasons.append(f"n_chips={n} < 128")
    if n_query < 32:
        tiny_reasons.append(f"n_query={n_query} < 32")
    return {
        "track": "colorado-match-pnp",
        "protocol": "local frozen-DINO grid matching inside retrieved gallery chips",
        "n_chips": n,
        "n_query": n_query,
        "n_gallery": int(gallery_idx.size),
        "k": k,
        "top_k": k,
        "retrieve_descriptor": retrieve_mode,
        "retrieve_recall_at_1": hits[1] / n_query,
        "retrieve_recall_at_5": hits[5] / n_query,
        "retrieve": {
            "descriptor": retrieve_mode,
            "recall_at_1": hits[1] / n_query,
            "recall_at_5": hits[5] / n_query,
        },
        "queries": queries,
        "aggregate": {
            "rank1_median_xy_error_m": percentile(rank1_errors, 50),
            "rank1_p90_xy_error_m": percentile(rank1_errors, 90),
            "matcher_median_xy_error_m": percentile(refined_errors, 50),
            "matcher_p90_xy_error_m": percentile(refined_errors, 90),
        },
        "split": {
            "kind": split["kind"],
            "axis": split["axis"],
            "query_fraction": split["query_fraction"],
            "threshold": split["threshold"],
            "disjoint_box": split["disjoint_box"],
            "tiny": bool(tiny_reasons),
            "tiny_reason": "; ".join(tiny_reasons) or None,
        },
        "xyz_kind": "coarse-chip-center",
        "xyz_is_chip_center": True,
        "dsm_z_may_be_nan": bool(np.isnan(xyz[:, 2]).any()),
        "network": False,
        "is_university1652": False,
        "is_gps_denied_flight_ate": False,
        "not": ["university1652", "gps-denied-flight-ate", "per-patch-metric-xyz"],
        "note": (
            "Results are for this local cache and spatial split only. Patch ties use coarse "
            "chip-center xyz; NaN DSM z is excluded from PnP and xy remains a horizontal fallback."
        ),
    }


def evaluate_match_pnp_dirs(
    extract_dir: str | Path,
    fsq_dir: str | Path,
    *,
    query_fraction: float = 0.25,
    axis: str | int | None = "auto",
    top_k: int = 5,
    retrieve_mode: str = BAG_OF_CODES_DESCRIPTOR,
    min_cosine: float = 0.8,
    out: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_retrieve_inputs(extract_dir, fsq_dir)
    patch_size = float(payload["extract_meta"].get("patch_size", 14))
    report = evaluate_match_pnp(
        payload["codes"], payload["features"], payload["xyz"],
        codebook_size=payload["codebook_size"], ids=payload["ids"],
        query_fraction=query_fraction, axis=axis, top_k=top_k,
        retrieve_mode=retrieve_mode, min_cosine=min_cosine, patch_size=patch_size,
    )
    report.update(
        extract_dir=payload["extract_dir"], fsq_dir=payload["fsq_dir"],
        xyz_source=payload["xyz_source"], backbone_mode=payload["extract_meta"].get("backbone_mode"),
    )
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["out"] = str(out_path)
    return report
