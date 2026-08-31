"""CPU same-place verification from local frozen-DINO grids and FSQ codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from monarc.localization.eval_match_pnp import match_dino_grids
from monarc.localization.eval_retrieve import load_retrieve_inputs, spatial_holdout_indices
from monarc.localization.global_retrieve import (
    BAG_OF_CODES_DESCRIPTOR,
    DINO_GRID_DESCRIPTOR,
    DINO_POOLED_DESCRIPTOR,
    bag_of_codes,
    pool_chip_features,
)

INLIER_MIN_COSINE = 0.8


def _grid(arr: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(arr)
    if value.ndim != 3:
        raise ValueError(f"{name} must be [N, H, W], got {value.shape}")
    return value


def _feature_grids(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features)
    if value.ndim != 4:
        raise ValueError(f"features must be [N, C, H, W], got {value.shape}")
    return value


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(av @ bv / denom) if denom > 0.0 else 0.0


def _sliding_grid_cosine(query: np.ndarray, gallery: np.ndarray) -> float:
    """Maximum flattened cosine over gallery windows matching query H/W."""
    q = np.asarray(query)
    g = np.asarray(gallery)
    if q.ndim != 3 or g.ndim != 3 or q.shape[0] != g.shape[0]:
        raise ValueError("DINO grids must be [C, H, W] with matching channels")
    qh, qw = q.shape[1:]
    gh, gw = g.shape[1:]
    if qh > gh or qw > gw:
        return _sliding_grid_cosine(g, q)
    return max(
        _cosine(q, g[:, row : row + qh, col : col + qw])
        for row in range(gh - qh + 1)
        for col in range(gw - qw + 1)
    )


def _auc(positive: list[float], negative: list[float]) -> float | None:
    """Pairwise AUROC with half credit for tied scores."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    delta = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(delta > 0.0) + 0.5 * np.count_nonzero(delta == 0.0)) / delta.size)


def _crop_bounds(height: int, width: int, margin: int, ordinal: int) -> tuple[slice, slice]:
    if margin < 0 or 2 * margin >= min(height, width):
        raise ValueError("crop_margin must leave at least one patch in each grid dimension")
    # Alternate an optional one-patch jitter while keeping the crop dimensions fixed.
    shift = 1 if margin > 0 and ordinal % 2 else 0
    return slice(margin - shift, height - margin - shift), slice(margin - shift, width - margin - shift)


def _mode_score(mode: str, query_codes: np.ndarray, query_features: np.ndarray,
                gallery_codes: np.ndarray, gallery_features: np.ndarray,
                codebook_size: int) -> float:
    if mode == BAG_OF_CODES_DESCRIPTOR:
        return _cosine(bag_of_codes(query_codes, codebook_size), bag_of_codes(gallery_codes, codebook_size))
    if mode == DINO_POOLED_DESCRIPTOR:
        return _cosine(pool_chip_features(query_features), pool_chip_features(gallery_features))
    if mode == DINO_GRID_DESCRIPTOR:
        return _sliding_grid_cosine(query_features, gallery_features)
    raise ValueError(f"unknown mode {mode}")


def evaluate_place_score(
    codes: np.ndarray,
    features: np.ndarray,
    xyz: np.ndarray,
    *,
    codebook_size: int,
    ids: list[str] | None = None,
    gsd_m: float = 0.3,
    size_px: int = 224,
    query_fraction: float = 0.25,
    axis: str | int | None = "auto",
    crop_margin: int = 2,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate crop-jitter/overlap positives against far spatial negatives."""
    code_grids = _grid(codes, "codes")
    feature_grids = _feature_grids(features)
    positions = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = positions.shape[0]
    if code_grids.shape[0] != n or feature_grids.shape[0] != n:
        raise ValueError("codes, features, and xyz chip counts differ")
    if code_grids.shape[1:] != feature_grids.shape[2:]:
        raise ValueError("codes and features must have matching H/W grids")
    if not np.isfinite(gsd_m) or gsd_m <= 0 or int(size_px) <= 0:
        raise ValueError("gsd_m and size_px must be positive")
    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    ids = ids if ids is not None and len(ids) == n else [f"chip-{i:04d}" for i in range(n)]
    split = spatial_holdout_indices(positions, query_fraction=query_fraction, axis=axis)
    gallery = split["gallery_idx"]
    far = split["query_idx"]
    chip_size_m = float(size_px) * float(gsd_m)
    overlap_radius_m = chip_size_m
    h, w = code_grids.shape[1:]

    queries: list[dict[str, Any]] = []
    for ordinal, gi_raw in enumerate(gallery.tolist()):
        gi = int(gi_raw)
        rows, cols = _crop_bounds(h, w, int(crop_margin), ordinal)
        queries.append({"kind": "crop-jitter", "source": gi, "true": gi,
                        "codes": code_grids[gi, rows, cols],
                        "features": feature_grids[gi, :, rows, cols]})

    # Each gallery chip gets at most one distinct, nearest overlap target. Self is
    # deliberately excluded: crop-of-self already measures that controlled case.
    overlap_pairs: list[tuple[int, int]] = []
    for qi_raw in gallery.tolist():
        qi = int(qi_raw)
        candidates = gallery[gallery != qi]
        if candidates.size:
            distances = np.linalg.norm(positions[candidates, :2] - positions[qi, :2], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= overlap_radius_m:
                overlap_pairs.append((qi, int(candidates[nearest])))
    for qi, true_i in overlap_pairs:
        queries.append({"kind": "spatial-overlap", "source": qi, "true": true_i,
                        "codes": code_grids[qi], "features": feature_grids[qi]})

    modes: dict[str, Any] = {}
    mode_names = (BAG_OF_CODES_DESCRIPTOR, DINO_POOLED_DESCRIPTOR, DINO_GRID_DESCRIPTOR)
    k = min(int(top_k), int(gallery.size))
    for mode in mode_names:
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        hits = 0
        conditional_xy: list[float] = []
        rows_out: list[dict[str, Any]] = []
        for query in queries:
            scores = np.asarray([
                _mode_score(mode, query["codes"], query["features"], code_grids[int(gi)],
                            feature_grids[int(gi)], codebook_size)
                for gi in gallery.tolist()
            ])
            # An overlap source is in the gallery but is not its overlap ground truth.
            if query["kind"] == "spatial-overlap":
                scores[gallery == int(query["source"])] = -np.inf
            order = np.argsort(-scores, kind="stable")
            ranked = gallery[order[:k]]
            true_i = int(query["true"])
            true_score = float(scores[np.flatnonzero(gallery == true_i)[0]])
            positive_scores.append(true_score)
            rank1_hit = bool(ranked.size and int(ranked[0]) == true_i)
            topk_hit = bool(np.any(ranked == true_i))
            hits += int(rank1_hit)
            if topk_hit:
                conditional_xy.append(float(np.linalg.norm(positions[query["source"], :2] - positions[true_i, :2])))
            rows_out.append({"kind": query["kind"], "query_id": ids[int(query["source"])],
                             "true_id": ids[true_i], "true_score": true_score,
                             "rank1_hit": rank1_hit, "true_in_top_k": topk_hit})
        for fi_raw in far.tolist():
            fi = int(fi_raw)
            negative_scores.extend(
                _mode_score(mode, code_grids[fi], feature_grids[fi], code_grids[int(gi)],
                            feature_grids[int(gi)], codebook_size)
                for gi in gallery.tolist()
            )
        modes[mode] = {
            "descriptor": mode,
            "auroc": _auc(positive_scores, negative_scores),
            "recall_at_1_same_place": hits / len(queries) if queries else None,
            "n_same_place_scores": len(positive_scores),
            "n_far_pair_scores": len(negative_scores),
            "xy_when_true_in_top_k": {
                "n": len(conditional_xy),
                "median_m": float(np.median(conditional_xy)) if conditional_xy else None,
                "values_m": conditional_xy,
            },
            "queries": rows_out,
        }

    same_inliers = [match_dino_grids(q["features"], feature_grids[int(q["true"])],
                                     min_cosine=INLIER_MIN_COSINE)[0].size for q in queries]
    far_inliers = [match_dino_grids(feature_grids[int(fi)], feature_grids[int(gi)],
                                    min_cosine=INLIER_MIN_COSINE)[0].size
                   for fi in far.tolist() for gi in gallery.tolist()]
    same_possible = sum(int(np.prod(q["features"].shape[1:])) for q in queries)
    far_possible = len(far_inliers) * h * w
    headline = modes[BAG_OF_CODES_DESCRIPTOR]
    note = (
        "Same-place crop-jitter and genuine chip-overlap verification is scored separately from "
        "spatial-holdout chip retrieval. The latter asks for a different geographic grain and is "
        "not the place-verification product bar. Overlap positives are reported only when present."
    )
    return {
        "track": "colorado-place-verification",
        "protocol": "same-place overlap / crop-jitter",
        "not": ["university1652", "ortholoc", "colorado-flight-ate", "hunter", "vla"],
        "network": False,
        "descriptor": BAG_OF_CODES_DESCRIPTOR,
        "auroc": headline["auroc"],
        "recall_at_1_same_place": headline["recall_at_1_same_place"],
        "n_crop_queries": int(gallery.size),
        "n_overlap_queries": len(overlap_pairs),
        "n_far_queries": int(far.size),
        "n_gallery": int(gallery.size),
        "chip_size_m": chip_size_m,
        "overlap_radius_m": overlap_radius_m,
        "crop_margin_patches": int(crop_margin),
        "top_k": k,
        "n_inliers_same_place": int(sum(same_inliers)),
        "n_inliers_far": int(sum(far_inliers)),
        "inlier_rates": {
            "same_place": float(sum(same_inliers) / same_possible) if same_possible else None,
            "far": float(sum(far_inliers) / far_possible) if far_possible else None,
        },
        "inlier_definition": {"matcher": "mutual-nearest-dino-grid", "min_cosine": INLIER_MIN_COSINE},
        "split": {"kind": split["kind"], "far_axis": split["axis"],
                  "far_query_fraction": split["query_fraction"], "threshold": split["threshold"],
                  "disjoint_box": split["disjoint_box"]},
        "modes": modes,
        "note": note,
    }


def evaluate_place_score_dirs(extract_dir: str | Path, fsq_dir: str | Path, *,
                              gsd_m: float = 0.3, query_fraction: float = 0.25,
                              axis: str | int | None = "auto", crop_margin: int = 2,
                              top_k: int = 5, out: str | Path | None = None) -> dict[str, Any]:
    payload = load_retrieve_inputs(extract_dir, fsq_dir)
    # load_retrieve_inputs intentionally flattens code grids for legacy chip
    # retrieval; place verification needs their stored spatial layout.
    spatial_codes = np.load(Path(fsq_dir) / "codes.npy")
    size_px = int(payload["extract_meta"].get("size", payload["extract_meta"].get("chip_size", 224)))
    report = evaluate_place_score(spatial_codes, payload["features"], payload["xyz"],
                                  codebook_size=payload["codebook_size"], ids=payload["ids"],
                                  gsd_m=gsd_m, size_px=size_px, query_fraction=query_fraction,
                                  axis=axis, crop_margin=crop_margin, top_k=top_k)
    report.update({"extract_dir": payload["extract_dir"], "fsq_dir": payload["fsq_dir"],
                   "xyz_source": payload["xyz_source"]})
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["out"] = str(path)
    return report


def write_place_score_fixture(extract_dir: str | Path, fsq_dir: str | Path, *,
                              spacing_m: float = 100.0, scramble_far: bool = False) -> None:
    """Write deterministic local arrays used only by CPU protocol tests."""
    extract_path, fsq_path = Path(extract_dir), Path(fsq_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    fsq_path.mkdir(parents=True, exist_ok=True)
    n, height, width, channels, codebook_size = 8, 6, 6, 12, 64
    rng = np.random.default_rng(7)
    chip_vectors = rng.normal(size=(n, channels, 1, 1))
    features = (chip_vectors + 0.05 * rng.normal(size=(n, channels, height, width))).astype(np.float32)
    codes = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None, None],
                            (n, height, width)).copy()
    xyz = np.asarray([(i % 4 * spacing_m, i // 4 * spacing_m, 80.0) for i in range(n)])
    if scramble_far:
        features[[3, 7]] = features[[0, 4]]
        codes[[3, 7]] = codes[[0, 4]]
    np.save(extract_path / "features.npy", features)
    np.save(extract_path / "xyz.npy", xyz)
    np.save(fsq_path / "codes.npy", codes)
    np.save(fsq_path / "xyz.npy", xyz)
    (extract_path / "ids.json").write_text(json.dumps([f"fixture-{i}" for i in range(n)]) + "\n")
    (extract_path / "meta.json").write_text(json.dumps({"size": 224, "backbone_mode": "fixture"}) + "\n")
    (fsq_path / "meta.json").write_text(json.dumps({"codebook_size": codebook_size}) + "\n")
