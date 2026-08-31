"""CPU dry-run: frozen-DINO tokens, tiny FSQ, index, retrieve, matcher+PnP/LM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from monarc.common.frustum import Camera, camera_matrix, look_at_cw
from monarc.localization.dpnp import invert_pose_error, solve_pnp_lm
from monarc.localization.global_retrieve import CodeRetriever
from monarc.localization.matcher import match_codes
from monarc.localization.posterior import PosePosterior
from monarc.map.metric_index import index_from_tokens
from monarc.map.quantizer import DRY_RUN_FSQ_LEVELS
from monarc.map.stage1 import default_stage1_modules, encode_fused, train_tiny_codebook


def _patch_centers_enu(
    n_row: int,
    n_col: int,
    patch_m: float,
    z_base: float,
    rng: np.random.Generator,
) -> np.ndarray:
    east = (np.arange(n_col) - (n_col - 1) / 2.0) * patch_m
    north = (np.arange(n_row) - (n_row - 1) / 2.0) * patch_m
    ee, nn = np.meshgrid(east, north)
    zz = z_base + 4.0 * np.sin(ee / 20.0) + 3.0 * np.cos(nn / 15.0)
    zz = zz + 0.25 * rng.standard_normal(zz.shape)
    return np.stack([ee, nn, zz], axis=-1).reshape(-1, 3)


def make_synthetic_scene(
    *,
    patch_size: int = 14,
    grid: int = 8,
    seed: int = 0,
) -> dict:
    """Unique-colored ortho chips plus DSM/vector rasters and ENU xyz."""
    rng = np.random.default_rng(seed)
    height = width = patch_size * grid
    rgb = np.zeros((1, 3, height, width), dtype=np.float32)
    dsm = np.zeros((1, 1, height, width), dtype=np.float32)
    vectors = np.zeros((1, 4, height, width), dtype=np.float32)
    xyz = _patch_centers_enu(grid, grid, patch_m=12.0, z_base=80.0, rng=rng)
    for r in range(grid):
        for c in range(grid):
            color = np.array(
                [
                    (r + 1) / (grid + 1),
                    (c + 1) / (grid + 1),
                    ((r * grid + c) % 5 + 1) / 6.0,
                ],
                dtype=np.float32,
            )
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            rgb[0, :, y0:y1, x0:x1] = color[:, None, None]
            z = float(xyz[r * grid + c, 2])
            dsm[0, 0, y0:y1, x0:x1] = z
            vectors[0, 0, y0:y1, x0:x1] = float((r + c) % 2)
            vectors[0, 1, y0:y1, x0:x1] = float(r == 0 or c == 0)
    chips = []
    for r in range(grid):
        for c in range(grid):
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            chips.append(
                {
                    "id": f"chip-{r:02d}-{c:02d}",
                    "rgb": rgb[:, :, y0:y1, x0:x1].copy(),
                    "dsm": dsm[:, :, y0:y1, x0:x1].copy(),
                    "vectors": vectors[:, :, y0:y1, x0:x1].copy(),
                    "xyz": xyz[r * grid + c],
                }
            )
    return {
        "rgb": rgb,
        "dsm": dsm,
        "vectors": vectors,
        "xyz": xyz,
        "chips": chips,
        "grid": grid,
        "patch_size": patch_size,
        "seed": seed,
    }


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def run_dry_run(
    out_dir: str | Path,
    *,
    seed: int = 0,
    steps: int = 8,
    device: str = "cpu",
) -> dict:
    """Execute the first working-model path on synthetic chips. CPU-only default."""
    if device != "cpu":
        raise ValueError("dry-run is locked to CPU for this increment")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_synthetic_scene(seed=seed)
    backbone, stem, mix, head = default_stage1_modules(levels=DRY_RUN_FSQ_LEVELS)
    train_stats = train_tiny_codebook(
        backbone,
        stem,
        mix,
        head,
        _as_tensor(scene["rgb"]),
        _as_tensor(scene["dsm"]),
        _as_tensor(scene["vectors"]),
        steps=steps,
        device=device,
    )
    _f_rgb, fused, codes = encode_fused(
        backbone,
        stem,
        mix,
        head,
        _as_tensor(scene["rgb"]),
        _as_tensor(scene["dsm"]),
        _as_tensor(scene["vectors"]),
        device=device,
    )
    code_grid = codes[0].numpy()
    codes_flat = code_grid.reshape(-1)
    meta = {
        "fsq_levels": list(head.fsq.levels),
        "codebook_size": int(head.fsq.codebook_size),
        "crs": "local-enu",
        "persist": ["fsq_codes", "xyz", "compact_metadata"],
        "backbone": "stub-dinov2-b-contract",
        "device": "cpu",
    }
    index = index_from_tokens(codes_flat, scene["xyz"], meta)
    index_dir = index.save(out_dir / "metric_index")
    retriever = CodeRetriever(codebook_size=head.fsq.codebook_size)
    tile = 2
    query_id = None
    query_codes = None
    query_grid = None
    for r in range(0, scene["grid"], tile):
        for c in range(0, scene["grid"], tile):
            doc_id = f"tile-{r:02d}-{c:02d}"
            doc_grid = code_grid[r : r + tile, c : c + tile]
            retriever.add(doc_id, doc_grid, code_grid=doc_grid)
            if r == 2 and c == 4:
                query_id = doc_id
                query_codes = doc_grid
                query_grid = doc_grid
    ranked = retriever.query(query_codes, k=3, code_grid=query_grid) if query_codes is not None else []
    retrieve_hit = bool(ranked) and ranked[0][0] == query_id

    width = height = 320
    K = camera_matrix(fx=220.0, fy=220.0, cx=width / 2.0, cy=height / 2.0)
    eye = np.array([0.0, -70.0, 160.0])
    target = np.array([0.0, 0.0, float(np.mean(scene["xyz"][:, 2]))])
    T_cw_gt = look_at_cw(eye, target)
    cam = Camera(K=K, T_cw=T_cw_gt, width=width, height=height)
    uv, _z, visible = cam.project(scene["xyz"])
    vis_idx = np.flatnonzero(visible)
    corr = match_codes(uv[vis_idx], codes_flat[vis_idx], index, max_candidates=2, unique_only=True)
    if len(corr) < 8:
        corr = match_codes(uv[vis_idx], codes_flat[vis_idx], index, max_candidates=2)
    pnp = solve_pnp_lm(corr, K, rng=np.random.default_rng(seed))
    t_err, r_err = (float("nan"), float("nan"))
    if pnp.success:
        t_err, r_err = invert_pose_error(pnp.T_cw, T_cw_gt)
    posterior = PosePosterior.from_single(pnp.T_cw if pnp.success else np.eye(4))
    report = {
        "backbone_mode": backbone.mode,
        "embed_dim": backbone.embed_dim,
        "patch_size": backbone.patch_size,
        "train_losses": [float(x) for x in train_stats["losses"]],
        "unique_codes": int(train_stats["unique_codes"]),
        "codebook_size": int(train_stats["codebook_size"]),
        "n_landmarks": index.n_landmarks,
        "index_dir": str(index_dir),
        "retrieve": {
            "query_id": query_id,
            "ranked": [{"id": i, "score": float(s)} for i, s in ranked],
            "rank1_identity": bool(retrieve_hit),
        },
        "pnp": {
            "success": bool(pnp.success),
            "n_correspondences": int(pnp.n_correspondences),
            "n_inliers": int(pnp.inliers.size),
            "reproj_rmse_px": float(pnp.reproj_rmse) if np.isfinite(pnp.reproj_rmse) else None,
            "translation_error_m": float(t_err) if np.isfinite(t_err) else None,
            "rotation_error_deg": float(r_err) if np.isfinite(r_err) else None,
        },
        "posterior_entropy": float(posterior.entropy()),
        "note": (
            "Numbers are from this synthetic dry-run only. They are not "
            "Colorado flight metrics and not University-1652 Recall@1."
        ),
    }
    (out_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "fusion_stem": stem.state_dict(),
            "channel_fusion": mix.state_dict(),
            "fsq_head": head.state_dict(),
        },
        out_dir / "stage1_cpu.pt",
    )
    return report
