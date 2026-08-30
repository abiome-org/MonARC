"""Matcher + PnP/LM recovers a known synthetic pose."""

import numpy as np

from monarc.common.frustum import Camera, camera_matrix, look_at_cw
from monarc.localization.dpnp import invert_pose_error, solve_pnp_lm
from monarc.localization.matcher import match_codes
from monarc.map.metric_index import index_from_tokens


def _scene(n: int = 20, seed: int = 4):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-30, 30, size=(n, 2))
    z = 50.0 + rng.uniform(-4, 8, size=(n,))
    xyz = np.column_stack([xy, z])
    codes = np.arange(n, dtype=np.int64)
    return xyz, codes


def test_pnp_recovers_look_at_pose():
    xyz, codes = _scene()
    index = index_from_tokens(codes, xyz, {"crs": "local-enu"})
    K = camera_matrix(180.0, 180.0, 160.0, 120.0)
    T_gt = look_at_cw(np.array([0.0, -55.0, 95.0]), np.array([0.0, 0.0, 52.0]))
    cam = Camera(K=K, T_cw=T_gt, width=320, height=240)
    uv, _, vis = cam.project(xyz)
    keep = np.flatnonzero(vis)
    assert keep.size >= 8
    corr = match_codes(uv[keep], codes[keep], index, unique_only=True)
    result = solve_pnp_lm(corr, K, rng=np.random.default_rng(0))
    assert result.success
    t_err, r_err = invert_pose_error(result.T_cw, T_gt)
    assert t_err < 0.5
    assert r_err < 1.0
    assert result.reproj_rmse < 1.0


def test_pnp_fails_gracefully_with_too_few_points():
    xyz, codes = _scene(n=4)
    index = index_from_tokens(codes, xyz, {})
    K = camera_matrix(100.0, 100.0, 50.0, 50.0)
    corr = match_codes(np.zeros((4, 2)), codes, index)
    result = solve_pnp_lm(corr, K)
    assert not result.success
