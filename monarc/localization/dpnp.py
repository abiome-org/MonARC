"""Geometric PnP initializer plus Levenberg-Marquardt SE(3) refine.

This is the v0 Where-Am-I pose path: matcher correspondences in, SE(3) out.
A Perceiver pose regressor is not the pose solver in this increment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from monarc.common.frustum import Camera
from monarc.common.se3 import exp_se3, invert_se3, log_se3, pose_from_rt
from monarc.localization.matcher import Correspondences


@dataclass
class PnPResult:
    T_cw: np.ndarray
    inliers: np.ndarray
    reproj_rmse: float
    success: bool
    n_correspondences: int


def _normalized(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    ones = np.ones((uv.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([uv, ones], axis=1)
    return (np.linalg.inv(K) @ homog.T).T


def dlt_pnp(xyz: np.ndarray, uv: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Direct linear transform pose from >=6 2D-3D ties. Returns ``T_cw`` or None."""
    xyz = np.asarray(xyz, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    if xyz.shape[0] < 6:
        return None
    xy_n = _normalized(uv, K)
    n = xyz.shape[0]
    A = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        X, Y, Z = xyz[i]
        x, y = xy_n[i, 0], xy_n[i, 1]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y]
    _, _, vh = np.linalg.svd(A)
    p = vh[-1].reshape(3, 4)
    if np.linalg.det(p[:, :3]) < 0:
        p = -p
    R_raw = p[:, :3]
    u, s, vt = np.linalg.svd(R_raw)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        vt[-1] *= -1
        R = u @ vt
    scale = float(s.mean())
    if abs(scale) < 1e-12:
        return None
    t = p[:, 3] / scale
    best = None
    best_pos = -1
    for sign in (1.0, -1.0):
        Rs, ts = R * sign, t * sign
        if np.linalg.det(Rs) < 0:
            continue
        Xc = (Rs @ xyz.T).T + ts
        n_pos = int(np.sum(Xc[:, 2] > 0))
        if n_pos > best_pos:
            best_pos = n_pos
            best = pose_from_rt(Rs, ts)
    return best


def _reproj_residuals(xi: np.ndarray, xyz: np.ndarray, uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    T = exp_se3(xi)
    proj, z, _ = Camera(K=K, T_cw=T, width=10**6, height=10**6, z_near=1e-6).project(xyz)
    behind = z <= 1e-6
    err = proj - uv
    err[behind] = 1e3
    return err.reshape(-1)


def refine_pnp_lm(
    T_cw: np.ndarray,
    xyz: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Levenberg-Marquardt refine of ``T_cw`` on se(3)."""
    x0 = log_se3(T_cw)
    result = least_squares(
        _reproj_residuals,
        x0,
        args=(xyz, uv, K),
        method="lm",
        max_nfev=80,
    )
    return exp_se3(result.x)


def reprojection_rmse(T_cw: np.ndarray, xyz: np.ndarray, uv: np.ndarray, K: np.ndarray) -> float:
    cam = Camera(K=K, T_cw=T_cw, width=10**6, height=10**6, z_near=1e-6)
    proj, z, _ = cam.project(xyz)
    valid = z > 1e-6
    if not np.any(valid):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((proj[valid] - uv[valid]) ** 2, axis=1))))


def ransac_pnp(
    corr: Correspondences,
    K: np.ndarray,
    *,
    reproj_thresh_px: float = 5.0,
    iters: int = 64,
    rng: np.random.Generator | None = None,
) -> PnPResult:
    """RANSAC DLT + LM. Requires at least six correspondences for DLT."""
    rng = rng or np.random.default_rng(0)
    n = len(corr)
    empty = PnPResult(
        T_cw=np.eye(4),
        inliers=np.zeros((0,), dtype=np.int64),
        reproj_rmse=float("inf"),
        success=False,
        n_correspondences=n,
    )
    if n < 6:
        return empty
    if not np.isfinite(corr.uv).all() or not np.isfinite(corr.xyz).all():
        return empty
    # DLT cannot recover a 6-DoF camera from repeated/co-linear/co-planar
    # chip-center ties.  Real extract+FSQ caches currently provide one xyz per
    # chip, so matcher evaluation commonly and correctly takes its xy fallback.
    if np.linalg.matrix_rank(corr.xyz - corr.xyz.mean(axis=0, keepdims=True)) < 3:
        return empty
    best_inliers = np.zeros((n,), dtype=bool)
    best_T = None
    sample_n = min(6, n)
    for _ in range(int(iters)):
        pick = rng.choice(n, size=sample_n, replace=False)
        T = dlt_pnp(corr.xyz[pick], corr.uv[pick], K)
        if T is None:
            continue
        cam = Camera(K=K, T_cw=T, width=10**6, height=10**6, z_near=1e-6)
        proj, z, _ = cam.project(corr.xyz)
        err = np.linalg.norm(proj - corr.uv, axis=1)
        inliers = (z > 1e-6) & (err < reproj_thresh_px)
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_T = T
    if best_T is None or best_inliers.sum() < 6:
        T_all = dlt_pnp(corr.xyz, corr.uv, K)
        if T_all is None:
            return empty
        best_T = T_all
        best_inliers = np.ones((n,), dtype=bool)
    xyz_i = corr.xyz[best_inliers]
    uv_i = corr.uv[best_inliers]
    T_ref = refine_pnp_lm(best_T, xyz_i, uv_i, K)
    rmse = reprojection_rmse(T_ref, xyz_i, uv_i, K)
    return PnPResult(
        T_cw=T_ref,
        inliers=np.flatnonzero(best_inliers).astype(np.int64),
        reproj_rmse=rmse,
        success=np.isfinite(rmse),
        n_correspondences=n,
    )


def solve_pnp_lm(corr: Correspondences, K: np.ndarray, rng: np.random.Generator | None = None) -> PnPResult:
    """Primary pose entry point used by the dry-run CLI."""
    return ransac_pnp(corr, K, rng=rng)


def invert_pose_error(T_hat: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Translation meters and rotation degrees between two world-to-camera poses."""
    delta = invert_se3(T_gt) @ T_hat
    t_err = float(np.linalg.norm(delta[:3, 3]))
    cos_theta = 0.5 * (np.trace(delta[:3, :3]) - 1.0)
    rot = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return t_err, rot
