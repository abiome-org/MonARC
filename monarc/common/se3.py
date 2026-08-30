"""Lie group SE(3) and Lie algebra se(3) operations."""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def hat_so3(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for so(3). ``omega`` is (3,) or (..., 3)."""
    omega = np.asarray(omega, dtype=np.float64)
    wx, wy, wz = np.moveaxis(omega, -1, 0)
    zeros = np.zeros_like(wx)
    row0 = np.stack([zeros, -wz, wy], axis=-1)
    row1 = np.stack([wz, zeros, -wx], axis=-1)
    row2 = np.stack([-wy, wx, zeros], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def vee_so3(W: np.ndarray) -> np.ndarray:
    """Inverse of ``hat_so3``."""
    W = np.asarray(W, dtype=np.float64)
    return np.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1)


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map so(3) -> SO(3)."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < _EPS:
        return np.eye(3) + hat_so3(omega)
    k = omega / theta
    K = hat_so3(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Principal logarithm SO(3) -> so(3)."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = 0.5 * (np.trace(R) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < _EPS:
        return vee_so3(0.5 * (R - R.T))
    if np.pi - theta < 1e-4:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (R + R.T))
        axis = eigvecs[:, np.argmax(eigvals)]
        return theta * axis
    return vee_so3((theta / (2.0 * np.sin(theta))) * (R - R.T))


def hat_se3(xi: np.ndarray) -> np.ndarray:
    """4x4 se(3) hat operator. ``xi`` is [v (3), omega (3)]."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    v, omega = xi[:3], xi[3:]
    out = np.zeros((4, 4), dtype=np.float64)
    out[:3, :3] = hat_so3(omega)
    out[:3, 3] = v
    return out


def vee_se3(Xi: np.ndarray) -> np.ndarray:
    """Inverse of ``hat_se3``."""
    Xi = np.asarray(Xi, dtype=np.float64)
    return np.concatenate([Xi[:3, 3], vee_so3(Xi[:3, :3])])


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) -> SE(3) as a 4x4 matrix."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    v, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    R = exp_so3(omega)
    if theta < _EPS:
        V = np.eye(3) + 0.5 * hat_so3(omega)
    else:
        K = hat_so3(omega / theta)
        V = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K)
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithm SE(3) -> se(3)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    omega = log_so3(R)
    theta = float(np.linalg.norm(omega))
    if theta < _EPS:
        V_inv = np.eye(3) - 0.5 * hat_so3(omega)
    else:
        K = hat_so3(omega / theta)
        V_inv = (
            np.eye(3)
            - 0.5 * hat_so3(omega)
            + (1.0 - 0.5 * theta / np.tan(0.5 * theta)) * (K @ K)
        )
    v = V_inv @ t
    return np.concatenate([v, omega])


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) matrix."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply SE(3) transform to points shaped (N, 3)."""
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def pose_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 SE(3) matrix from rotation and translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T
