"""Pinhole camera projection and frustum tests on 2.5D points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monarc.common.se3 import invert_se3, pose_from_rt, transform_points


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """3x3 pinhole intrinsics."""
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


@dataclass
class Camera:
    """World-to-camera pose ``T_cw`` with pinhole intrinsics ``K``."""

    K: np.ndarray
    T_cw: np.ndarray
    width: int
    height: int
    z_near: float = 1.0
    z_far: float = 5000.0

    @property
    def T_wc(self) -> np.ndarray:
        return invert_se3(self.T_cw)

    def project(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points to pixels. Returns ``uv, z_cam, valid``."""
        xyz_cam = transform_points(self.T_cw, np.asarray(xyz_world, dtype=np.float64))
        z = xyz_cam[:, 2]
        valid_z = (z > self.z_near) & (z < self.z_far)
        uv_h = xyz_cam @ self.K.T
        uv = np.zeros((xyz_cam.shape[0], 2), dtype=np.float64)
        safe = np.abs(z) > 1e-8
        uv[safe] = uv_h[safe, :2] / z[safe, None]
        in_image = (
            valid_z
            & (uv[:, 0] >= 0.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 0] < float(self.width))
            & (uv[:, 1] < float(self.height))
        )
        return uv, z, in_image


def look_at_cw(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """World-to-camera matrix for a camera at ``eye`` looking at ``target`` (ENU)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = np.asarray(up, dtype=np.float64).reshape(3)
    z_cam = target - eye
    z_norm = np.linalg.norm(z_cam)
    if z_norm < 1e-8:
        raise ValueError("eye and target are coincident")
    z_cam = z_cam / z_norm
    x_cam = np.cross(z_cam, up)
    x_norm = np.linalg.norm(x_cam)
    if x_norm < 1e-8:
        x_cam = np.cross(z_cam, np.array([0.0, 1.0, 0.0]))
        x_norm = np.linalg.norm(x_cam)
    x_cam = x_cam / x_norm
    y_cam = np.cross(z_cam, x_cam)
    R_wc = np.stack([x_cam, y_cam, z_cam], axis=1)
    if np.linalg.det(R_wc) < 0:
        y_cam = -y_cam
        R_wc = np.stack([x_cam, y_cam, z_cam], axis=1)
    t_wc = eye
    T_wc = pose_from_rt(R_wc, t_wc)
    return invert_se3(T_wc)


def points_in_frustum(camera: Camera, xyz_world: np.ndarray) -> np.ndarray:
    """Boolean mask of world points that project inside the image with valid depth."""
    _, _, valid = camera.project(xyz_world)
    return valid
