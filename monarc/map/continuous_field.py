"""Continuous aerial feature field over a metric patch grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridFeatureField:
    """Bilinear sampler of a feature grid in local east-north meters.

    ``grid`` is ``[C, H, W]`` with row 0 at the northern edge (image convention).
    ``origin_xy`` is the east/north coordinate of the northwest corner.
    ``gsd`` is meters per pixel along east (x) and north (y, decreasing with row).
    """

    grid: np.ndarray
    origin_xy: tuple[float, float]
    gsd: float
    crs: str = "local-enu"

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.float32)
        if grid.ndim != 3:
            raise ValueError(f"grid must be [C, H, W], got {grid.shape}")
        self.grid = grid
        self.channels, self.height, self.width = grid.shape

    def query(self, xy: np.ndarray) -> np.ndarray:
        """Sample features at ENU positions ``xy`` shaped (N, 2)."""
        xy = np.asarray(xy, dtype=np.float64)
        origin_e, origin_n = self.origin_xy
        col = (xy[:, 0] - origin_e) / self.gsd
        row = (origin_n - xy[:, 1]) / self.gsd
        col = np.clip(col, 0.0, self.width - 1.0001)
        row = np.clip(row, 0.0, self.height - 1.0001)
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        c1 = np.clip(c0 + 1, 0, self.width - 1)
        r1 = np.clip(r0 + 1, 0, self.height - 1)
        dc = (col - c0).astype(np.float32)
        dr = (row - r0).astype(np.float32)
        g = self.grid
        v00 = g[:, r0, c0]
        v01 = g[:, r0, c1]
        v10 = g[:, r1, c0]
        v11 = g[:, r1, c1]
        top = v00 * (1.0 - dc) + v01 * dc
        bot = v10 * (1.0 - dc) + v11 * dc
        return (top * (1.0 - dr) + bot * dr).T
