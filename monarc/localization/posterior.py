"""SE(3) posterior container used after matcher+PnP (no pixel inputs)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monarc.common.se3 import log_se3


@dataclass
class PosePosterior:
    """Mixture of SE(3) modes. Hunter is out of scope for this increment."""

    T_cw: np.ndarray
    weights: np.ndarray
    covariances: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.T_cw = np.asarray(self.T_cw, dtype=np.float64)
        if self.T_cw.ndim == 2:
            self.T_cw = self.T_cw[None, ...]
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if self.T_cw.shape[0] != self.weights.shape[0]:
            raise ValueError("mode count mismatch")
        wsum = self.weights.sum()
        if wsum <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = self.weights / wsum

    @classmethod
    def from_single(cls, T_cw: np.ndarray) -> "PosePosterior":
        return cls(T_cw=np.asarray(T_cw, dtype=np.float64), weights=np.array([1.0]))

    def entropy(self) -> float:
        w = np.clip(self.weights, 1e-12, 1.0)
        return float(-np.sum(w * np.log(w)))

    def mean_tangent(self) -> np.ndarray:
        return log_se3(self.T_cw[int(np.argmax(self.weights))])
