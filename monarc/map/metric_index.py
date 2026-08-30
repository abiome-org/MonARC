"""Inverted metric index: FSQ code -> xyz plus co-visible constellations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MetricIndex:
    """Compact on-disk map: integer codes, ENU coordinates, optional constellations.

    Rasters are not stored. ``meta`` holds CRS, FSQ levels, and AOI tags only.
    """

    codes: np.ndarray
    xyz: np.ndarray
    meta: dict
    neighbor_idx: np.ndarray | None = None
    neighbor_delta: np.ndarray | None = None
    neighbor_bearing: np.ndarray | None = None
    neighbor_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.codes = np.asarray(self.codes, dtype=np.int64).reshape(-1)
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N, 3], got {self.xyz.shape}")
        if self.codes.shape[0] != self.xyz.shape[0]:
            raise ValueError("codes and xyz length mismatch")
        self._lists: dict[int, np.ndarray] | None = None

    @property
    def n_landmarks(self) -> int:
        return int(self.codes.shape[0])

    def inverted(self) -> dict[int, np.ndarray]:
        if self._lists is None:
            lists: dict[int, list[int]] = {}
            for i, code in enumerate(self.codes.tolist()):
                lists.setdefault(int(code), []).append(i)
            self._lists = {k: np.asarray(v, dtype=np.int64) for k, v in lists.items()}
        return self._lists

    def lookup(self, code: int) -> np.ndarray:
        """Return xyz rows whose FSQ code equals ``code``."""
        idx = self.inverted().get(int(code))
        if idx is None:
            return np.zeros((0, 3), dtype=np.float64)
        return self.xyz[idx]

    def lookup_indices(self, code: int) -> np.ndarray:
        return self.inverted().get(int(code), np.zeros((0,), dtype=np.int64))

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "codes.npy", self.codes)
        np.save(directory / "xyz.npy", self.xyz)
        payload = {
            "n_landmarks": self.n_landmarks,
            "meta": self.meta,
            "has_constellations": self.neighbor_idx is not None,
        }
        (directory / "meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if self.neighbor_idx is not None:
            np.save(directory / "neighbor_idx.npy", self.neighbor_idx)
            np.save(directory / "neighbor_delta.npy", self.neighbor_delta)
            np.save(directory / "neighbor_bearing.npy", self.neighbor_bearing)
            np.save(directory / "neighbor_count.npy", self.neighbor_count)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "MetricIndex":
        directory = Path(directory)
        payload = json.loads((directory / "meta.json").read_text())
        neighbor_idx = None
        neighbor_delta = None
        neighbor_bearing = None
        neighbor_count = None
        if payload.get("has_constellations") and (directory / "neighbor_idx.npy").exists():
            neighbor_idx = np.load(directory / "neighbor_idx.npy")
            neighbor_delta = np.load(directory / "neighbor_delta.npy")
            neighbor_bearing = np.load(directory / "neighbor_bearing.npy")
            neighbor_count = np.load(directory / "neighbor_count.npy")
        return cls(
            codes=np.load(directory / "codes.npy"),
            xyz=np.load(directory / "xyz.npy"),
            meta=payload.get("meta", {}),
            neighbor_idx=neighbor_idx,
            neighbor_delta=neighbor_delta,
            neighbor_bearing=neighbor_bearing,
            neighbor_count=neighbor_count,
        )


def build_constellations(
    xyz: np.ndarray,
    radius_m: float = 80.0,
    max_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force co-visible neighbors inside ``radius_m`` (CPU, tiny indexes)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    idx = -np.ones((n, max_neighbors), dtype=np.int64)
    delta = np.zeros((n, max_neighbors, 3), dtype=np.float64)
    bearing = np.zeros((n, max_neighbors), dtype=np.float64)
    count = np.zeros((n,), dtype=np.int64)
    if n == 0:
        return idx, delta, bearing, count
    d = xyz[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    for i in range(n):
        order = np.argsort(dist[i])
        kept = [j for j in order.tolist() if dist[i, j] <= radius_m][:max_neighbors]
        count[i] = len(kept)
        for k, j in enumerate(kept):
            idx[i, k] = j
            delta[i, k] = xyz[j] - xyz[i]
            bearing[i, k] = float(np.arctan2(delta[i, k, 0], delta[i, k, 1]))
    return idx, delta, bearing, count


def index_from_tokens(
    codes: np.ndarray,
    xyz: np.ndarray,
    meta: dict,
    *,
    covis_radius_m: float = 80.0,
) -> MetricIndex:
    """Assemble an inverted index and metric constellations from token arrays."""
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    nidx, ndelta, nbear, ncount = build_constellations(xyz, radius_m=covis_radius_m)
    return MetricIndex(
        codes=codes,
        xyz=xyz,
        meta=meta,
        neighbor_idx=nidx,
        neighbor_delta=ndelta,
        neighbor_bearing=nbear,
        neighbor_count=ncount,
    )
