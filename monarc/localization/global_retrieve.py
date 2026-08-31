"""Lost-in-space retrieval by bag-of-codes and frozen DINO descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURE_POOL_MEAN = "mean"
FEATURE_POOL_FLATTEN = "flatten"
FEATURE_POOL_MODES = (FEATURE_POOL_MEAN, FEATURE_POOL_FLATTEN)
DINO_POOLED_DESCRIPTOR = "dino-pooled-cosine"
DINO_GRID_DESCRIPTOR = "dino-grid-cosine"
BAG_OF_CODES_DESCRIPTOR = "bag-of-codes"


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, 0.0)


def pool_chip_features(features: np.ndarray, pool: str = FEATURE_POOL_MEAN) -> np.ndarray:
    """Pool one chip grid ``[C, H, W]`` / ``[C, T]`` / ``[D]`` to an L2 vector."""
    if pool not in FEATURE_POOL_MODES:
        raise ValueError(f"pool must be one of {FEATURE_POOL_MODES}, got {pool!r}")
    arr = _finite(np.asarray(features, dtype=np.float64))
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr.mean(axis=1) if pool == FEATURE_POOL_MEAN else arr.reshape(-1)
    elif arr.ndim == 3:
        vec = arr.mean(axis=(1, 2)) if pool == FEATURE_POOL_MEAN else arr.reshape(-1)
    else:
        raise ValueError(f"chip features must be [D], [C, T], or [C, H, W], got {arr.shape}")
    return l2_normalize(vec, axis=0)


def pool_feature_batch(features: np.ndarray, pool: str = FEATURE_POOL_MEAN) -> np.ndarray:
    """Pool a chip batch ``[N, C, H, W]`` / ``[N, C, T]`` / ``[N, D]`` to ``[N, D]``."""
    if pool not in FEATURE_POOL_MODES:
        raise ValueError(f"pool must be one of {FEATURE_POOL_MODES}, got {pool!r}")
    arr = _finite(np.asarray(features, dtype=np.float64))
    if arr.ndim == 2:
        batch = arr
    elif arr.ndim == 3:
        batch = arr.mean(axis=2) if pool == FEATURE_POOL_MEAN else arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 4:
        batch = arr.mean(axis=(2, 3)) if pool == FEATURE_POOL_MEAN else arr.reshape(arr.shape[0], -1)
    else:
        raise ValueError(f"feature batch must be [N, D], [N, C, T], or [N, C, H, W], got {arr.shape}")
    return l2_normalize(batch, axis=1)


def bag_of_codes(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    """Normalized histogram over FSQ codes."""
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    hist = np.zeros((codebook_size,), dtype=np.float64)
    valid = (codes >= 0) & (codes < codebook_size)
    np.add.at(hist, codes[valid], 1.0)
    return l2_normalize(hist, axis=0)


def spatial_ngrams(code_grid: np.ndarray, codebook_size: int) -> np.ndarray:
    """Normalized histogram of horizontal and vertical adjacent code pairs."""
    grid = np.asarray(code_grid, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError(f"code_grid must be [H, W], got {grid.shape}")
    pairs: list[int] = []
    height, width = grid.shape
    modulus = codebook_size
    for r in range(height):
        for c in range(width - 1):
            pairs.append(int(grid[r, c]) + modulus * int(grid[r, c + 1]))
    for r in range(height - 1):
        for c in range(width):
            pairs.append(int(grid[r, c]) + modulus * int(grid[r + 1, c]))
    size = modulus * modulus
    hist = np.zeros((size,), dtype=np.float64)
    if pairs:
        np.add.at(hist, np.asarray(pairs, dtype=np.int64), 1.0)
    return l2_normalize(hist, axis=0)


@dataclass
class CodeRetriever:
    """In-memory gallery of bag-of-code (and optional n-gram) descriptors."""

    codebook_size: int
    ids: list[str] = field(default_factory=list)
    bags: np.ndarray | None = None
    ngrams: np.ndarray | None = None
    use_ngrams: bool = False

    def add(self, doc_id: str, codes: np.ndarray, code_grid: np.ndarray | None = None) -> None:
        bag = bag_of_codes(codes, self.codebook_size).reshape(1, -1)
        if self.bags is None:
            self.bags = bag
        else:
            self.bags = np.concatenate([self.bags, bag], axis=0)
        self.ids.append(str(doc_id))
        if self.use_ngrams:
            if code_grid is None:
                raise ValueError("n-gram retrieval requires a 2D code_grid")
            gram = spatial_ngrams(code_grid, self.codebook_size).reshape(1, -1)
            self.ngrams = gram if self.ngrams is None else np.concatenate([self.ngrams, gram], axis=0)

    def query(
        self,
        codes: np.ndarray,
        k: int = 5,
        code_grid: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        if self.bags is None or not self.ids:
            return []
        q = bag_of_codes(codes, self.codebook_size)
        scores = self.bags @ q
        if self.use_ngrams and self.ngrams is not None:
            if code_grid is None:
                raise ValueError("n-gram query requires a 2D code_grid")
            scores = scores + 0.5 * (self.ngrams @ spatial_ngrams(code_grid, self.codebook_size))
        k = min(int(k), len(self.ids))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]


@dataclass
class FeatureRetriever:
    """In-memory gallery of L2-normalized frozen DINO descriptors (cosine)."""

    pool: str = FEATURE_POOL_MEAN
    ids: list[str] = field(default_factory=list)
    descriptors: np.ndarray | None = None

    @classmethod
    def from_batch(
        cls,
        ids: list[str],
        features: np.ndarray,
        pool: str = FEATURE_POOL_MEAN,
    ) -> FeatureRetriever:
        if len(ids) != int(np.asarray(features).shape[0]):
            raise ValueError("ids and features chip counts differ")
        return cls(pool=pool, ids=[str(x) for x in ids], descriptors=pool_feature_batch(features, pool=pool))

    def add(self, doc_id: str, features: np.ndarray) -> None:
        vec = pool_chip_features(features, pool=self.pool).reshape(1, -1)
        if self.descriptors is None:
            self.descriptors = vec
        else:
            if vec.shape[1] != self.descriptors.shape[1]:
                raise ValueError(
                    f"descriptor dim {vec.shape[1]} != gallery dim {self.descriptors.shape[1]}"
                )
            self.descriptors = np.concatenate([self.descriptors, vec], axis=0)
        self.ids.append(str(doc_id))

    def query(self, features: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if self.descriptors is None or not self.ids:
            return []
        q = pool_chip_features(features, pool=self.pool)
        if q.shape[0] != self.descriptors.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != gallery dim {self.descriptors.shape[1]}"
            )
        scores = self.descriptors @ q
        k = min(int(k), len(self.ids))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]
