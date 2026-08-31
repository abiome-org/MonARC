"""Lost-in-space retrieval by bag-of-codes and adjacent n-grams."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


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
        order = np.argsort(-scores)[:k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]
