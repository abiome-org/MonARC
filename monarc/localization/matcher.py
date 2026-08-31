"""Code-to-xyz matching against the inverted metric index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monarc.map.metric_index import MetricIndex


@dataclass
class Correspondences:
    """2D-3D ties for geometric pose. Images are not stored."""

    uv: np.ndarray
    xyz: np.ndarray
    codes: np.ndarray
    query_index: np.ndarray

    def __post_init__(self) -> None:
        self.uv = np.asarray(self.uv, dtype=np.float64).reshape(-1, 2)
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        self.codes = np.asarray(self.codes, dtype=np.int64).reshape(-1)
        self.query_index = np.asarray(self.query_index, dtype=np.int64).reshape(-1)
        n = self.uv.shape[0]
        if self.xyz.shape[0] != n or self.codes.shape[0] != n or self.query_index.shape[0] != n:
            raise ValueError("correspondence array lengths must match")

    def __len__(self) -> int:
        return int(self.uv.shape[0])


def match_codes(
    query_uv: np.ndarray,
    query_codes: np.ndarray,
    index: MetricIndex,
    *,
    max_candidates: int = 4,
    unique_only: bool = False,
) -> Correspondences:
    """Expand each query code into 2D-3D candidates from the inverted index.

    Ambiguous codes (many xyz hits) are truncated to ``max_candidates``. If
    ``unique_only`` is set, codes with more than one occurrence are dropped.
    """
    query_uv = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)
    query_codes = np.asarray(query_codes, dtype=np.int64).reshape(-1)
    uvs: list[np.ndarray] = []
    xyzs: list[np.ndarray] = []
    codes: list[int] = []
    qidx: list[int] = []
    inverted = index.inverted()
    for i, code in enumerate(query_codes.tolist()):
        hits = inverted.get(int(code))
        if hits is None or len(hits) == 0:
            continue
        if unique_only and len(hits) != 1:
            continue
        take = hits[:max_candidates]
        for j in take.tolist():
            uvs.append(query_uv[i])
            xyzs.append(index.xyz[j])
            codes.append(int(code))
            qidx.append(i)
    if not uvs:
        return Correspondences(
            uv=np.zeros((0, 2)),
            xyz=np.zeros((0, 3)),
            codes=np.zeros((0,), dtype=np.int64),
            query_index=np.zeros((0,), dtype=np.int64),
        )
    return Correspondences(
        uv=np.stack(uvs, axis=0),
        xyz=np.stack(xyzs, axis=0),
        codes=np.asarray(codes, dtype=np.int64),
        query_index=np.asarray(qidx, dtype=np.int64),
    )
