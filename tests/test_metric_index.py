"""Metric index persistence and inverted lookup."""

import numpy as np

from monarc.map.continuous_field import GridFeatureField
from monarc.map.metric_index import index_from_tokens


def test_index_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    codes = np.array([1, 2, 1, 7], dtype=np.int64)
    xyz = rng.normal(size=(4, 3))
    index = index_from_tokens(codes, xyz, {"crs": "local-enu"})
    path = index.save(tmp_path / "idx")
    loaded = type(index).load(path)
    assert loaded.n_landmarks == 4
    hits = loaded.lookup(1)
    assert hits.shape[0] == 2
    assert loaded.neighbor_count is not None
    assert (tmp_path / "idx" / "codes.npy").exists()
    assert not list(tmp_path.glob("**/*.tif"))


def test_feature_field_bilinear():
    grid = np.zeros((2, 2, 2), dtype=np.float32)
    grid[0, 0, 0] = 1.0
    grid[0, 0, 1] = 3.0
    field = GridFeatureField(grid, origin_xy=(0.0, 10.0), gsd=10.0)
    sample = field.query(np.array([[5.0, 10.0]]))
    assert sample.shape == (1, 2)
    assert abs(float(sample[0, 0]) - 2.0) < 1e-5
