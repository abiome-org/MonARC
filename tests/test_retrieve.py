"""Bag-of-codes and frozen DINO retrieval on fixture chips."""

import numpy as np

from monarc.localization.global_retrieve import (
    FEATURE_POOL_FLATTEN,
    FEATURE_POOL_MEAN,
    CodeRetriever,
    FeatureRetriever,
    bag_of_codes,
    pool_chip_features,
    pool_feature_batch,
    spatial_ngrams,
)


def test_rank1_identity_retrieval():
    rng = np.random.default_rng(3)
    K = 40
    retriever = CodeRetriever(codebook_size=K)
    gallery = {}
    for i in range(6):
        codes = rng.integers(0, K, size=16)
        gallery[f"g{i}"] = codes
        retriever.add(f"g{i}", codes)
    ranked = retriever.query(gallery["g2"], k=3)
    assert ranked[0][0] == "g2"
    assert ranked[0][1] >= ranked[1][1]


def test_ngram_histogram_shape():
    grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
    hist = spatial_ngrams(grid, codebook_size=5)
    assert hist.shape == (25,)
    assert abs(np.linalg.norm(hist) - 1.0) < 1e-6
    bag = bag_of_codes(grid, 5)
    assert bag.shape == (5,)


def test_pooled_dino_cosine_identity_retrieval():
    rng = np.random.default_rng(4)
    gallery = rng.normal(size=(6, 8, 2, 2))
    retriever = FeatureRetriever.from_batch(
        [f"g{i}" for i in range(6)],
        gallery,
        pool=FEATURE_POOL_MEAN,
    )
    ranked = retriever.query(gallery[2], k=3)
    assert ranked[0][0] == "g2"
    assert ranked[0][1] >= ranked[1][1]
    assert abs(ranked[0][1] - 1.0) < 1e-6


def test_mean_pool_and_flatten_differ_on_spatial_pattern():
    grid = np.zeros((3, 2, 2), dtype=np.float64)
    grid[0, 0, 0] = 4.0
    grid[1, 1, 1] = 4.0
    mean_vec = pool_chip_features(grid, pool=FEATURE_POOL_MEAN)
    flat_vec = pool_chip_features(grid, pool=FEATURE_POOL_FLATTEN)
    assert mean_vec.shape == (3,)
    assert flat_vec.shape == (12,)
    batch = pool_feature_batch(grid.reshape(1, 3, 2, 2), pool=FEATURE_POOL_MEAN)
    assert batch.shape == (1, 3)
    assert np.allclose(batch[0], mean_vec)


def test_feature_retriever_add_matches_from_batch():
    grids = np.array(
        [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
        ],
        dtype=np.float64,
    )
    batch = FeatureRetriever.from_batch(["a", "b"], grids)
    looped = FeatureRetriever()
    looped.add("a", grids[0])
    looped.add("b", grids[1])
    assert np.allclose(batch.descriptors, looped.descriptors)
    assert batch.query(grids[1], k=1)[0][0] == "b"
