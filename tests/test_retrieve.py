"""Bag-of-codes retrieval on fixture chips."""

import numpy as np

from monarc.localization.global_retrieve import CodeRetriever, bag_of_codes, spatial_ngrams


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
