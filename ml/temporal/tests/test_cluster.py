from __future__ import annotations

import numpy as np

from ml.temporal.cluster import kmeans, window_embeddings

from .conftest import make_samples


def test_kmeans_separates_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (10, 4)) + [1, 0, 0, 0]
    b = rng.normal(0, 0.05, (10, 4)) + [0, 1, 0, 0]
    lab, C = kmeans(np.vstack([a, b]), 2, seed=0)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[10] and C.shape == (2, 4)


def test_window_embeddings_only_positive_originals():
    s = make_samples(n_pos=3, n_neg=2, T=60)
    X, meta = window_embeddings(s)
    assert X.shape[0] == 3 and len(meta) == 3 and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
