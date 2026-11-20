"""Directed kNN graph builder; PageRank/HITS live in whymoved.graphalgo (re-exported here)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def knn_graph(S: np.ndarray, k: int = 10) -> sp.csr_matrix:
    """A[i, j] = cosine(i, j) if j is among i's top-k neighbours (self excluded)."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    n = S.shape[0]
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    return sp.csr_matrix((S[rows, nn.ravel()], (rows, nn.ravel())), shape=(n, n))


from ..graphalgo import hits, pagerank  # noqa: F401  (shared implementation)
