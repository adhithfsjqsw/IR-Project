"""Spherical k-means (cosine) with k-means++ initialisation and restarts, from scratch."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _norm_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def _sim(X: sp.csr_matrix, C: np.ndarray) -> np.ndarray:
    return np.asarray(X @ C.T)


def kmeanspp_init(X: sp.csr_matrix, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding with squared cosine distance d = 1 - cos."""
    n = X.shape[0]
    first = int(rng.integers(n))
    C = [X[first].toarray().ravel()]
    d = 1 - _sim(X, np.array(C)).ravel()
    for _ in range(1, k):
        w = np.maximum(d, 0) ** 2
        p = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
        j = int(rng.choice(n, p=p))
        C.append(X[j].toarray().ravel())
        d = np.minimum(d, 1 - _sim(X, np.array(C[-1:])).ravel())
    return _norm_rows(np.array(C))


def spherical_kmeans(X: sp.spmatrix, k: int, n_init: int = 10, max_iter: int = 100, tol: float = 1e-6,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster L2-normalised rows by cosine. Returns (labels, centroids, objective = mean cosine to centroid).

    Best of `n_init` restarts by objective. Empty clusters are re-seeded with the point farthest from its centroid.
    """
    X = sp.csr_matrix(X, dtype=float)
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.csr_matrix(sp.diags(1.0 / np.where(n == 0, 1, n)) @ X)
    rng = np.random.default_rng(seed)
    best = (None, None, -np.inf)
    for _ in range(n_init):
        C = kmeanspp_init(X, k, rng)
        prev = -np.inf
        for _it in range(max_iter):
            S = _sim(X, C)
            lab = S.argmax(axis=1)
            obj = float(S[np.arange(X.shape[0]), lab].mean())
            newC = np.zeros_like(C)
            for j in range(k):
                m = lab == j
                if m.any():
                    newC[j] = np.asarray(X[m].sum(axis=0)).ravel()
                else:
                    far = int(np.argmin(S[np.arange(X.shape[0]), lab]))
                    newC[j] = X[far].toarray().ravel()
                    lab[far] = j
            C = _norm_rows(newC)
            if obj - prev < tol:
                break
            prev = obj
        S = _sim(X, C)
        lab = S.argmax(axis=1)
        obj = float(S[np.arange(X.shape[0]), lab].mean())
        if obj > best[2]:
            best = (lab, C, obj)
    return best
