"""Shared graph algorithms (power iteration): PageRank and HITS. Used by the sector and shock modules."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def pagerank(A: sp.spmatrix, d: float = 0.85, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Weighted PageRank: r = (1-d)/N + d * P^T r, P row-normalised weights; dangling mass spread uniformly."""
    A = sp.csr_matrix(A, dtype=float)
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(1.0 / np.where(out == 0, 1, out)) @ A
    dangling = out == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = (1 - d) / n + d * (P.T @ r + r[dangling].sum() / n)
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return r / r.sum()


def hits(A: sp.spmatrix, tol: float = 1e-10, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """HITS by power iteration: a = A^T h, h = A a (L2-normalised each step). Returns (hubs, authorities) summing to 1."""
    A = sp.csr_matrix(A, dtype=float)
    n = A.shape[0]
    h = np.full(n, 1.0 / np.sqrt(n))
    a = h.copy()
    for _ in range(max_iter):
        a_new = A.T @ h
        a_new /= np.linalg.norm(a_new) or 1.0
        h_new = A @ a_new
        h_new /= np.linalg.norm(h_new) or 1.0
        done = np.abs(h_new - h).sum() + np.abs(a_new - a).sum() < tol
        h, a = h_new, a_new
        if done:
            break
    return h / h.sum(), a / a.sum()
