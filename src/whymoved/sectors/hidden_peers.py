"""Peers by text similarity, peer-sector purity and 'hidden peer' explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .vectors import TfidfVectorizer


def similarity(X: sp.spmatrix) -> np.ndarray:
    """Cosine similarity matrix of L2-normalised rows (dense; ~500x500)."""
    return (X @ X.T).toarray()


def peers(S: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of top-k neighbours per row (self excluded), ordered by descending similarity."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, nn, axis=1), axis=1)
    return np.take_along_axis(nn, order, axis=1)


def peer_purity(nn: np.ndarray, sector: np.ndarray) -> np.ndarray:
    """Share of each company's neighbours that share its GICS sector."""
    return (sector[nn] == sector[:, None]).mean(axis=1)


def shared_terms(vec: TfidfVectorizer, X: sp.csr_matrix, i: int, j: int, top: int = 5) -> list[str]:
    """Top terms by product of TF-IDF weights between two companies (why they look alike)."""
    prod = X[i].multiply(X[j]).toarray().ravel()
    return [str(vec.terms_[t]) for t in np.argsort(-prod)[:top] if prod[t] > 0]


def hidden_peers(companies: pd.DataFrame, vec: TfidfVectorizer, X: sp.csr_matrix, nn: np.ndarray,
                 purity: np.ndarray, n: int = 20) -> pd.DataFrame:
    """The n lowest-purity companies with neighbours (cross-sector ones flagged) and shared-term 'why'."""
    rows = []
    for i in np.argsort(purity, kind="stable")[:n]:
        nb = []
        for j in nn[i][:5]:
            nb.append({"ticker": companies.ticker.iloc[j], "sector": companies.gics_sector.iloc[j],
                       "why": shared_terms(vec, X, i, j)})
        cross = [x for x in nb if x["sector"] != companies.gics_sector.iloc[i]]
        rows.append({"ticker": companies.ticker.iloc[i], "name": companies.name.iloc[i],
                     "sector": companies.gics_sector.iloc[i], "purity": float(purity[i]),
                     "neighbors": nb,
                     "why": "; ".join(f"{x['ticker']} ({x['sector']}): {', '.join(x['why'])}" for x in cross[:3])})
    return pd.DataFrame(rows)
