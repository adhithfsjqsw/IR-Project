"""Agglomerative clustering (scipy linkage: average, cosine distance) and dendrogram figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist


def hierarchical(X: sp.spmatrix, n_clusters: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Returns (linkage matrix, cluster labels 0..n_clusters-1) cut at n_clusters."""
    D = pdist(sp.csr_matrix(X).toarray(), metric="cosine")
    D = np.nan_to_num(D, nan=1.0)
    Z = linkage(D, method="average")
    return Z, fcluster(Z, t=n_clusters, criterion="maxclust") - 1


def dendrogram_figure(X: sp.spmatrix, names: list[str], idx: list[int], path: str) -> None:
    """Dendrogram on a readable subset of companies (indices `idx`)."""
    Xs = sp.csr_matrix(X)[idx].toarray()
    Z = linkage(np.nan_to_num(pdist(Xs, metric="cosine"), nan=1.0), method="average")
    fig, ax = plt.subplots(figsize=(10, max(8, len(idx) * 0.18)))
    dendrogram(Z, labels=[names[i] for i in idx], orientation="left", leaf_font_size=7, ax=ax)
    ax.set_title("Average-linkage dendrogram (cosine distance) of a company subset")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
