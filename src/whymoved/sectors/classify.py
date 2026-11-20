"""From-scratch text classifiers on sparse matrices: Multinomial NB, Rocchio centroid, kNN; plus metrics/CV."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MultinomialNB:
    """Multinomial Naive Bayes with Laplace (add-alpha) smoothing, computed in log space. Input: term counts."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: sp.spmatrix, y: np.ndarray) -> "MultinomialNB":
        X = sp.csr_matrix(X)
        self.classes_ = np.unique(y)
        V = X.shape[1]
        self.log_prior_ = np.zeros(len(self.classes_))
        self.log_lik_ = np.zeros((len(self.classes_), V))
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            tc = np.asarray(Xc.sum(axis=0)).ravel()
            self.log_prior_[i] = np.log(Xc.shape[0] / X.shape[0])
            self.log_lik_[i] = np.log(tc + self.alpha) - np.log(tc.sum() + self.alpha * V)
        return self

    def predict_log_proba_unnorm(self, X: sp.spmatrix) -> np.ndarray:
        return np.asarray(sp.csr_matrix(X) @ self.log_lik_.T) + self.log_prior_

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        return self.classes_[self.predict_log_proba_unnorm(X).argmax(axis=1)]


def _l2(X: sp.spmatrix) -> sp.csr_matrix:
    X = sp.csr_matrix(X, dtype=float)
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ X


class RocchioClassifier:
    """Nearest-centroid by cosine similarity (centroids of L2-normalised TF-IDF vectors)."""

    def fit(self, X: sp.spmatrix, y: np.ndarray) -> "RocchioClassifier":
        X = _l2(X)
        self.classes_ = np.unique(y)
        C = np.vstack([np.asarray(X[y == c].mean(axis=0)).ravel() for c in self.classes_])
        n = np.linalg.norm(C, axis=1, keepdims=True)
        self.centroids_ = C / np.where(n == 0, 1, n)
        return self

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        return self.classes_[np.asarray(_l2(X) @ self.centroids_.T).argmax(axis=1)]


class KNNClassifier:
    """k-nearest-neighbours by cosine similarity; majority vote, ties broken by summed similarity."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X: sp.spmatrix, y: np.ndarray) -> "KNNClassifier":
        self.X_, self.y_ = _l2(X), np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: sp.spmatrix) -> np.ndarray:
        S = (_l2(X) @ self.X_.T).toarray()
        k = min(self.k, S.shape[1])
        nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
        cidx = {c: i for i, c in enumerate(self.classes_)}
        out = []
        for i in range(S.shape[0]):
            votes = np.zeros(len(self.classes_))
            sims = np.zeros(len(self.classes_))
            for j in nn[i]:
                ci = cidx[self.y_[j]]
                votes[ci] += 1
                sims[ci] += S[i, j]
            order = np.lexsort((-sims, -votes))  # primary: votes, secondary: similarity
            out.append(self.classes_[order[0]])
        return np.array(out)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 and confusion matrix (rows = true)."""
    M = confusion(y_true, y_pred, labels)
    tp = np.diag(M).astype(float)
    prec = tp / np.maximum(M.sum(axis=0), 1)
    rec = tp / np.maximum(M.sum(axis=1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = M.sum(axis=1) > 0
    return {"accuracy": float(tp.sum() / max(1, M.sum())), "macro_f1": float(f1[present].mean()),
            "per_class": {l: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                              "support": int(M.sum(axis=1)[i])} for i, l in enumerate(labels)},
            "confusion": M.tolist(), "labels": list(labels)}
