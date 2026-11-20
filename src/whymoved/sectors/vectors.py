"""From-scratch TF-IDF vectorizer (project tokenizer, optional bigrams, min_df/max_df, sublinear tf, L2)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from ..ir.tokenize import tokenize


def ngrams(tokens: list[str], n_max: int) -> list[str]:
    out = list(tokens)
    for n in range(2, n_max + 1):
        out += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return out


class TfidfVectorizer:
    """tf = 1 + log10(count) (sublinear), idf = log10(N/df), rows L2-normalised.

    min_df: absolute count; max_df: fraction of documents. Vocabulary is fitted on the training docs only.
    """

    def __init__(self, min_df: int = 2, max_df: float = 0.5, ngram_max: int = 2, sublinear: bool = True) -> None:
        self.min_df, self.max_df, self.ngram_max, self.sublinear = min_df, max_df, ngram_max, sublinear

    def _terms(self, text: str) -> list[str]:
        return ngrams(tokenize(text), self.ngram_max)

    def fit(self, docs: Iterable[str]) -> "TfidfVectorizer":
        counts = [Counter(self._terms(d)) for d in docs]
        N = len(counts)
        df = Counter(t for c in counts for t in c)
        keep = sorted(t for t, d in df.items() if d >= self.min_df and d <= self.max_df * N)
        self.vocab_ = {t: i for i, t in enumerate(keep)}
        self.terms_ = np.array(keep)
        self.idf_ = np.array([math.log10(N / df[t]) for t in keep])
        return self

    def counts(self, docs: Iterable[str]) -> sp.csr_matrix:
        """Raw term-count matrix (for Naive Bayes)."""
        rows, cols, vals = [], [], []
        n = 0
        for i, d in enumerate(docs):
            n += 1
            for t, c in Counter(self._terms(d)).items():
                j = self.vocab_.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab_)), dtype=float)

    def tfidf_from_counts(self, C: sp.csr_matrix) -> sp.csr_matrix:
        X = C.copy().tocsr()
        if self.sublinear:
            X.data = 1 + np.log10(X.data)
        X = X @ sp.diags(self.idf_)
        n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        n[n == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / n) @ X)

    def transform(self, docs: Iterable[str]) -> sp.csr_matrix:
        return self.tfidf_from_counts(self.counts(docs))

    def fit_transform(self, docs: list[str]) -> sp.csr_matrix:
        return self.fit(docs).transform(docs)
