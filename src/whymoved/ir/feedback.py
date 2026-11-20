"""Rocchio relevance feedback (pseudo and explicit) over TF-IDF document vectors."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .index import PositionalIndex


def doc_vector(idx: PositionalIndex, did: int) -> dict[str, float]:
    """L2-normalised TF-IDF vector (1+log10 tf)*log10(N/df) from the forward index."""
    v = {t: (1 + math.log10(c)) * idx.idf_tfidf(t) for t, c in idx.doc_terms[did].items()}
    v = {t: w for t, w in v.items() if w > 0}
    n = math.sqrt(sum(w * w for w in v.values())) or 1.0
    return {t: w / n for t, w in v.items()}


def centroid(idx: PositionalIndex, dids: Iterable[int]) -> dict[str, float]:
    """Mean of normalised doc vectors."""
    dids = list(dids)
    c: dict[str, float] = defaultdict(float)
    for d in dids:
        for t, w in doc_vector(idx, d).items():
            c[t] += w / max(1, len(dids))
    return dict(c)


def rocchio(idx: PositionalIndex, query: dict[str, float], relevant: Iterable[int],
            nonrelevant: Iterable[int] = (), alpha: float = 1.0, beta: float = 0.75,
            gamma: float = 0.0, n_terms: int = 10) -> dict[str, float]:
    """q' = alpha*q + beta*centroid(R) - gamma*centroid(NR).

    Original query terms are always kept; at most `n_terms` new expansion terms
    (highest positive weight) are added.
    """
    rel_c = centroid(idx, relevant)
    non_c = centroid(idx, nonrelevant) if gamma else {}
    new: dict[str, float] = {t: alpha * w for t, w in query.items()}
    for t in query:
        new[t] += beta * rel_c.get(t, 0.0) - gamma * non_c.get(t, 0.0)
    extra = {t: beta * w - gamma * non_c.get(t, 0.0) for t, w in rel_c.items() if t not in query}
    for t, w in sorted(extra.items(), key=lambda x: -x[1])[:n_terms]:
        if w > 0:
            new[t] = w
    return {t: w for t, w in new.items() if w > 0}
