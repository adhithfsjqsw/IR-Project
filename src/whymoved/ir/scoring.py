"""TF-IDF cosine and BM25 term-at-a-time scoring with heap top-k and per-term contributions."""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Iterable

from .index import PositionalIndex

Scores = tuple[dict[int, float], dict[int, dict[str, float]]]


def allowed_docs(idx: PositionalIndex, days: Iterable[str] | None = None,
                 ticker: str | None = None) -> set[int] | None:
    """Doc ids passing the effective-day window and optional ticker filter (None = unfiltered)."""
    allowed: set[int] | None = None
    if days is not None:
        allowed = set()
        for d in days:
            allowed.update(idx.day_docs.get(d, ()))
    if ticker is not None:
        pool = allowed if allowed is not None else range(idx.N)
        allowed = {d for d in pool if idx.meta[d].get("ticker") == ticker}
    return allowed


def bm25_idf(idx: PositionalIndex, term: str) -> float:
    """idf = ln(1 + (N - df + 0.5)/(df + 0.5))."""
    df = idx.df(term)
    return math.log(1 + (idx.N - df + 0.5) / (df + 0.5))


def bm25_scores(idx: PositionalIndex, qweights: dict[str, float], k1: float = 1.2, b: float = 0.75,
                allowed: set[int] | None = None) -> Scores:
    """Term-at-a-time BM25 with accumulators. qweights maps stemmed term -> query weight."""
    acc: dict[int, float] = defaultdict(float)
    contrib: dict[int, dict[str, float]] = defaultdict(dict)
    avgdl = idx.avgdl
    for term, qw in qweights.items():
        plist = idx.postings.get(term)
        if not plist:
            continue
        idf = bm25_idf(idx, term)
        for did, pos in plist.items():
            if allowed is not None and did not in allowed:
                continue
            tf = len(pos)
            s = qw * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * idx.doc_len[did] / avgdl))
            acc[did] += s
            contrib[did][term] = s
    return acc, contrib


def tfidf_scores(idx: PositionalIndex, qweights: dict[str, float],
                 allowed: set[int] | None = None) -> Scores:
    """Cosine similarity. Doc weight (1+log10 tf)*idf; query weight = qw*idf (qw = raw term weight)."""
    if not idx._finalized:
        idx.finalize()
    qvec = {t: w * idx.idf_tfidf(t) for t, w in qweights.items() if idx.idf_tfidf(t) > 0 and w > 0}
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
    acc: dict[int, float] = defaultdict(float)
    contrib: dict[int, dict[str, float]] = defaultdict(dict)
    for t, qv in qvec.items():
        idf = idx.idf_tfidf(t)
        for did, pos in idx.postings[t].items():
            if allowed is not None and did not in allowed:
                continue
            s = qv * (1 + math.log10(len(pos))) * idf
            acc[did] += s
            contrib[did][t] = s
    for did in list(acc):
        denom = (idx.norms[did] or 1.0) * qnorm
        acc[did] /= denom
        for t in contrib[did]:
            contrib[did][t] /= denom
    return acc, contrib


def top_k(acc: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Top-k by score via heap; ties broken by lower doc id."""
    return heapq.nlargest(k, acc.items(), key=lambda x: (x[1], -x[0]))
