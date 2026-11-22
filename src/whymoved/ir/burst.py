"""Burst terms: what is unusual about a ticker's coverage on the move day compared with its own recent baseline.

The plain company-name query cannot tell documents about the same company apart. Terms that are frequent in the
ticker's documents in the event window but rare in its previous weeks (e.g. "guidance", "recall", "acquire") describe
*what happened*, so they separate the relevant story from background coverage. Nothing here looks at document type, so
it does not favour 8-K filings.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .index import PositionalIndex
from .scoring import bm25_idf


def ticker_day_docs(idx: PositionalIndex) -> dict[tuple[str, str], list[int]]:
    """(ticker, effective_day) -> doc ids. Built once and cached on the index object."""
    cached = getattr(idx, "_ticker_day_docs", None)
    if cached is None:
        cached = defaultdict(list)
        for day, ids in idx.day_docs.items():
            for d in ids:
                cached[(idx.meta[d].get("ticker"), day)].append(d)
        idx._ticker_day_docs = cached
    return cached


def _counts(idx: PositionalIndex, ticker: str, days: list[str]) -> tuple[Counter, int, int]:
    td = ticker_day_docs(idx)
    c: Counter = Counter()
    n_docs = 0
    for day in days:
        for d in td.get((ticker, day), ()):
            c.update(idx.doc_terms[d])
            n_docs += 1
    return c, sum(c.values()), n_docs


def burst_terms(idx: PositionalIndex, ticker: str, event_days: list[str], base_days: list[str], n: int = 10,
                exclude: set[str] | frozenset[str] = frozenset(), min_tf: int = 2, min_ratio: float = 1.5,
                min_df: int = 3) -> dict[str, float]:
    """Top-n burst terms as {stemmed term: weight in (0, 1]} (weight 1 = strongest). {} if there is no evidence.

    score(t) = log(ratio) * log(1 + tf_event) * idf, with ratio = smoothed rate in the event window / rate in the baseline.
    """
    cw, nw, _ = _counts(idx, ticker, event_days)
    if not cw:
        return {}
    cb, nb, _ = _counts(idx, ticker, base_days)
    scored: dict[str, float] = {}
    for t, f in cw.items():
        if f < min_tf or t in exclude or idx.df(t) < min_df or t.isdigit():
            continue
        ratio = ((f + 1) / (nw + 1)) / ((cb.get(t, 0) + 1) / (nb + 1))
        if ratio < min_ratio:
            continue
        scored[t] = math.log(ratio) * math.log1p(f) * bm25_idf(idx, t)
    top = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    if not top or top[0][1] <= 0:
        return {}
    m = top[0][1]
    return {t: s / m for t, s in top}
