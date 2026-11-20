"""High-level searcher: BM25/TF-IDF + optional PRF + time prior over a date window."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .feedback import rocchio
from .index import PositionalIndex
from .scoring import allowed_docs, bm25_scores, tfidf_scores, top_k


@dataclass
class Hit:
    doc_id: int
    score: float
    contributions: dict[str, float] = field(default_factory=dict)


class Searcher:
    """Ranks documents for a query dict {term: weight} inside an effective-day window."""

    def __init__(self, idx: PositionalIndex, k1: float = 1.2, b: float = 0.75) -> None:
        self.idx, self.k1, self.b = idx, k1, b

    def window_days(self, day: str, calendar: list[str], back: int = 1) -> list[str]:
        """Trading days {t-back..t} from the sorted calendar list."""
        i = calendar.index(day)
        return calendar[max(0, i - back): i + 1]

    def search(self, query: dict[str, float], days: list[str] | None, ticker: str | None = None,
               k: int = 10, method: str = "bm25", prf: dict | None = None, time_lambda: float = 0.0,
               event_day: str | None = None, prior: Callable[[int], float] | None = None) -> list[Hit]:
        """Run retrieval.

        prf: {"alpha","beta","m","t"} enables pseudo-relevance feedback (top-m of first pass).
        time_lambda: score *= exp(-lambda * days_before_move) using each doc's effective_day.
        prior: optional per-doc multiplicative prior (e.g. source reputation).
        """
        idx = self.idx
        allowed = allowed_docs(idx, days, ticker)
        fn = bm25_scores if method == "bm25" else tfidf_scores
        kw = {"k1": self.k1, "b": self.b} if method == "bm25" else {}
        acc, contrib = fn(idx, query, allowed=allowed, **kw)
        if prf and acc:
            first = [d for d, _ in top_k(acc, int(prf.get("m", 5)))]
            q2 = rocchio(idx, query, first, alpha=prf.get("alpha", 1.0), beta=prf.get("beta", 0.75),
                         n_terms=int(prf.get("t", 10)))
            acc, contrib = fn(idx, q2, allowed=allowed, **kw)
        if time_lambda and event_day:
            ev = pd.Timestamp(event_day)
            for d in acc:
                dt = max(0, (ev - pd.Timestamp(idx.meta[d]["effective_day"])).days)
                acc[d] *= math.exp(-time_lambda * dt)
        if prior is not None:
            for d in acc:
                acc[d] *= prior(d)
        return [Hit(d, s, contrib.get(d, {})) for d, s in top_k(acc, k)]
