"""Per-move retrieval evaluation for a given system configuration (H1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..ir.index import PositionalIndex
from ..ir.query import build_query
from ..ir.search import Searcher
from ..ir.tolerant import normalize_name
from ..ir.tokenize import tokenize
from ..ir.burst import burst_terms
from .candidates import candidate_pool, window_days
from .metrics import rank_metrics


@dataclass
class System:
    """One ablation row."""
    name: str
    method: str = "bm25"            # bm25 | tfidf | random
    direction_w: float = 0.0        # 0 = no direction expansion
    prf: dict | None = None
    lam: float = 0.0
    prior: Callable[[int], float] | None = None
    exclude_price_reports: bool = False
    event_w: float = 0.0            # neutral event-lexicon expansion weight (0 = off)
    type_w8k: float = 1.0           # multiplier on 8-K docs (1 = off); a benchmark-aware diagnostic
    burst_w: float = 0.0            # weight of 'burst' terms (distinctive of the ticker on the move day); 0 = off
    burst_n: int = 10               # number of burst terms
    base_days: int = 20             # baseline window (trading days before the event window)
    params: dict = field(default_factory=dict)


class Evaluator:
    def __init__(self, idx: PositionalIndex, names: dict[str, str], gold: dict[tuple[str, str], set[int]],
                 cal: list[str], k1: float = 1.2, b: float = 0.75, seed: int = 42,
                 price_report_docs: set[int] | None = None) -> None:
        self.idx, self.names, self.gold, self.cal = idx, names, gold, cal
        self.pos = {d: i for i, d in enumerate(cal)}
        self.searcher = Searcher(idx, k1, b)
        self.seed = seed
        self.pm = price_report_docs or set()

    def run(self, moves: pd.DataFrame, system: System, mode: str = "hard", k: int = 200,
            require_gold: bool = True, return_top: bool = False) -> pd.DataFrame:
        """Evaluate every move. mode 'hard' = global pool over all tickers; 'easy' = ticker-filtered pool.

        Moves lacking gold are skipped when require_gold (otherwise reported with NaN metrics and top-1 score).
        """
        rows = []
        rng = np.random.default_rng(self.seed)
        exclude = self.pm if system.exclude_price_reports else None
        for m in moves.itertuples():
            day = pd.Timestamp(m.date).strftime("%Y-%m-%d")
            rel = self.gold.get((m.ticker, day), set())
            if require_gold and not rel:
                continue
            if day not in self.pos:
                continue
            days = window_days(self.cal, self.pos, day)
            tk = m.ticker if mode == "easy" else None
            rec = {"ticker": m.ticker, "date": day, "n_gold": len(rel), "top1": np.nan}
            if system.method == "random":
                pool = sorted(candidate_pool(self.idx, days, tk, exclude))
                ranked = list(rng.permutation(pool)) if pool else []
                rec["pool"] = len(pool)
            else:
                q = build_query(self.names[m.ticker], m.ticker,
                                int(m.direction) if system.direction_w > 0 else None, system.direction_w or 0.4,
                                system.event_w)
                if system.burst_w > 0:
                    i0 = self.pos[days[0]]
                    base = self.cal[max(0, i0 - system.base_days): i0]
                    for t, w in burst_terms(self.idx, m.ticker, days, base, system.burst_n, exclude=set(q)).items():
                        q[t] = system.burst_w * w
                hits = self.searcher.search(q, days, tk, k=k, method=system.method, prf=system.prf,
                                            time_lambda=system.lam, event_day=day, prior=system.prior,
                                            exclude=exclude,
                                            type_weights={"8k": system.type_w8k} if system.type_w8k != 1.0 else None)
                ranked = [h.doc_id for h in hits]
                rec["top1"] = hits[0].score if hits else 0.0
            if rel:
                rec.update(rank_metrics(ranked, rel))
            if return_top:
                rec["ranked"] = ranked[:10]
            rows.append(rec)
        return pd.DataFrame(rows)
