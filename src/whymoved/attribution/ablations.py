"""Dev-set tuning and test-set ablation table (H1)."""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from ..common import get_logger
from .evaluate import Evaluator, System
from .metrics import bootstrap_ci, paired_bootstrap_p

log = get_logger("ablations")
METRICS = ["rr", "recall@1", "recall@5", "recall@10", "ndcg10"]


def _mrr(ev: Evaluator, moves: pd.DataFrame, s: System, mode: str = "hard") -> float:
    r = ev.run(moves, s, mode)
    return float(r.rr.mean()) if len(r) else float("nan")


def tune_on_dev(ev: Evaluator, dev: pd.DataFrame, base: dict) -> dict:
    """Coordinate-wise grid search on DEV (hard mode, MRR). Returns chosen params + the grids' scores."""
    log_ = {}
    # 1. direction weight (BM25 + expansion)
    grid_dw = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    sc = {w: _mrr(ev, dev, System("A3", direction_w=w)) for w in grid_dw}
    dw = max(grid_dw, key=lambda w: (round(sc[w], 9), -abs(w - 0.4)))
    log_["direction_w"] = sc
    # 2. Rocchio PRF (alpha, beta, t), m fixed
    grid = list(itertools.product([1.0], [0.25, 0.5, 0.75, 1.0], [5, 10, 20]))
    sc2 = {}
    for a, b, t in grid:
        sc2[(a, b, t)] = _mrr(ev, dev, System("A4", direction_w=dw, prf={"alpha": a, "beta": b, "m": 5, "t": t}))
    base_mrr = sc[dw]
    best = max(grid, key=lambda g: sc2[g])
    prf = {"alpha": best[0], "beta": best[1], "m": 5, "t": best[2]} if sc2[best] > base_mrr + 1e-9 else None
    log_["prf"] = {str(k): v for k, v in sc2.items()}
    # 3. time prior lambda
    grid_l = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    sc3 = {l: _mrr(ev, dev, System("A5", direction_w=dw, prf=prf, lam=l)) for l in grid_l}
    lam = max(grid_l, key=lambda l: (round(sc3[l], 9), -l))
    log_["lambda"] = sc3
    return {"direction_w": dw, "prf": prf, "lambda": lam, "grids": log_,
            "dev_mrr": {"A3": sc[dw], "A4": sc2[best], "A5": sc3[lam]}}


def systems_from_params(p: dict, prior: Callable[[int], float] | None = None) -> list[System]:
    s = [System("A0 random", method="random"),
         System("A1 TF-IDF name-only", method="tfidf"),
         System("A2 BM25 name-only"),
         System("A3 BM25 + direction", direction_w=p["direction_w"]),
         System("A4 + Rocchio PRF", direction_w=p["direction_w"], prf=p["prf"]),
         System("A5 + time prior", direction_w=p["direction_w"], prf=p["prf"], lam=p["lambda"])]
    if prior is not None:
        s.append(System("A6 + source reputation", direction_w=p["direction_w"], prf=p["prf"], lam=p["lambda"],
                        prior=prior))
    return s


def ablation_table(ev: Evaluator, test: pd.DataFrame, systems: list[System], mode: str,
                   n_boot: int = 1000) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every system on TEST moves with gold; bootstrap CIs; paired bootstrap vs A2 on RR."""
    per = {s.name: ev.run(test, s, mode) for s in systems}
    ref = next(k for k in per if k.startswith("A2"))
    rows = []
    for name, df in per.items():
        row = {"system": name, "n_moves": len(df)}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(df[m].to_numpy(), n_boot)
            row[m], row[m + "_lo"], row[m + "_hi"] = mean, lo, hi
        if name != ref and len(df) == len(per[ref]):
            d, p = paired_bootstrap_p(df.rr.to_numpy(), per[ref].rr.to_numpy(), n_boot)
            row["mrr_diff_vs_A2"], row["p_vs_A2"] = d, p
        rows.append(row)
    return pd.DataFrame(rows), per
