"""Ranking metrics (binary relevance) and bootstrap utilities."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rank_metrics(ranked: Sequence[int], relevant: set[int]) -> dict[str, float]:
    """MRR component, Recall@{1,5,10}, nDCG@10. Relevant docs absent from `ranked` count as misses."""
    pos = [i + 1 for i, d in enumerate(ranked) if d in relevant]
    rr = 1.0 / pos[0] if pos else 0.0
    dcg = sum(1.0 / math.log2(p + 1) for p in pos if p <= 10)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    out = {"rr": rr, "ndcg10": dcg / ideal if ideal else 0.0}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = len({d for d in ranked[:k] if d in relevant}) / len(relevant)
    return out


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile (1-alpha) CI by resampling units with replacement."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_p(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Two-sided paired bootstrap p-value for mean(a - b) != 0. Returns (mean diff, p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(min(1.0, max(p, 1.0 / n)))
