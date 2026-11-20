"""Analog retrieval over past shock days (own similarity code) with a strict no-look-ahead library rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .daydocs import DayVectors
from .features import FEATURES, GROUPS, group_weights

METHODS = ["rand", "recent", "era", "vol", "num", "text1", "text2", "fused", "fused_t1", "rrf"]


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(n == 0, 1, n)
    return Xn @ Xn.T


class ShockLib:
    """All shock days with precomputed text similarity matrices; library selection happens per query."""

    def __init__(self, sd: pd.DataFrame, dv: DayVectors, horizon: int = 20, seed: int = 42) -> None:
        self.sd = sd.reset_index(drop=True)
        self.h, self.seed = horizon, seed
        self.n = len(self.sd)
        self.pos = self.sd.pos.to_numpy()
        self.ep = self.sd.episode_id.to_numpy()
        self.year = self.sd.date.dt.year.to_numpy()
        self.X = self.sd[[f + "_std" for f in FEATURES]].to_numpy(float)
        days = self.sd.date.dt.strftime("%Y-%m-%d").tolist()
        self.thin = dv.thin[[dv.pos[d] for d in days]]
        self.ok_outcome = self.sd.fwd_ret_20.notna().to_numpy()
        self.S = {}
        for name, fn in (("text1", dv.t1), ("text2", dv.t2)):
            V = fn(days)
            S = (V @ V.T).toarray()
            S[self.thin, :] = 0.0
            S[:, self.thin] = 0.0
            self.S[name] = S
        self.vol = self.sd[["rv20_std", "log_vix_std"]].to_numpy(float)

    # ---- library rule --------------------------------------------------
    def library(self, q: int, text_ok: bool = False) -> np.ndarray:
        """Indices d with d's forward window completed before q (pos_d + h <= pos_q), a different episode, and outcomes known."""
        m = (self.pos + self.h <= self.pos[q]) & (self.ep != self.ep[q]) & self.ok_outcome
        if text_ok:
            m &= ~self.thin
        return np.where(m)[0]

    # ---- similarities ----------------------------------------------------
    def num_sim(self, q: int, L: np.ndarray, w: dict[str, float] | None) -> np.ndarray:
        gw = group_weights(w or {g: 1.0 for g in GROUPS})
        A = self.X[L] * gw
        a = self.X[q] * gw
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(a)
        return (A @ a) / np.where(den == 0, 1, den)

    @staticmethod
    def _z(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / (s if s > 0 else 1.0)

    def scores(self, method: str, q: int, L: np.ndarray, w: dict[str, float] | None = None, wf: float = 0.5) -> np.ndarray:
        """Higher = more similar, aligned with L."""
        if method == "rand":
            return np.random.default_rng(self.seed + q).random(len(L))
        if method == "recent":
            return self.pos[L].astype(float)
        if method == "era":
            return -np.abs(self.year[L] - self.year[q]) + 1e-6 * self.pos[L]
        if method == "vol":
            return -np.linalg.norm(self.vol[L] - self.vol[q], axis=1)
        if method == "num":
            return self.num_sim(q, L, w)
        if method in ("text1", "text2"):
            return self.S[method][q, L]
        if method in ("fused", "fused_t1"):
            t = self.S["text2" if method == "fused" else "text1"][q, L]
            return wf * self._z(self.num_sim(q, L, w)) + (1 - wf) * self._z(t)
        if method == "rrf":
            rn = (-self.num_sim(q, L, w)).argsort().argsort() + 1
            rt = (-self.S["text2"][q, L]).argsort().argsort() + 1
            return 1 / (60 + rn) + 1 / (60 + rt)
        raise ValueError(method)

    def topk(self, method: str, q: int, k: int, text_ok: bool = False, w=None, wf: float = 0.5) -> np.ndarray:
        L = self.library(q, text_ok or method.startswith(("text", "fused", "rrf")))
        if len(L) == 0:
            return L
        s = self.scores(method, q, L, w, wf)
        order = np.lexsort((-self.pos[L], -s))  # ties -> more recent first
        return L[order[:k]]
