"""Shock-mode endpoints. Analogs are descriptive, not a forecast."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..common import PROCESSED, load_config
from ..reporting import read_results
from ..shock.analogs import ShockLib, cosine_matrix
from ..shock.daydocs import build_day_vectors, top_terms
from ..shock.features import FEATURES, GROUPS, SECTORS, group_weights

router = APIRouter()
VIEWS = {"numeric": "num", "text": "text1", "burst": "text2", "fused": "fused"}


class ShockState:
    def __init__(self) -> None:
        self.cfg = load_config()["shock"]
        self.sd = pd.read_parquet(PROCESSED / "shock_days.parquet").reset_index(drop=True)
        self.all = pd.read_parquet(PROCESSED / "shock_features_all.parquet").set_index("date")
        self.dv = build_day_vectors()
        self.lib = ShockLib(self.sd, self.dv, self.cfg["horizon"], load_config()["seed"])
        self.docs = pd.read_parquet(PROCESSED / "day_docs.parquet").set_index("day")
        self.R = read_results()
        tuned = self.R.get("shock_analogs", {}).get("tuned", {})
        self.w = tuned.get("group_weights") or {g: 1.0 for g in GROUPS}
        self.wf = tuned.get("fusion_w", 0.5)
        gw = group_weights(self.w)
        Xw = self.lib.X * gw
        self.Snum = cosine_matrix(Xw)
        days = self.sd.date.dt.strftime("%Y-%m-%d").tolist()
        self.days = days
        self.regime = {}
        reg = self.R.get("shock_regimes", {}).get("clusters", {})
        self.cluster = reg.get("labels")

    def idx(self, date: str) -> int:
        hit = np.where(self.sd.date == pd.Timestamp(date))[0]
        if not len(hit):
            raise HTTPException(404, f"{date} is not a detected shock day (see /api/shock/days)")
        return int(hit[0])

    def excerpts(self, date: str, n: int = 6) -> list[str]:
        return self.docs.text[date].split(" . ")[:n] if date in self.docs.index else []


@lru_cache(maxsize=1)
def st() -> ShockState:
    return ShockState()


@router.get("/api/shock/days")
def shock_days(start: str | None = None, end: str | None = None):
    s = st()
    d = s.sd.assign(cluster=s.cluster if s.cluster else None, thin_text=s.lib.thin)
    if start:
        d = d[d.date >= pd.Timestamp(start)]
    if end:
        d = d[d.date <= pd.Timestamp(end)]
    return [{"date": r.date.strftime("%Y-%m-%d"), "r_spy": r.r_spy, "z_spy": r.z_spy, "vix_z": None if pd.isna(r.vix_z) else r.vix_z,
             "direction": int(r.direction), "episode_id": int(r.episode_id), "thin_text": bool(r.thin_text),
             "cluster": None if pd.isna(r.cluster) else int(r.cluster)} for r in d.itertuples()]


@router.get("/api/shock/timeline")
def timeline(step: int = 5):
    """SPY close series (downsampled) for the timeline chart."""
    a = st().all
    return [{"date": i.strftime("%Y-%m-%d"), "close": float(c)} for i, c in a.close.iloc[::max(1, step)].items()]


@router.get("/api/shock/day")
def shock_day(date: str):
    s = st()
    i = s.idx(date)
    r = s.sd.iloc[i]
    raw = {f: float(r[f]) for f in FEATURES}
    std = {f: float(r[f + "_std"]) for f in FEATURES}
    sector_ret = {x: float(r[f"rel_{x}"] + r.r_spy) for x in SECTORS}
    X1, X2 = s.dv.t1([date]), s.dv.t2([date])
    return {"date": date, "flags": {"z_spy": float(r.z_spy), "vix_z": None if pd.isna(r.vix_z) else float(r.vix_z),
                                   "direction": int(r.direction)}, "episode_id": int(r.episode_id),
            "features_raw": raw, "features_std": std, "sector_returns": sector_ret, "thin_text": bool(s.lib.thin[i]),
            "n_headlines": int(s.docs.n_headlines[date]) if date in s.docs.index else 0, "headlines": s.excerpts(date),
            "top_terms_t1": top_terms(X1, s.dv, 0), "top_terms_t2": top_terms(X2, s.dv, 0),
            "cluster": int(s.cluster[i]) if s.cluster else None,
            "outcomes": {k: (None if pd.isna(r[k]) else float(r[k])) for k in ("fwd_ret_5", "fwd_ret_20", "fwd_maxdd_20", "fwd_vix_chg_20")}}


def _path(s: ShockState, date: str) -> list[dict]:
    a = s.all
    i = a.index.get_loc(pd.Timestamp(date))
    seg = a.close.iloc[max(0, i - 5): i + 21]
    base = a.close.iloc[i]
    return [{"offset": k - min(i, 5), "value": float(v / base)} for k, v in enumerate(seg.to_numpy())]


def _analog_out(s: ShockState, q: int, ids, scores) -> list[dict]:
    out = []
    for a, sc in zip(ids, scores):
        row = s.sd.iloc[a]
        d = row.date.strftime("%Y-%m-%d")
        out.append({"date": d, "score": float(sc), "r_spy": float(row.r_spy), "episode_id": int(row.episode_id),
                    "view_scores": {"numeric": float(s.Snum[q, a]), "text": float(s.lib.S["text1"][q, a]), "burst": float(s.lib.S["text2"][q, a])},
                    "outcomes": {k: float(row[k]) for k in ("fwd_ret_5", "fwd_ret_20", "fwd_maxdd_20", "fwd_vix_chg_20")},
                    "path": _path(s, d), "headlines": s.excerpts(d, 4), "thin_text": bool(s.lib.thin[a]), "year": int(row.date.year)})
    return out


def _analogs(s: ShockState, date: str, view: str, k: int, rel=(), nonrel=()):
    q = s.idx(date)
    method = VIEWS[view]
    text_ok = method != "num"
    L = s.lib.library(q, text_ok)
    if len(L) == 0:
        raise HTTPException(404, "no eligible library days before this date")
    if rel or nonrel:  # Rocchio in similarity space (equivalent to vector Rocchio for cosine scoring)
        def comb(S):
            sc = S[q, L].copy()
            if rel:
                sc += 0.75 * S[np.ix_(list(rel), L)].mean(axis=0)
            if nonrel:
                sc -= 0.25 * S[np.ix_(list(nonrel), L)].mean(axis=0)
            return sc
        if method == "num":
            scores = comb(s.Snum)
        elif method == "fused":
            scores = s.wf * s.lib._z(comb(s.Snum)) + (1 - s.wf) * s.lib._z(comb(s.lib.S["text2"]))
        else:
            scores = comb(s.lib.S[method])
    else:
        scores = s.lib.scores(method, q, L, s.w, s.wf)
    order = np.lexsort((-s.lib.pos[L], -scores))[:k]
    ids = L[order]
    an = _analog_out(s, q, ids, scores[order])
    fan = {p: [float(np.nanpercentile([x["path"][j]["value"] for x in an if j < len(x["path"])], pc)) for j in range(26)]
           for p, pc in (("q25", 25), ("median", 50), ("q75", 75))}
    return {"date": date, "view": view, "k": k, "library_size": int(len(L)),
            "caution": "Analogs are descriptive, not a forecast.", "n_independent_episodes": int(len(set(s.lib.ep[ids]))),
            "query_path": _path(s, date), "analogs": an, "fan": {"offsets": list(range(-5, 21)), **fan}}


@router.get("/api/shock/analogs")
def shock_analogs(date: str, view: str = "fused", k: int = 5):
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {list(VIEWS)}")
    return _analogs(st(), date, view, max(1, min(k, 20)))


class ShockFeedback(BaseModel):
    date: str
    view: str = "fused"
    k: int = 5
    relevant: list[str] = []
    nonrelevant: list[str] = []


@router.post("/api/shock/feedback")
def shock_feedback(req: ShockFeedback):
    s = st()
    return _analogs(s, req.date, req.view, req.k, [s.idx(d) for d in req.relevant], [s.idx(d) for d in req.nonrelevant])


@router.get("/api/shock/search")
def shock_search(q: str = Query(..., min_length=2), k: int = 10):
    """BM25 over day documents (index built lazily and cached)."""
    from ..shock.dayindex import load_day_index
    from ..ir.scoring import bm25_scores, top_k
    from ..ir.tokenize import tokenize
    idx = load_day_index()
    acc, contrib = bm25_scores(idx, {t: 1.0 for t in tokenize(q)})
    s = st()
    out = []
    for d, sc in top_k(acc, k):
        day = idx.meta[d]["effective_day"]
        out.append({"date": day, "score": float(sc), "contributions": {t: float(v) for t, v in contrib[d].items()},
                    "is_shock_day": bool((s.sd.date == pd.Timestamp(day)).any()), "headlines": s.excerpts(day, 3)})
    return out


@router.get("/api/shock/archetypes")
def archetypes():
    return st().R.get("shock_regimes", {}).get("archetypes", {})


@router.get("/api/shock/clusters")
def clusters():
    c = st().R.get("shock_regimes", {}).get("clusters", {})
    return {k: v for k, v in c.items() if k != "labels"}


@router.get("/api/eval/shock")
def eval_shock():
    R = st().R
    return {k: v for k, v in R.items() if k.startswith("shock_") or k.startswith("search_bench")}
