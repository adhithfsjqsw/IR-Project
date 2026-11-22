"""FastAPI app. All endpoints read real artifacts produced by the pipeline (no placeholder data)."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..common import PROCESSED, REPORTS, load_config
from ..ir.feedback import rocchio
from ..ir.query import build_query
from ..ir.search import Searcher
from ..ir.tokenize import stem, tokenize
from ..ir.tolerant import EntityIndex
from .schemas import (clean_str, Entity, ExplainResponse, Explanation, FeedbackRequest, Move, PeerOut, PeersResponse)

app = FastAPI(title="WhyMoved API", description="Not investment advice.")
from .search import router as search_router  # noqa: E402
from .shock import router as shock_router  # noqa: E402
from .rag import router as rag_router  # noqa: E402
app.include_router(search_router)
app.include_router(shock_router)
from .price import router as price_router  # noqa: E402
app.include_router(rag_router)
app.include_router(price_router)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class State:
    """Lazily loaded artifacts."""

    def __init__(self) -> None:
        from ..data.docs import trading_days
        from ..ir.build import load_or_build
        self.cfg = load_config()
        self.docs = pd.read_parquet(PROCESSED / "docs.parquet")
        self.idx = load_or_build()
        self.comp = pd.read_parquet(PROCESSED / "companies.parquet")
        self.names = dict(zip(self.comp.ticker, self.comp.name))
        self.ents = EntityIndex(self.comp.to_dict(orient="records"))
        self.moves = pd.read_parquet(PROCESSED / "moves.parquet")
        self.cal = trading_days(pd.read_parquet(PROCESSED / "prices.parquet"), self.cfg["market_proxy"])
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self.searcher = Searcher(self.idx, self.cfg["bm25"]["k1"], self.cfg["bm25"]["b"])
        gold = pd.read_parquet(PROCESSED / "gold.parquet")
        self.gold = {(r.ticker, r.date): set(r.doc_ids) for r in gold.itertuples()}
        res = self._results()
        h1 = res.get("h1", {})
        self.tuned = h1.get("tuned", {"direction_w": 0.4, "prf": None, "lambda": 0.0})
        self.tau = h1.get("unexplained", {}).get("tau")
        try:
            from ..attribution.cause import load_classifier
            self.cause = load_classifier()
        except Exception:
            self.cause = None

    @staticmethod
    def _results() -> dict:
        p = REPORTS / "results.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@lru_cache(maxsize=1)
def state() -> State:
    return State()


def _surface_matches(text: str, stems: set[str]) -> list[str]:
    import re
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z0-9]+", text):
        if stem(w.lower()) in stems and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out[:12]


def _explain(ticker: str, date: str, mode: str, relevant=(), nonrelevant=()) -> ExplainResponse:
    s = state()
    ticker = ticker.upper()
    if ticker not in s.names:
        raise HTTPException(404, f"unknown ticker {ticker}")
    if date not in s.pos:
        raise HTTPException(400, f"{date} is not a trading day in the data")
    mv = s.moves[(s.moves.ticker == ticker) & (s.moves.date == pd.Timestamp(date))]
    direction = int(mv.direction.iloc[0]) if len(mv) else None
    days = s.cal[max(0, s.pos[date] - s.cfg["candidate_days_back"]): s.pos[date] + 1]
    q = build_query(s.names[ticker], ticker, direction, s.tuned["direction_w"])
    prf = s.tuned.get("prf")
    if relevant or nonrelevant:
        q = rocchio(s.idx, q, list(relevant), list(nonrelevant), alpha=1.0, beta=0.75, gamma=0.25, n_terms=10)
        prf = None
    tk = ticker if mode == "easy" else None
    hits = s.searcher.search(q, days, tk, k=15, prf=prf, time_lambda=s.tuned.get("lambda", 0.0), event_day=date)
    gold = s.gold.get((ticker, date), set())
    out = []
    labels_pred = {}
    if s.cause and hits:
        from ..attribution.cause import predict_cause
        news_ids = [h.doc_id for h in hits if s.idx.meta[h.doc_id]["doc_type"] == "news"]
        if news_ids:
            labels_pred = dict(zip(news_ids, predict_cause(s.docs.text.iloc[news_ids].tolist(), s.cause)))
    for r, h in enumerate(hits, 1):
        m = s.idx.meta[h.doc_id]
        row = s.docs.iloc[h.doc_id]
        is8k = m["doc_type"] == "8k"
        out.append(Explanation(
            doc_id=h.doc_id, rank=r, score=float(h.score), doc_type=m["doc_type"], publisher=clean_str(m.get("publisher")),
            effective_day=m["effective_day"], title=str(row.title), snippet=str(row.text)[:280],
            cause_label=clean_str(m.get("cause_label") if is8k else labels_pred.get(h.doc_id)),
            cause_source="8-K items" if is8k else "classifier", is_gold=h.doc_id in gold,
            matched_terms=_surface_matches(str(row.text)[:1500], set(h.contributions)), url=clean_str(row.url)))
    top = hits[0].score if hits else 0.0
    return ExplainResponse(ticker=ticker, date=date, mode=mode, query_terms=q, tau=s.tau,
                           unexplained=bool(s.tau is not None and top < s.tau), results=out)


@app.get("/api/entities", response_model=list[Entity])
def entities(q: str = Query(..., min_length=1)):
    s = state()
    sec = dict(zip(s.comp.ticker, s.comp.gics_sector))
    return [Entity(ticker=t, name=s.names[t], sector=sec[t], score=sc) for t, sc in s.ents.resolve(q, 8)]


@app.get("/api/moves", response_model=list[Move])
def moves(ticker: str, start: str | None = None, end: str | None = None):
    s = state()
    m = s.moves[s.moves.ticker == ticker.upper()]
    if start:
        m = m[m.date >= pd.Timestamp(start)]
    if end:
        m = m[m.date <= pd.Timestamp(end)]
    return [Move(ticker=r.ticker, date=r.date.strftime("%Y-%m-%d"), ret=r.ret, abnormal=r.abnormal, z=r.z,
                 direction=int(r.direction), has_8k=(r.ticker, r.date.strftime("%Y-%m-%d")) in s.gold)
            for r in m.itertuples()]


@app.get("/api/explain", response_model=ExplainResponse)
def explain(ticker: str, date: str, mode: str = "easy"):
    return _explain(ticker, date, mode)


@app.post("/api/feedback", response_model=ExplainResponse)
def feedback(req: FeedbackRequest):
    return _explain(req.ticker, req.date, req.mode, req.relevant, req.nonrelevant)


@app.get("/api/peers", response_model=PeersResponse)
def peers(ticker: str):
    s = state()
    sec = pd.read_parquet(PROCESSED / "sectors.parquet").set_index("ticker")
    t = ticker.upper()
    if t not in sec.index:
        raise HTTPException(404, f"unknown ticker {t}")
    r = sec.loc[t]
    osec = dict(zip(s.comp.ticker, s.comp.gics_sector))
    ps = [PeerOut(ticker=p, name=s.names[p], sector=osec[p], similarity=float(sim), same_sector=osec[p] == r.gics_sector)
          for p, sim in zip(r.peers, r.peer_sims)]
    return PeersResponse(ticker=t, name=r["name"], official_sector=r.gics_sector, sub_industry=r.gics_sub_industry,
                         kmeans_cluster=int(r.kmeans_cluster), peer_purity=float(r.peer_purity), peers=ps,
                         gics_sector_members_sample=[x for x in sec.index[sec.gics_sector == r.gics_sector][:12]])


@app.get("/api/sectors/clusters")
def clusters():
    d = pd.read_parquet(PROCESSED / "sectors.parquet")
    return d[["ticker", "name", "gics_sector", "gics_sub_industry", "kmeans_cluster", "hier_cluster", "x", "y",
              "peer_purity"]].to_dict(orient="records")


@app.get("/api/sectors/hidden_peers")
def hidden():
    return json.loads((PROCESSED / "hidden_peers.json").read_text(encoding="utf-8"))


@app.get("/api/graph/centrality")
def centrality():
    return pd.read_parquet(PROCESSED / "centrality.parquet").to_dict(orient="records")


@app.get("/api/eval/summary")
def eval_summary():
    p = REPORTS / "results.json"
    if not p.exists():
        raise HTTPException(404, "reports/results.json not found; run the pipeline")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/figures/{name}")
def figure(name: str):
    p = REPORTS / "figures" / name
    if not p.is_file() or p.parent != (REPORTS / "figures"):
        raise HTTPException(404)
    return FileResponse(p)
