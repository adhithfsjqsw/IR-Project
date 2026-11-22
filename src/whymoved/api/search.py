"""GET /api/search: standalone news search (plain terms, "phrases", AND/OR/NOT, wildcard terms like nvd*)."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .schemas import clean_str

from ..ir.scoring import bm25_scores, tfidf_scores, top_k
from ..ir.tokenize import stem, tokenize

router = APIRouter()
TITLE_BOOST, COMPANY_BOOST = 1.0, 0.5
OPS = {"AND", "OR", "NOT", "(", ")"}


class SearchHit(BaseModel):
    doc_id: int
    rank: int
    score: float
    contributions: dict[str, float]
    doc_type: str
    publisher: str | None
    ticker: str
    effective_day: str
    title: str
    snippet: str
    matched_terms: list[str]
    cause_label: str | None
    url: str | None = None
    also_tagged: list[str] = []  # other tickers carrying the identical article (same title and day)


class SearchResponse(BaseModel):
    query: str
    mode: str
    n_candidates: int
    expanded_wildcards: dict[str, list[str]]
    did_you_mean: list[str]
    results: list[SearchHit]


def parse_query(q: str, vocab) -> tuple[str, dict[str, float], dict[str, list[str]], bool]:
    """-> (boolean expression with wildcards expanded, positive-term weights, expansions, uses_boolean)."""
    toks = re.findall(r'"[^"]+"|\(|\)|[^\s()"]+', q)
    out, weights, expansions, boolean, neg = [], {}, {}, False, False
    for t in toks:
        if t in OPS:
            boolean = True
            neg = t == "NOT"
            out.append(t)
            continue
        if t.startswith('"'):
            boolean = True
            out.append(t)
            if not neg:
                weights.update({s: 1.0 for s in tokenize(t.strip('"'))})
        elif "*" in t:
            rx = re.compile("^" + ".*".join(re.escape(p) for p in t.lower().split("*")) + "$")
            hits = sorted(v for v in vocab if rx.match(v))[:50]
            expansions[t] = hits
            out.append("(" + " OR ".join(hits) + ")" if hits else '"zzzznomatchzzzz"')
            boolean = boolean or bool(hits)
            if not neg:
                weights.update({h: 1.0 for h in hits})
        else:
            out.append(t)
            if not neg:
                weights.update({s: 1.0 for s in tokenize(t)})
        if t not in OPS:
            neg = False
    return " ".join(out), weights, expansions, boolean


@router.get("/api/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), mode: str = "bm25", ticker: str | None = None, start: str | None = None,
           end: str | None = None, publisher: str | None = None, doc_type: str | None = None, k: int = 10):
    from .main import _surface_matches, state
    s = state()
    if mode not in ("bm25", "tfidf"):
        raise HTTPException(400, "mode must be bm25 or tfidf")
    expr, weights, expansions, boolean = parse_query(q, s.idx.postings)
    if not weights:
        raise HTTPException(400, "query has no searchable positive terms")
    allowed = s.idx.boolean(expr) if boolean else None
    meta = s.idx.meta
    def ok(d: int) -> bool:
        m = meta[d]
        return ((ticker is None or m["ticker"] == ticker.upper()) and (doc_type is None or m["doc_type"] == doc_type)
                and (publisher is None or m.get("publisher") == publisher)
                and (start is None or m["effective_day"] >= start) and (end is None or m["effective_day"] <= end))
    filt = {d for d in (allowed if allowed is not None else range(s.idx.N)) if ok(d)}
    fn = bm25_scores if mode == "bm25" else tfidf_scores
    acc, contrib = fn(s.idx, weights, allowed=filt)
    stems = set(weights)
    k = max(1, min(k, 100))
    # the same article is stored once per ticker it was tagged with: keep the best-scoring copy, list the others
    seen: dict[tuple[str, str], int] = {}
    uniq: list[tuple[int, float]] = []
    extra: dict[int, list[str]] = {}
    # Re-rank the best candidates: (1) headline matches count for more than matches buried in a long body,
    # (2) when the query names a company ("tesla ..."), documents tagged with that ticker get a boost.
    ranked = top_k(acc, max(400, k * 8))
    if not boolean:
        names = {t for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) > 2
                 for t, sc in s.ents.resolve(w, 1) if sc >= 0.9}
        qstems = set(weights)

        def boosted(d: int, sc: float) -> float:
            tstems = set(tokenize(str(s.docs.title.iloc[d])))
            frac = len(qstems & tstems) / len(qstems)
            return sc * (1 + TITLE_BOOST * frac) * (1 + COMPANY_BOOST * (meta[d]["ticker"] in names))
        ranked = sorted(((d, boosted(d, sc)) for d, sc in ranked), key=lambda x: -x[1])
    for d, sc in ranked:
        key = (str(s.docs.title.iloc[d]).strip().lower(), meta[d]["effective_day"])
        if key in seen:
            extra[seen[key]].append(meta[d]["ticker"])
        elif len(uniq) < k:
            seen[key] = d
            extra[d] = []
            uniq.append((d, sc))
    res = []
    for r, (d, sc) in enumerate(uniq, 1):
        row = s.docs.iloc[d]
        text = str(row.text)
        res.append(SearchHit(doc_id=d, rank=r, score=float(sc), contributions={t: float(v) for t, v in contrib[d].items()},
                             doc_type=meta[d]["doc_type"], publisher=clean_str(meta[d].get("publisher")), ticker=meta[d]["ticker"],
                             effective_day=meta[d]["effective_day"], title=str(row.title),
                             snippet=text[:300], matched_terms=_surface_matches(text[:1500], set(contrib[d])),
                             cause_label=clean_str(meta[d].get("cause_label")), url=clean_str(row.url),
                             also_tagged=sorted(set(extra[d]) - {meta[d]["ticker"]})))
    sugg = [f"{t} ({s.names[t]})" for t, sc in s.ents.resolve(q, 3) if sc >= 0.55] if len(q.split()) <= 3 and not boolean else []
    return SearchResponse(query=q, mode=mode, n_candidates=len(filt), expanded_wildcards=expansions, did_you_mean=sugg, results=res)
