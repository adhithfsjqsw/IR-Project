"""GET /api/rag: retrieve with the BM25 pipeline, then generate a cited explanation with the configured LLM."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..rag.answer import answer_move
from ..rag.llm import LLMError, make_backend
from .schemas import Explanation

router = APIRouter()


class RagResponse(BaseModel):
    ticker: str
    date: str
    mode: str
    model: str
    answer: str
    cited: list[int]
    invalid_citations: list[int]
    unexplained: bool
    sources: list[Explanation]


@router.get("/api/rag", response_model=RagResponse)
def rag(ticker: str, date: str, mode: str = "easy", top_k: int | None = None):
    from .main import _explain, state
    s = state()
    cfg = s.cfg.get("rag", {})
    k = top_k or cfg.get("top_k", 5)
    ex = _explain(ticker, date, mode)
    hits = ex.results[:k]
    sources = [{"doc_type": h.doc_type, "effective_day": h.effective_day, "title": h.title,
                "text": str(s.docs.iloc[h.doc_id].text)} for h in hits]
    mv = s.moves[(s.moves.ticker == ex.ticker) & (s.moves.date == pd.Timestamp(date))]
    move = None if mv.empty else {"ret": float(mv.ret.iloc[0]), "abnormal": float(mv.abnormal.iloc[0]),
                                  "z": float(mv.z.iloc[0])}
    try:
        llm = make_backend(cfg)
        out = answer_move(llm, ex.ticker, s.names[ex.ticker], date, move, sources, cfg.get("max_chars", 1200))
    except LLMError as e:
        raise HTTPException(503, str(e))
    return RagResponse(ticker=ex.ticker, date=date, mode=mode, model=getattr(llm, "model", "?"), answer=out["answer"],
                       cited=out["cited"], invalid_citations=out["invalid"], unexplained=ex.unexplained, sources=hits)
