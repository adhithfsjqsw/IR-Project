"""GET /api/price: a stock's price path around a news date (rebased to 100 on the close before the event day)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..common import PROCESSED, load_config

router = APIRouter()


class PricePoint(BaseModel):
    date: str
    rel: float            # close rebased to 100 at the close before the event day
    ret: float | None     # that day's return
    market_ret: float | None
    abnormal: float | None  # ret - market (simple market-adjusted)
    is_event: bool


class PriceResponse(BaseModel):
    ticker: str
    event_day: str
    market: str
    event_ret: float | None
    event_abnormal: float | None
    after_3d_abnormal: float | None   # cumulative market-adjusted return, event day through +2 trading days
    flagged_move: bool                # the project's market model flagged this day as an abnormal move
    points: list[PricePoint]


@lru_cache(maxsize=1)
def _prices() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    mkt = load_config()["market_proxy"]
    p = pd.read_parquet(PROCESSED / "prices.parquet")
    wide_ret = p.pivot(index="date", columns="ticker", values="ret").sort_index()
    wide_px = p.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return wide_ret, wide_px, mkt


def _f(x) -> float | None:
    return None if pd.isna(x) else float(x)


@router.get("/api/price", response_model=PriceResponse)
def price(ticker: str, date: str, before: int = 5, after: int = 5):
    ret, px, mkt = _prices()
    t = ticker.upper()
    if t not in px.columns:
        raise HTTPException(404, f"no price data for {t}")
    d = pd.Timestamp(date)
    days = list(px.index)
    if d < days[0] or d > days[-1]:
        raise HTTPException(400, f"{date} is outside the price data ({days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d})")
    i = max(i for i, x in enumerate(days) if x <= d)  # event day, or the last trading day before a non-trading date
    lo, hi = max(1, i - max(0, min(before, 20))), min(len(days) - 1, i + max(0, min(after, 20)))
    base = px[t].iloc[i - 1]
    pts = []
    for j in range(lo, hi + 1):
        r, m = _f(ret[t].iloc[j]), _f(ret[mkt].iloc[j])
        pts.append(PricePoint(date=f"{days[j]:%Y-%m-%d}", rel=float(px[t].iloc[j] / base * 100), ret=r, market_ret=m,
                              abnormal=None if r is None or m is None else r - m, is_event=j == i))
    ev = next(p for p in pts if p.is_event)
    win = [p.abnormal for p in pts if p.date >= ev.date][:3]
    moves = pd.read_parquet(PROCESSED / "moves.parquet")
    flagged = bool(((moves.ticker == t) & (moves.date == days[i])).any())
    return PriceResponse(ticker=t, event_day=ev.date, market=mkt, event_ret=ev.ret, event_abnormal=ev.abnormal,
                         after_3d_abnormal=None if len(win) < 3 or None in win else float(sum(win)),
                         flagged_move=flagged, points=pts)
