"""Pydantic response/request models for the WhyMoved API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def clean_str(x) -> str | None:
    """Parquet/pandas missing values (NaN, None, pd.NA) -> None, otherwise str. Response models want str | None."""
    import pandas as pd
    return None if x is None or (not isinstance(x, str) and pd.isna(x)) else str(x)


class Entity(BaseModel):
    ticker: str
    name: str
    sector: str
    score: float


class Move(BaseModel):
    ticker: str
    date: str
    ret: float
    abnormal: float
    z: float
    direction: int
    has_8k: bool


class Explanation(BaseModel):
    doc_id: int
    rank: int
    score: float
    doc_type: str
    publisher: str | None
    effective_day: str
    title: str
    snippet: str
    cause_label: str | None
    cause_source: str = Field(description="'8-K items' for filings, 'classifier' for predicted news labels")
    matched_terms: list[str]
    is_gold: bool
    url: str | None = None


class ExplainResponse(BaseModel):
    ticker: str
    date: str
    mode: str
    query_terms: dict[str, float]
    unexplained: bool
    tau: float | None
    results: list[Explanation]


class FeedbackRequest(BaseModel):
    ticker: str
    date: str
    mode: str = "easy"
    relevant: list[int] = []
    nonrelevant: list[int] = []


class PeerOut(BaseModel):
    ticker: str
    name: str
    sector: str
    similarity: float
    same_sector: bool


class PeersResponse(BaseModel):
    ticker: str
    name: str
    official_sector: str
    sub_industry: str
    kmeans_cluster: int
    peer_purity: float
    peers: list[PeerOut]
    gics_sector_members_sample: list[str]


class Any_(BaseModel):
    data: Any
