"""Candidate pool for a move: documents effective in {t-1, t} (optionally ticker-filtered)."""
from __future__ import annotations

import re

from ..ir.index import PositionalIndex
from ..ir.scoring import allowed_docs

# headlines that merely report the price move itself ("shares jump", "stock falls 5%", ...)
PRICE_MOVE_RE = re.compile(
    r"\b(shares?|stock|stocks)\b.*\b(jump|jumps|jumped|fall|falls|fell|surge|surges|surged|plunge|plunges|"
    r"plunged|soar|soars|soared|tumble|tumbles|tumbled|slide|slides|slid|sink|sinks|sank|rally|rallies|"
    r"rallied|gain|gains|gained|drop|drops|dropped|trading (higher|lower)|moving (higher|lower))\b|"
    r"\b\d+(\.\d+)?\s?%", re.I)


def window_days(cal: list[str], pos: dict[str, int], day: str, back: int = 1) -> list[str]:
    i = pos[day]
    return cal[max(0, i - back): i + 1]


def candidate_pool(idx: PositionalIndex, days: list[str], ticker: str | None,
                   exclude: set[int] | None = None) -> set[int]:
    pool = allowed_docs(idx, days, ticker) or set()
    return pool - exclude if exclude else pool


def price_move_docs(idx: PositionalIndex) -> set[int]:
    """Doc ids of news whose title looks like a price-move report."""
    return {d for d, m in enumerate(idx.meta)
            if m.get("doc_type") == "news" and PRICE_MOVE_RE.search(m.get("title") or "")}
