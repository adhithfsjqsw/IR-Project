"""Query builder: company-name tokens + ticker + direction-expansion terms at reduced weight."""
from __future__ import annotations

from .tokenize import tokenize
from .tolerant import normalize_name

UP_LEXICON = ["beat", "beats", "raises", "raised", "upgrade", "upgraded", "approval", "approved", "record",
              "surge", "strong", "growth", "contract", "wins", "award", "dividend", "buyback", "partnership"]
DOWN_LEXICON = ["miss", "misses", "missed", "cut", "cuts", "downgrade", "downgraded", "probe", "investigation",
                "lawsuit", "recall", "warning", "weak", "delay", "resigns", "layoffs", "restructuring",
                "impairment", "bankruptcy"]


def build_query(name: str, ticker: str, direction: int | None = None,
                direction_weight: float = 0.4) -> dict[str, float]:
    """Return {stemmed term: weight}. Name+ticker terms get weight 1; direction terms get `direction_weight`.

    direction: +1 up move, -1 down move, None/0 = no expansion.
    """
    q: dict[str, float] = {}
    for t in tokenize(normalize_name(name)) + tokenize(ticker):
        q[t] = 1.0
    if direction:
        for w in (UP_LEXICON if direction > 0 else DOWN_LEXICON):
            for t in tokenize(w):
                q[t] = max(q.get(t, 0.0), direction_weight)
    return q
