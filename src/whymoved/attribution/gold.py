"""Gold lookup: (ticker, day) -> set of 8-K doc ids effective that day."""
from __future__ import annotations

import pandas as pd


def gold_lookup(gold: pd.DataFrame) -> dict[tuple[str, str], set[int]]:
    return {(r.ticker, r.date): set(r.doc_ids) for r in gold.itertuples()}
