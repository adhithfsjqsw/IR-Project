"""BM25 index over shock-mode day documents (cached). `python -m whymoved.shock.dayindex`."""
from __future__ import annotations

import pandas as pd

from ..common import PROCESSED, get_logger
from ..ir.index import PositionalIndex
from .daydocs import build_day_docs

log = get_logger("shock.dayindex")
PATH = PROCESSED / "day_index.pkl"


def load_day_index() -> PositionalIndex:
    if PATH.exists():
        return PositionalIndex.load(PATH)
    docs = build_day_docs()
    idx = PositionalIndex()
    for r in docs.itertuples():
        idx.add_document(r.text, {"effective_day": r.day, "ticker": "", "doc_type": "day"})
    idx.finalize()
    idx.save(PATH)
    log.info("day index: %d docs, %d terms", idx.N, len(idx.postings))
    return idx


if __name__ == "__main__":
    load_day_index()
