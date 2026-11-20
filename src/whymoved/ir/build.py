"""Build and persist the positional index over docs.parquet: `python -m whymoved.ir.build`."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from ..common import PROCESSED, get_logger
from .index import PositionalIndex

log = get_logger("ir.build")
INDEX_PATH = PROCESSED / "index.pkl"


def build_index(docs: pd.DataFrame) -> PositionalIndex:
    """Index every doc; internal doc id == docs.doc_id (rows are ordered by doc_id)."""
    assert (docs.doc_id.to_numpy() == range(len(docs))).all()
    idx = PositionalIndex()
    for r in tqdm(docs.itertuples(), total=len(docs), desc="index"):
        idx.add_document(r.text, {"ticker": r.ticker, "doc_type": r.doc_type, "publisher": r.publisher,
                                  "effective_day": r.effective_day, "cause_label": r.cause_label,
                                  "title": r.title})
    idx.finalize()
    return idx


def load_or_build() -> PositionalIndex:
    if INDEX_PATH.exists():
        return PositionalIndex.load(INDEX_PATH)
    idx = build_index(pd.read_parquet(PROCESSED / "docs.parquet"))
    idx.save(INDEX_PATH)
    log.info("index: %d docs, %d terms", idx.N, len(idx.postings))
    return idx


if __name__ == "__main__":
    load_or_build()
