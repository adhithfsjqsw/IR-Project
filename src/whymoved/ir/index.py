"""Positional inverted index with phrase and boolean queries, persisted with pickle."""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .tokenize import tokenize


class PositionalIndex:
    """term -> {doc_id: [sorted positions]} plus doc lengths, metadata and per-day doc lists."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[int, list[int]]] = defaultdict(dict)
        self.doc_len: list[int] = []
        self.meta: list[dict[str, Any]] = []
        self.doc_terms: list[Counter] = []  # forward index (used by Rocchio)
        self.day_docs: dict[str, list[int]] = defaultdict(list)
        self.norms: list[float] = []
        self._finalized = False

    # ---- building -------------------------------------------------------
    def add_document(self, text: str, meta: dict[str, Any] | None = None) -> int:
        """Add a document; returns its integer doc id. Positions are appended in order, hence sorted."""
        did = len(self.doc_len)
        toks = tokenize(text)
        for pos, t in enumerate(toks):
            self.postings[t].setdefault(did, []).append(pos)
        self.doc_len.append(len(toks))
        self.doc_terms.append(Counter(toks))
        meta = dict(meta or {})
        self.meta.append(meta)
        if "effective_day" in meta:
            self.day_docs[str(meta["effective_day"])[:10]].append(did)
        self._finalized = False
        return did

    @property
    def N(self) -> int:
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        return sum(self.doc_len) / max(1, self.N)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def tf(self, term: str, did: int) -> int:
        return len(self.postings.get(term, {}).get(did, ()))

    def idf_tfidf(self, term: str) -> float:
        """idf = log10(N/df); 0 for unseen terms."""
        d = self.df(term)
        return math.log10(self.N / d) if d else 0.0

    def finalize(self) -> None:
        """Precompute TF-IDF document norms (tf = 1+log10 tf, idf = log10 N/df)."""
        sq = [0.0] * self.N
        for term, plist in self.postings.items():
            idf = self.idf_tfidf(term)
            for did, pos in plist.items():
                w = (1 + math.log10(len(pos))) * idf
                sq[did] += w * w
        self.norms = [math.sqrt(s) for s in sq]
        self._finalized = True

    # ---- queries --------------------------------------------------------
    def docs_with(self, term: str) -> set[int]:
        return set(self.postings.get(term, {}).keys())

    def phrase(self, text: str) -> set[int]:
        """Docs containing the tokenized phrase at consecutive positions."""
        terms = tokenize(text)
        if not terms:
            return set()
        cands = self.docs_with(terms[0])
        for t in terms[1:]:
            cands &= self.docs_with(t)
        out: set[int] = set()
        for d in cands:
            later = [set(self.postings[t][d]) for t in terms[1:]]
            for s in self.postings[terms[0]][d]:
                if all((s + i + 1) in later[i] for i in range(len(later))):
                    out.add(d)
                    break
        return out

    def boolean(self, query: str) -> set[int]:
        """Evaluate AND / OR / NOT with quoted phrases and parentheses.

        Precedence NOT > AND > OR; adjacent operands imply AND.
        """
        toks = re.findall(r'"[^"]+"|\(|\)|[^\s()"]+', query)
        pos = 0
        universe = set(range(self.N))

        def peek() -> str | None:
            return toks[pos] if pos < len(toks) else None

        def parse_or() -> set[int]:
            nonlocal pos
            r = parse_and()
            while peek() == "OR":
                pos += 1
                r = r | parse_and()
            return r

        def parse_and() -> set[int]:
            nonlocal pos
            r = parse_not()
            while peek() is not None and peek() not in ("OR", ")"):
                if peek() == "AND":
                    pos += 1
                r = r & parse_not()
            return r

        def parse_not() -> set[int]:
            nonlocal pos
            if peek() == "NOT":
                pos += 1
                return universe - parse_not()
            return parse_atom()

        def parse_atom() -> set[int]:
            nonlocal pos
            t = peek()
            pos += 1
            if t == "(":
                r = parse_or()
                pos += 1  # consume ')'
                return r
            if t is not None and t.startswith('"'):
                return self.phrase(t.strip('"'))
            ts = tokenize(t or "")
            return self.docs_with(ts[0]) if ts else set()

        return parse_or()

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        state = dict(self.__dict__)
        state["postings"] = dict(self.postings)
        state["day_docs"] = dict(self.day_docs)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PositionalIndex":
        idx = cls()
        with open(path, "rb") as f:
            idx.__dict__.update(pickle.load(f))
        return idx
