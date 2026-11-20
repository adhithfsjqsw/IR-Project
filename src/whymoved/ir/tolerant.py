"""Tolerant retrieval: k-gram index, Levenshtein, Soundex, wildcards, entity resolution."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|holdings|holding|plc|ltd|limited|llc|group|"
    r"the|class\s+[a-c]|common\s+stock)\b\.?", re.I)


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insert/delete/substitute, unit cost) by dynamic programming, O(min(m,n)) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def soundex(word: str) -> str:
    """American Soundex code (letter + 3 digits)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    codes = {c: d for d, cs in {"1": "bfpv", "2": "cgjkqsxz", "3": "dt", "4": "l", "5": "mn", "6": "r"}.items()
             for c in cs}
    out = w[0].upper()
    last = codes.get(w[0], "")
    for c in w[1:]:
        d = codes.get(c, "")
        if d and d != last:
            out += d
        if c not in "hw":  # h/w do not separate equal codes
            last = d
    return (out + "000")[:4]


def kgrams(term: str, k: int) -> set[str]:
    """k-grams of term with '$' boundary markers."""
    t = f"${term}$"
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def normalize_name(name: str) -> str:
    """Lowercase, strip corporate suffixes and punctuation."""
    s = _SUFFIXES.sub(" ", name.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


class EntityIndex:
    """Company names / tickers / aliases with 2- and 3-gram indexes."""

    def __init__(self, companies: Iterable[dict[str, str]]) -> None:
        self.terms: dict[str, set[str]] = defaultdict(set)  # surface form -> tickers
        for c in companies:
            t = c["ticker"]
            forms = {t.lower(), normalize_name(c["name"])}
            first = normalize_name(c["name"]).split(" ")[0] if normalize_name(c["name"]) else ""
            if first:
                forms.add(first)
            for f in forms:
                if f:
                    self.terms[f].add(t)
        self.gram: dict[int, dict[str, set[str]]] = {2: defaultdict(set), 3: defaultdict(set)}
        for form in self.terms:
            for k in (2, 3):
                for g in kgrams(form, k):
                    self.gram[k][g].add(form)
        self.sound: dict[str, set[str]] = defaultdict(set)
        for form in self.terms:
            for w in form.split():
                self.sound[soundex(w)].add(form)

    def wildcard(self, pattern: str) -> set[str]:
        """Surface forms matching a wildcard pattern (* only) via k-gram index + regex post-filter."""
        pat = pattern.lower()
        pieces = [f"${p}" if i == 0 else p for i, p in enumerate(pat.split("*"))]
        pieces[-1] = pieces[-1] + "$" if not pat.endswith("*") else pieces[-1]
        cands: set[str] | None = None
        for piece in pieces:
            for g in (piece[i:i + 3] for i in range(max(0, len(piece) - 2))):
                s = self.gram[3].get(g, set())
                cands = set(s) if cands is None else cands & s
        if cands is None:
            cands = set(self.terms)
        rx = re.compile("^" + ".*".join(re.escape(p) for p in pat.split("*")) + "$")
        return {c for c in cands if rx.match(c)}

    def resolve(self, text: str, top: int = 5) -> list[tuple[str, float]]:
        """Ranked (ticker, score in [0,1]) for a possibly misspelled / wildcard / phonetic query."""
        q = normalize_name(text) if "*" not in text else text.lower().strip()
        scores: dict[str, float] = defaultdict(float)
        if "*" in q:
            for form in self.wildcard(q):
                for t in self.terms[form]:
                    scores[t] = max(scores[t], 0.9)
        else:
            cand: dict[str, float] = {}
            qg = {k: kgrams(q, k) for k in (2, 3)}
            for k in (2, 3):
                for g in qg[k]:
                    for form in self.gram[k].get(g, ()):
                        cand.setdefault(form, 0.0)
            for form in cand:
                j = max(len(qg[k] & kgrams(form, k)) / len(qg[k] | kgrams(form, k)) for k in (2, 3))
                if j < 0.2:
                    continue
                d = levenshtein(q, form)
                sim = 1 - d / max(len(q), len(form), 1)
                s = 0.7 * sim + 0.3 * j
                if form == q:
                    s = 1.0
                for t in self.terms[form]:
                    scores[t] = max(scores[t], s)
            for w in q.split():
                for form in self.sound.get(soundex(w), ()):
                    for t in self.terms[form]:
                        scores[t] = max(scores[t], 0.5 + 0.3 * (1 - levenshtein(q, form) / max(len(q), len(form), 1)))
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top]
