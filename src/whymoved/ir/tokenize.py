"""Tokenizer pipeline: lowercase -> [a-z0-9]+ -> stop-word removal -> Porter stem (numbers kept)."""
from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Porter-stem a single word (cached); pure numbers are kept as-is."""
    return word if word.isdigit() else _STEMMER.stem(word)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on [a-z0-9]+, drop stop words, Porter-stem.

    Positions used by the index are indices into the surviving token list, so
    documents and phrase queries are tokenized consistently.
    """
    toks = _TOKEN_RE.findall((text or "").lower())
    return [stem(t) for t in toks if t not in STOP_WORDS]
