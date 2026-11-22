import pytest

from whymoved.rag.answer import answer_move, build_prompt, check_citations
from whymoved.rag.llm import LLMError, make_backend

SRC = [{"doc_type": "news", "effective_day": "2022-01-03", "title": "Acme beats earnings", "text": "Acme reported."},
       {"doc_type": "8k", "effective_day": "2022-01-03", "title": "8-K", "text": "Item 2.02 results."}]


class Fake:
    def __init__(self, out):
        self.out, self.seen = out, None

    def generate(self, system, prompt):
        self.seen = (system, prompt)
        return self.out


def test_prompt_numbers_sources_and_includes_move():
    p = build_prompt("ACM", "Acme", "2022-01-03", {"ret": 0.05, "abnormal": 0.04, "z": 3.2}, SRC, 1200)
    assert "[1] (news" in p and "[2] (SEC 8-K" in p and "+5.00%" in p


def test_prompt_truncates_text():
    p = build_prompt("A", "A", "d", None, [{**SRC[0], "text": "x" * 5000}], 100)
    assert "x" * 100 in p and "x" * 101 not in p


def test_citation_check_flags_invalid():
    assert check_citations("Up on earnings [1][2]; also [7].", 2) == {"cited": [1, 2], "invalid": [7]}


def test_answer_uses_llm_and_reports_citations():
    f = Fake("Earnings beat [1].")
    out = answer_move(f, "ACM", "Acme", "2022-01-03", None, SRC)
    assert out["cited"] == [1] and out["invalid"] == [] and "ONLY" in f.seen[0]


def test_no_sources_skips_llm():
    f = Fake("should not be called")
    out = answer_move(f, "ACM", "Acme", "2022-01-03", None, [])
    assert f.seen is None and out["cited"] == []


def test_unknown_backend():
    with pytest.raises(LLMError):
        make_backend({"backend": "nope"})


def test_clean_str_maps_missing_values_to_none():
    import numpy as np
    import pandas as pd

    from whymoved.api.schemas import clean_str
    assert clean_str(float("nan")) is None and clean_str(None) is None and clean_str(pd.NA) is None
    assert clean_str(np.nan) is None and clean_str("x") == "x" and clean_str("") == ""
