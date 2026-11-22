import math

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer

from whymoved.ir.feedback import rocchio
from whymoved.ir.index import PositionalIndex
from whymoved.ir.query import build_query
from whymoved.ir.scoring import bm25_scores, tfidf_scores
from whymoved.ir.search import Searcher
from whymoved.ir.tokenize import tokenize
from whymoved.ir.tolerant import EntityIndex, levenshtein, soundex

DOCS = [
    "The Federal Reserve raised the lending rate again",
    "lending in the rate of growth remains strong",
    "Apple announces record iPhone sales and strong growth",
    "Microsoft cuts jobs amid restructuring",
    "lending rate hike worries investors",
]


def make_index():
    idx = PositionalIndex()
    for i, t in enumerate(DOCS):
        idx.add_document(t, {"ticker": "X", "effective_day": f"2021-01-0{i % 2 + 1}"})
    idx.finalize()
    return idx


def test_tokenizer():
    assert tokenize("The Running dogs, 2021!") == ["run", "dog", "2021"]


def test_postings_positions_phrase_boolean():
    idx = make_index()
    assert idx.df("lend") == 3
    pos = idx.postings["lend"][0]
    assert pos == sorted(pos)
    # doc 1 ("lending in the rate") is adjacent once stop words are removed - by design
    assert idx.phrase("lending rate") == {0, 1, 4}
    assert idx.boolean("lending AND rate") == {0, 1, 4}
    assert idx.boolean("lending AND NOT hike") == {0, 1}
    assert idx.boolean("apple OR microsoft") == {2, 3}
    assert idx.boolean('"lending rate" AND hike') == {4}


def test_persistence(tmp_path):
    idx = make_index()
    idx.save(tmp_path / "i.pkl")
    j = PositionalIndex.load(tmp_path / "i.pkl")
    assert j.postings == idx.postings and j.N == idx.N


def _corpus():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(60)]
    return [" ".join(rng.choice(vocab, size=rng.integers(15, 60))) for _ in range(80)]


def test_tfidf_matches_sklearn_formula():
    docs = _corpus()
    idx = PositionalIndex()
    for d in docs:
        idx.add_document(d)
    idx.finalize()
    q = {"w1": 1.0, "w2": 1.0, "w3": 1.0}
    acc, _ = tfidf_scores(idx, q)
    # sklearn supplies counts; tf=1+log10(tf), idf=log10(N/df) applied by hand (sklearn's sublinear tf uses ln)
    tv = CountVectorizer(token_pattern=r"w\d+")
    C = tv.fit_transform(docs).toarray().astype(float)
    vocab = tv.vocabulary_
    df = (C > 0).sum(0)
    idf = np.log10(len(docs) / np.maximum(df, 1))
    X = np.where(C > 0, 1 + np.log10(np.maximum(C, 1)), 0.0)
    Xw = X * idf
    Xw /= np.maximum(np.linalg.norm(Xw, axis=1, keepdims=True), 1e-12)
    qv = np.zeros(X.shape[1])
    for t in q:
        qv[vocab[t]] = idf[vocab[t]]
    qv /= np.linalg.norm(qv)
    ref = Xw @ qv
    mine = np.array([acc.get(i, 0.0) for i in range(len(docs))])
    assert np.allclose(mine, ref, atol=1e-9)


def test_bm25_matches_rank_bm25():
    docs = _corpus()
    idx = PositionalIndex()
    for d in docs:
        idx.add_document(d)
    q = ["w1", "w2", "w5"]
    acc, _ = bm25_scores(idx, {t: 1.0 for t in q}, k1=1.2, b=0.75)
    bm = BM25Okapi([d.split() for d in docs], k1=1.2, b=0.75)
    N = len(docs)  # inject the spec's idf: rank_bm25's own idf variant can be negative
    bm.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in
              ((t, sum(t in d.split() for d in docs)) for t in bm.idf)}
    ref = bm.get_scores(q)
    mine = np.array([acc.get(i, 0.0) for i in range(len(docs))])
    assert spearmanr(mine, ref).correlation >= 0.99
    assert np.allclose(mine, ref, atol=1e-9)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0
    rng = np.random.default_rng(1)
    import difflib  # sanity: symmetric & triangle
    for _ in range(50):
        a, b, c = ("".join(rng.choice(list("abc"), size=rng.integers(0, 8))) for _ in range(3))
        assert levenshtein(a, b) == levenshtein(b, a)
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)


def test_soundex():
    assert soundex("Robert") == "R163" and soundex("Rupert") == "R163"
    assert soundex("Tymczak") == "T522" and soundex("Pfister") == "P236"


COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "NVDA", "name": "Nvidia Corporation"}, {"ticker": "NVR", "name": "NVR, Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
]


def test_kgram_misspellings_and_wildcard():
    e = EntityIndex(COMPANIES)
    assert e.resolve("Appel")[0][0] == "AAPL"
    assert e.resolve("Microsft")[0][0] == "MSFT"
    assert e.resolve("Nvidea")[0][0] == "NVDA"
    assert {t for t, _ in e.resolve("NVD*")} == {"NVDA"}
    assert {t for t, _ in e.resolve("NV*")} >= {"NVDA", "NVR"}


def test_rocchio_toy():
    idx = make_index()
    q = {"appl": 1.0}
    q2 = rocchio(idx, q, [2], alpha=1.0, beta=1.0, n_terms=3)
    assert q2["appl"] > 1.0 - 1e-9 and len(q2) == 4
    assert "record" in q2 or "iphon" in q2


def test_search_window_and_time_prior():
    idx = make_index()
    s = Searcher(idx)
    q = {"lend": 1.0, "rate": 1.0}
    hits = s.search(q, ["2021-01-01", "2021-01-02"], k=5)
    assert hits and hits[0].score >= hits[-1].score
    h2 = s.search(q, ["2021-01-01"], k=5)
    assert all(idx.meta[h.doc_id]["effective_day"] == "2021-01-01" for h in h2)
    h3 = s.search(q, ["2021-01-01", "2021-01-02"], k=5, time_lambda=2.0, event_day="2021-01-02")
    assert h3[0].score <= hits[0].score


def test_query_builder():
    q = build_query("Apple Inc.", "AAPL", direction=1, direction_weight=0.4)
    assert q["appl"] == 1.0 and q["beat"] == 0.4
    qd = build_query("Apple Inc.", "AAPL", direction=-1)
    assert "lawsuit" in qd and "recal" in qd and "beat" not in qd


def test_event_lexicon_and_type_weights():
    from whymoved.attribution.ablations import expected_random_mrr
    from whymoved.ir.index import PositionalIndex
    from whymoved.ir.query import build_query
    from whymoved.ir.search import Searcher
    q0, q1 = build_query("Acme Corp", "ACM"), build_query("Acme Corp", "ACM", event_weight=0.3)
    assert set(q0) < set(q1) and all(q1[t] == 0.3 for t in set(q1) - set(q0))
    idx = PositionalIndex()
    a = idx.add_document("acme results", {"doc_type": "news", "effective_day": "2022-01-03"})
    b = idx.add_document("acme results", {"doc_type": "8k", "effective_day": "2022-01-03"})
    from whymoved.ir.tokenize import tokenize
    s, qt = Searcher(idx), {tokenize("acme")[0]: 1.0}
    assert [h.doc_id for h in s.search(qt, None, k=2)] == [a, b]            # tie -> lower id
    assert s.search(qt, None, k=2, type_weights={"8k": 2.0})[0].doc_id == b  # prior reorders
    # H_n/n: n=1 -> 1.0; n=2 -> (1 + 1/2)/2 = 0.75
    assert expected_random_mrr([1]) == 1.0 and abs(expected_random_mrr([2]) - 0.75) < 1e-12


def _burst_index():
    from whymoved.ir.index import PositionalIndex
    idx = PositionalIndex()
    base = ["acme sells widgets", "acme widgets market", "acme quarterly widgets review"]
    for i, t in enumerate(base):  # baseline days: generic coverage
        idx.add_document(t, {"ticker": "ACM", "doc_type": "news", "effective_day": f"2022-01-0{i + 1}"})
    ev = ["acme recall recall of widgets", "acme announces recall after safety probe"]
    for t in ev:                  # event day: a recall story
        idx.add_document(t, {"ticker": "ACM", "doc_type": "news", "effective_day": "2022-02-01"})
    idx.add_document("other recall recall news", {"ticker": "ZZZ", "doc_type": "news", "effective_day": "2022-02-01"})
    return idx


def test_burst_terms_find_event_vocabulary_not_baseline_or_other_tickers():
    from whymoved.ir.burst import burst_terms
    from whymoved.ir.tokenize import tokenize
    idx = _burst_index()
    bt = burst_terms(idx, "ACM", ["2022-02-01"], ["2022-01-01", "2022-01-02", "2022-01-03"], n=5,
                     exclude={tokenize("acme")[0]}, min_df=1)
    assert tokenize("recall")[0] in bt and bt[tokenize("recall")[0]] == 1.0
    assert tokenize("widgets")[0] not in bt                      # common in the baseline -> not a burst
    assert burst_terms(idx, "ACM", ["2030-01-01"], [], 5) == {}   # no documents in the window -> no evidence


def test_judged_metrics_and_label_parsing():
    from whymoved.attribution.judged import metrics, parse_label
    assert parse_label("YES") == "YES" and parse_label(" partial.") == "PARTIAL" and parse_label("maybe") is None
    q = {1: 2, 2: 0, 3: 1}
    m = metrics([1, 3, 2], q)                       # best docs first -> perfect nDCG
    assert abs(m["ndcg10"] - 1.0) < 1e-9 and m["mrr"] == 1.0 and m["p5"] == 0.2
    assert metrics([2, 3, 1], q)["mrr"] == 1 / 3     # first grade-2 doc is at rank 3
    assert metrics([1], {1: 0, 2: 1}) is None        # nothing graded 2 -> move is skipped
