import random
import re

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB as SkNB

from whymoved.graphalgo import hits, pagerank
from whymoved.ir.search_bench import corrupt, derive_aliases, make_queries
from whymoved.ir.tolerant import EntityIndex, levenshtein
from whymoved.sectors.classify import MultinomialNB
from whymoved.sectors.graph import knn_graph

COMP = pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA", "NVR", "GOOG"],
                     "name": ["Apple Inc.", "Microsoft Corporation", "Nvidia Corporation", "NVR, Inc.", "Alphabet Inc. (Class C)"]})


def test_corruption_generator_deterministic_and_nontrivial():
    al = derive_aliases(COMP)
    a, b = make_queries(COMP, 42, al), make_queries(COMP, 42, al)
    assert a.equals(b) and len(a) > 20
    assert (a["query"] != a["clean"]).all()
    assert not make_queries(COMP, 7, al).equals(a)
    e1 = a[a.kind == "edit1"]
    assert all(levenshtein(r.clean, r.query) <= 2 for r in e1.itertuples())  # a transposition costs 2 under plain Levenshtein


def test_wildcards_match_regex_ground_truth():
    ent = EntityIndex(COMP.to_dict(orient="records"))
    for pat in ["nv*", "*soft", "a*e", "nvd*", "*", "app*"]:
        rx = re.compile("^" + ".*".join(re.escape(p) for p in pat.split("*")) + "$")
        assert ent.wildcard(pat) == {t for t in ent.terms if rx.match(t)}, pat


def test_pagerank_hits_on_similarity_graph_vs_networkx():
    rng = np.random.default_rng(5)
    F = rng.normal(size=(50, 8))
    Fu = F / np.linalg.norm(F, axis=1, keepdims=True)
    A = knn_graph(np.clip(Fu @ Fu.T, 0, None), 10)
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    assert np.allclose(pagerank(A, 0.85, 1e-12), [nx.pagerank(G, 0.85, tol=1e-12, weight="weight")[i] for i in range(50)], atol=1e-6)
    h, a = hits(A)
    rh, ra = nx.hits(G, max_iter=2000, tol=1e-12)
    assert np.allclose(h, [rh[i] for i in range(50)], atol=1e-5) and np.allclose(a, [ra[i] for i in range(50)], atol=1e-5)


def test_nb_matches_sklearn_on_day_count_vectors():
    rng = np.random.default_rng(3)
    C = sp.csr_matrix(rng.poisson(0.7, size=(90, 60)).astype(float))
    y = rng.integers(0, 6, 90)
    assert (MultinomialNB(1.0).fit(C, y).predict(C) == SkNB(alpha=1.0).fit(C, y).predict(C)).all()
