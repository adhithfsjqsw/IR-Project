import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.naive_bayes import MultinomialNB as SkNB

from whymoved.sectors.classify import KNNClassifier, MultinomialNB, RocchioClassifier, scores
from whymoved.sectors.graph import hits, knn_graph, pagerank
from whymoved.sectors.kmeans import spherical_kmeans
from whymoved.sectors.vectors import TfidfVectorizer


def _blobs(seed=0, k=4, n=30, d=20):
    rng = np.random.default_rng(seed)
    centers = np.abs(rng.normal(size=(k, d))) * (rng.random((k, d)) < 0.3)
    X = np.vstack([np.abs(c + 0.05 * rng.normal(size=(n, d))) for c in centers])
    return X, np.repeat(np.arange(k), n)


def test_nb_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.poisson(1.0, size=(120, 40)).astype(float)
    y = rng.integers(0, 3, 120)
    X[y == 1, :5] += rng.poisson(3, size=((y == 1).sum(), 5))
    mine = MultinomialNB(1.0).fit(sp.csr_matrix(X), y)
    ref = SkNB(alpha=1.0).fit(X, y)
    assert (mine.predict(sp.csr_matrix(X)) == ref.predict(X)).all()
    assert np.allclose(mine.predict_log_proba_unnorm(sp.csr_matrix(X)), ref.predict_joint_log_proba(X), atol=1e-8)


def test_rocchio_knn_on_blobs():
    X, y = _blobs()
    Xs = sp.csr_matrix(X)
    assert (RocchioClassifier().fit(Xs, y).predict(Xs) == y).mean() > 0.99
    assert (KNNClassifier(5).fit(Xs, y).predict(Xs) == y).mean() > 0.99
    s = scores(y, y, [0, 1, 2, 3])
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0


def test_kmeans_vs_sklearn():
    X, y = _blobs()
    lab, C, obj = spherical_kmeans(sp.csr_matrix(X), 4, n_init=5, seed=0)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    ref = KMeans(4, n_init=5, random_state=0).fit(Xn)
    assert adjusted_rand_score(y, lab) > 0.95
    assert adjusted_rand_score(ref.labels_, lab) > 0.95
    assert abs(obj - (1 - ref.inertia_ / (2 * len(Xn)))) < 1e-3  # unit vectors: ||x-c||^2 = 2 - 2cos


def test_pagerank_hits_vs_networkx():
    rng = np.random.default_rng(3)
    S = rng.random((40, 40))
    S = (S + S.T) / 2
    A = knn_graph(S, 5)
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    pr = pagerank(A, 0.85, 1e-10)
    ref = nx.pagerank(G, alpha=0.85, tol=1e-12, weight="weight")
    assert np.allclose(pr, [ref[i] for i in range(40)], atol=1e-6)
    h, a = hits(A)
    rh, ra = nx.hits(G, max_iter=1000, tol=1e-12, normalized=True)
    assert np.allclose(h, [rh[i] for i in range(40)], atol=1e-5)
    assert np.allclose(a, [ra[i] for i in range(40)], atol=1e-5)


def test_tfidf_vectorizer():
    docs = ["apple makes phones", "apple sells phones and tablets", "bank offers loans", "bank offers credit loans"]
    v = TfidfVectorizer(min_df=2, max_df=0.75, ngram_max=2).fit(docs)
    X = v.transform(docs)
    assert np.allclose(np.asarray(X.multiply(X).sum(axis=1)).ravel(), 1.0)
    assert "appl" in v.vocab_ and "bank" in v.vocab_ and "bank offer" in v.vocab_ and "offer loan" not in v.vocab_
