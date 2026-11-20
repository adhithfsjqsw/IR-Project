"""Phase 5: text clusters vs GICS, classification, hidden peers, graph centrality. `python -m whymoved.sectors.run`."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold

from ..common import PROCESSED, REPORTS, get_logger, load_config, seed_all
from ..reporting import save_table, update_results
from .classify import KNNClassifier, MultinomialNB, RocchioClassifier, scores
from .graph import hits, knn_graph, pagerank
from .hidden_peers import hidden_peers, peer_purity, peers, similarity
from .hierarchical import dendrogram_figure, hierarchical
from .kmeans import spherical_kmeans
from .vectors import TfidfVectorizer

log = get_logger("sectors")
FIG = REPORTS / "figures"


def purity(true: np.ndarray, pred: np.ndarray) -> float:
    """Cluster purity: sum over clusters of the majority true-class count, divided by n."""
    return float(sum(pd.Series(true[pred == c]).value_counts().iloc[0] for c in np.unique(pred)) / len(true))


def eval_clustering(labels: np.ndarray, sector: np.ndarray, sub: np.ndarray, rng: np.random.Generator) -> dict:
    out = {}
    for name, truth in (("sector", sector), ("sub_industry", sub)):
        out[name] = {"nmi": float(normalized_mutual_info_score(truth, labels)),
                     "ari": float(adjusted_rand_score(truth, labels)), "purity": purity(truth, labels)}
        sh = [(normalized_mutual_info_score(rng.permutation(truth), labels),
               adjusted_rand_score(rng.permutation(truth), labels), purity(rng.permutation(truth), labels))
              for _ in range(100)]
        m = np.mean(sh, axis=0)
        out[name]["shuffled_baseline"] = {"nmi": float(m[0]), "ari": float(m[1]), "purity": float(m[2])}
    return out


def classification_cv(texts: list[str], y: np.ndarray, seed: int) -> dict:
    """5-fold stratified CV; vectorizer fitted inside each training fold (no leakage)."""
    labels = sorted(set(y))
    preds = {n: np.empty(len(y), dtype=object) for n in ("nb", "rocchio", "knn")}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
        vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_max=2).fit([texts[i] for i in tr])
        Ctr, Cte = vec.counts([texts[i] for i in tr]), vec.counts([texts[i] for i in te])
        Xtr, Xte = vec.tfidf_from_counts(Ctr), vec.tfidf_from_counts(Cte)
        preds["nb"][te] = MultinomialNB(1.0).fit(Ctr, y[tr]).predict(Cte)
        preds["rocchio"][te] = RocchioClassifier().fit(Xtr, y[tr]).predict(Xte)
        preds["knn"][te] = KNNClassifier(5).fit(Xtr, y[tr]).predict(Xte)
    return {n: scores(y, p.astype(str), labels) for n, p in preds.items()}


def main() -> None:
    cfg = load_config()
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    FIG.mkdir(parents=True, exist_ok=True)
    comp = pd.read_parquet(PROCESSED / "companies.parquet").merge(
        pd.read_parquet(PROCESSED / "descriptions.parquet"), on="ticker").reset_index(drop=True)
    texts = comp.description.tolist()
    vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_max=2)
    X = vec.fit_transform(texts)
    log.info("TF-IDF: %d docs x %d terms", *X.shape)
    sector, sub = comp.gics_sector.to_numpy(), comp.gics_sub_industry.to_numpy()
    S = similarity(X)
    nn = peers(S, cfg["peer_k"])
    pur = peer_purity(nn, sector)
    res: dict = {"n_companies": len(comp), "n_terms": int(X.shape[1]), "mean_peer_purity": float(pur.mean())}
    # random baseline for peer purity: share of same-sector pairs
    vc = pd.Series(sector).value_counts(normalize=True)
    res["random_peer_purity"] = float((vc ** 2).sum())

    # k-means at k=11 and scan
    k = cfg["n_clusters"]
    lab, _, obj = spherical_kmeans(X, k, 10, seed=cfg["seed"])
    res["kmeans_k11"] = eval_clustering(lab, sector, sub, rng)
    res["kmeans_k11"]["objective_mean_cosine"] = obj
    scan = []
    for kk in range(5, 31):
        l, _, o = spherical_kmeans(X, kk, 10, seed=cfg["seed"])
        scan.append({"k": kk, "silhouette": float(silhouette_score(X, l, metric="cosine")),
                     "nmi_sector": float(normalized_mutual_info_score(sector, l)),
                     "ari_sector": float(adjusted_rand_score(sector, l)), "objective": o})
    scan_df = pd.DataFrame(scan)
    save_table(scan_df, "kmeans_scan")
    res["kmeans_scan"] = scan
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scan_df.k, scan_df.silhouette, label="silhouette (cosine)")
    ax.plot(scan_df.k, scan_df.nmi_sector, label="NMI vs GICS sector")
    ax.axvline(11, ls=":", c="gray")
    ax.set_xlabel("k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "kmeans_scan.png", dpi=130)
    plt.close(fig)

    # hierarchical
    Z, hl = hierarchical(X, k)
    res["hierarchical_k11"] = eval_clustering(hl, sector, sub, rng)
    sub_idx = []
    for s in sorted(set(sector)):
        sub_idx += list(np.where(sector == s)[0][:6])
    dendrogram_figure(X, [f"{t} ({s[:4]})" for t, s in zip(comp.ticker, sector)], sub_idx, str(FIG / "dendrogram.png"))

    # classification of GICS sector from description
    cv = classification_cv(texts, sector, cfg["seed"])
    res["sector_classification"] = {n: {kk: v for kk, v in s.items() if kk != "confusion"} | {"confusion": s["confusion"]}
                                    for n, s in cv.items()}
    save_table(pd.DataFrame([{"model": n, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"]} for n, s in cv.items()]),
               "sector_classification")
    best = max(cv, key=lambda n: cv[n]["macro_f1"])
    M = np.array(cv[best]["confusion"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(M, cmap="Blues")
    labs = cv[best]["labels"]
    ax.set_xticks(range(len(labs)), [l[:12] for l in labs], rotation=90)
    ax.set_yticks(range(len(labs)), [l[:12] for l in labs])
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6)
    ax.set_title(f"GICS sector confusion ({best}, 5-fold CV)")
    fig.tight_layout()
    fig.savefig(FIG / "sector_confusion.png", dpi=130)
    plt.close(fig)

    # hidden peers
    hp = hidden_peers(comp, vec, X, nn, pur, 20)
    hp.to_json(PROCESSED / "hidden_peers.json", orient="records")
    res["hidden_peers"] = hp.to_dict(orient="records")

    # graph centrality
    A = knn_graph(S, cfg["peer_k"])
    pr = pagerank(A, 0.85, 1e-8)
    hub, auth = hits(A)
    cent = pd.DataFrame({"ticker": comp.ticker, "name": comp.name, "sector": sector, "pagerank": pr,
                         "hub": hub, "authority": auth})
    cent.to_parquet(PROCESSED / "centrality.parquet", index=False)
    res["centrality_top20"] = {c: cent.nlargest(20, c)[["ticker", "name", "sector", c]].to_dict(orient="records")
                               for c in ("pagerank", "hub", "authority")}

    # persist peers + clusters + 2-D SVD projection for the API/UI
    svd = TruncatedSVD(2, random_state=cfg["seed"]).fit_transform(X)
    out = comp[["ticker", "name", "gics_sector", "gics_sub_industry"]].copy()
    out["kmeans_cluster"], out["hier_cluster"] = lab, hl
    out["x"], out["y"], out["peer_purity"] = svd[:, 0], svd[:, 1], pur
    out["peers"] = [list(comp.ticker.iloc[r]) for r in nn]
    out["peer_sims"] = [[float(S[i, j]) for j in r] for i, r in enumerate(nn)]
    out.to_parquet(PROCESSED / "sectors.parquet", index=False)
    for col, fn in (("gics_sector", "proj_gics"), ("kmeans_cluster", "proj_kmeans")):
        fig, ax = plt.subplots(figsize=(7, 6))
        for g, d in out.groupby(col):
            ax.scatter(d.x, d.y, s=10, label=str(g)[:14])
        ax.set_title(f"SVD projection coloured by {col}")
        ax.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / f"{fn}.png", dpi=130)
        plt.close(fig)
    update_results("sectors", res)
    log.info("k=11 k-means vs GICS: %s", res["kmeans_k11"]["sector"])
    log.info("mean peer purity %.3f (random %.3f)", res["mean_peer_purity"], res["random_peer_purity"])


if __name__ == "__main__":
    main()
