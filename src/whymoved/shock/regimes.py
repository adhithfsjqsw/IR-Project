"""Phase 12: regime clustering, archetype days (PageRank/HITS) and H4 text->regime classification."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from ..common import PROCESSED, REPORTS, get_logger, load_config
from ..graphalgo import hits, pagerank
from ..reporting import save_table, update_results
from ..sectors.classify import KNNClassifier, MultinomialNB, RocchioClassifier, scores
from ..sectors.graph import knn_graph
from ..sectors.kmeans import spherical_kmeans
from .daydocs import DayVectors, build_day_vectors
from .features import FEATURES, GROUPS, group_weights

log = get_logger("shock.regimes")
REGIMES = ["low_vix_down", "low_vix_up", "mid_vix_down", "mid_vix_up", "high_vix_down", "high_vix_up"]


def unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def fused_matrix(sd: pd.DataFrame, V: sp.csr_matrix, thin: np.ndarray, w: dict, wf: float = 0.5) -> np.ndarray:
    """[sqrt(wf) * unit(weighted numeric) | sqrt(1-wf) * unit(burst text)]; thin days have a zero text part."""
    num = unit(sd[[f + "_std" for f in FEATURES]].to_numpy() * group_weights(w))
    txt = V.toarray()
    txt[thin] = 0.0
    return np.hstack([np.sqrt(wf) * num, np.sqrt(1 - wf) * txt])


def regime_labels(sd: pd.DataFrame, dev_end: str) -> np.ndarray:
    """VIX-level tercile (cuts on DEV shock days) x SPY direction."""
    dev = sd[sd.date <= dev_end].log_vix
    c1, c2 = dev.quantile(1 / 3), dev.quantile(2 / 3)
    lvl = np.digitize(sd.log_vix, [c1, c2])
    return np.array([f"{['low', 'mid', 'high'][l]}_vix_{'up' if d > 0 else 'down'}" for l, d in zip(lvl, sd.direction)])


def cluster_regimes(sd, F, dv, V, cfg, labels) -> dict:
    dev = (sd.date <= cfg["dev_end"]).to_numpy()
    scan = []
    for k in range(3, 13):
        lab, _, obj = spherical_kmeans(sp.csr_matrix(F), k, 5, seed=42)
        sil = float(silhouette_score(F[dev], lab[dev], metric="cosine")) if len(set(lab[dev])) > 1 else np.nan
        scan.append({"k": k, "silhouette_dev": sil, "objective": obj,
                     "nmi_regime": float(normalized_mutual_info_score(labels, lab))})
    k = int(max(scan, key=lambda r: np.nan_to_num(r["silhouette_dev"], nan=-1))["k"])
    lab, C, _ = spherical_kmeans(sp.csr_matrix(F), k, 10, seed=42)
    runs = [spherical_kmeans(sp.csr_matrix(F), k, 3, seed=s)[0] for s in range(10)]
    aris = [adjusted_rand_score(runs[i], runs[j]) for i in range(10) for j in range(i + 1, 10)]
    inv = {j: t for t, j in dv.vocab.items()}
    prof = []
    nnum = len(FEATURES)
    for c in range(k):
        m = lab == c
        tc = np.asarray(V[m].mean(axis=0)).ravel()
        rep = np.argsort(-(unit(F[m]) @ unit(C[[c]]).T).ravel())[:3]
        d = sd[m]
        prof.append({"cluster": c, "n": int(m.sum()), "top_terms": [inv[j] for j in np.argsort(-tc)[:8] if tc[j] > 0],
                     "median_vix": float(np.exp(d.log_vix.median())), "median_r_spy": float(d.r_spy.median()),
                     "median_drawdown": float(d.drawdown.median()), "median_fwd_ret_20": float(d.fwd_ret_20.median()),
                     "median_fwd_maxdd_20": float(d.fwd_maxdd_20.median()),
                     "representative_dates": d.date.iloc[rep].dt.strftime("%Y-%m-%d").tolist()})
    Z = linkage(pdist(F, "cosine").clip(0), "average")
    from scipy.cluster.hierarchy import fcluster
    hl = fcluster(Z, k, "maxclust") - 1
    fig, ax = plt.subplots(figsize=(8, 14))
    dendrogram(Z, labels=sd.date.dt.strftime("%Y-%m-%d").tolist(), orientation="left", leaf_font_size=5, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "shock_dendrogram.png", dpi=110)
    plt.close(fig)
    ev = {n: {"nmi_regime": float(normalized_mutual_info_score(labels, l)), "ari_regime": float(adjusted_rand_score(labels, l)),
              "nmi_episode": float(normalized_mutual_info_score(sd.episode_id, l)),
              "ari_episode": float(adjusted_rand_score(sd.episode_id, l))} for n, l in (("kmeans", lab), ("hierarchical", hl))}
    save_table(pd.DataFrame(scan), "shock_k_scan")
    save_table(pd.DataFrame(prof).assign(top_terms=lambda d: d.top_terms.map(", ".join)), "shock_cluster_profiles")
    return {"k": k, "scan": scan, "stability_mean_pairwise_ari": float(np.mean(aris)), "evaluation": ev, "profiles": prof,
            "labels": lab.tolist()}


def archetypes(sd, F) -> dict:
    Fu = unit(F)
    S = np.clip(Fu @ Fu.T, 0, None)  # PageRank needs non-negative edge weights
    A = knn_graph(S, 10)
    pr = pagerank(A, 0.85, 1e-8)
    hub, auth = hits(A)
    out = {}
    for name, v in (("pagerank", pr), ("hub", hub), ("authority", auth)):
        top = np.argsort(-v)[:15]
        out[name] = [{"date": sd.date.iloc[i].strftime("%Y-%m-%d"), "score": float(v[i]), "r_spy": float(sd.r_spy.iloc[i]),
                      "vix": float(np.exp(sd.log_vix.iloc[i])), "drawdown": float(sd.drawdown.iloc[i]),
                      "fwd_ret_20": float(sd.fwd_ret_20.iloc[i]) if pd.notna(sd.fwd_ret_20.iloc[i]) else None} for i in top]
        save_table(pd.DataFrame(out[name]), f"shock_archetypes_{name}")
    return out


def h4(sd: pd.DataFrame, dv: DayVectors, labels: np.ndarray, cfg: dict) -> dict:
    """Expanding-window: for each TEST year, train on earlier shock days and test on that year's shock days."""
    days = sd.date.dt.strftime("%Y-%m-%d").tolist()
    idx = [dv.pos[d] for d in days]
    C = dv.counts[idx]
    T1, T2 = dv.t1(days), dv.t2(days)
    year = sd.date.dt.year.to_numpy()
    ok = ~dv.thin[idx]
    Xnum = sd[[f + "_std" for f in FEATURES]].to_numpy()
    test_years = sorted(set(year[(sd.date >= cfg["test_start"]).to_numpy()]))
    preds = {m: {} for m in ["majority", "era_only", "nb_counts", "rocchio_t1", "rocchio_t2", "knn_t1", "knn_t2", "oracle_numeric_knn"]}
    for Y in test_years:
        tr = np.where((year < Y) & ok)[0]
        te = np.where((year == Y) & ok)[0]
        if len(tr) < 10 or len(te) == 0 or len(set(labels[tr])) < 2:
            continue
        ytr = labels[tr]
        maj = pd.Series(ytr).mode()[0]
        recent = tr[year[tr] >= year[tr].max() - 2]
        era = pd.Series(labels[recent]).mode()[0]
        fits = {"nb_counts": (MultinomialNB(1.0).fit(C[tr], ytr), C[te]),
                "rocchio_t1": (RocchioClassifier().fit(T1[tr], ytr), T1[te]),
                "rocchio_t2": (RocchioClassifier().fit(T2[tr], ytr), T2[te]),
                "knn_t1": (KNNClassifier(5).fit(T1[tr], ytr), T1[te]),
                "knn_t2": (KNNClassifier(5).fit(T2[tr], ytr), T2[te]),
                "oracle_numeric_knn": (KNNClassifier(5).fit(sp.csr_matrix(Xnum[tr]), ytr), sp.csr_matrix(Xnum[te]))}
        for i in te:
            preds["majority"][i], preds["era_only"][i] = maj, era
        for name, (clf, X) in fits.items():
            for i, p in zip(te, clf.predict(X)):
                preds[name][i] = p
    res, rng = {}, np.random.default_rng(42)
    qs = sorted(preds["majority"])
    ep = sd.episode_id.to_numpy()
    eps = np.unique(ep[qs])
    groups = {e: [i for i in qs if ep[i] == e] for e in eps}
    draws = [np.concatenate([groups[e] for e in rng.choice(eps, len(eps))]) for _ in range(1000)]
    y = labels
    labs = sorted(set(y[qs]))
    boot = {}
    for m, p in preds.items():
        pm = np.array([p[i] for i in qs])
        s = scores(y[qs], pm, labs)
        accs = np.array([np.mean([p[i] == y[i] for i in d]) for d in draws])
        f1s = np.array([scores(y[d], np.array([p[i] for i in d]), labs)["macro_f1"] for d in draws])
        boot[m] = (accs, f1s)
        res[m] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "acc_ci": [float(np.quantile(accs, .025)), float(np.quantile(accs, .975))],
                  "f1_ci": [float(np.quantile(f1s, .025)), float(np.quantile(f1s, .975))], "confusion": s["confusion"], "labels": labs}
    for m in ("nb_counts", "rocchio_t1", "rocchio_t2", "knn_t1", "knn_t2"):
        for ref in ("majority", "era_only"):
            d = boot[m][1] - boot[ref][1]
            res[m][f"f1_diff_vs_{ref}"] = float(d.mean())
            res[m][f"p_vs_{ref}"] = float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / 1000)))
    best = max(("nb_counts", "rocchio_t1", "rocchio_t2", "knn_t1", "knn_t2"), key=lambda m: res[m]["macro_f1"])
    holds = res[best]["f1_diff_vs_majority"] > 0 and res[best]["p_vs_majority"] < .05 and res[best]["f1_diff_vs_era_only"] > 0 and res[best]["p_vs_era_only"] < .05
    save_table(pd.DataFrame([{"model": m, **{k: v for k, v in r.items() if k not in ("confusion", "labels")}} for m, r in res.items()]), "shock_h4")
    return {"n_queries": len(qs), "n_episodes": int(len(eps)), "n_classes_in_test": len(labs), "models": res, "best_text_model": best,
            "h4_supported": bool(holds)}


def main() -> None:
    cfg = load_config()["shock"]
    sd = pd.read_parquet(PROCESSED / "shock_days.parquet").reset_index(drop=True)
    dv = build_day_vectors()
    days = sd.date.dt.strftime("%Y-%m-%d").tolist()
    V = dv.t2(days)
    thin = dv.thin[[dv.pos[d] for d in days]]
    tuned = (load_results().get("shock_analogs", {}).get("tuned", {}).get("group_weights")) or {g: 1.0 for g in GROUPS}
    F = fused_matrix(sd, V, thin, tuned, 0.5)
    labels = regime_labels(sd, cfg["dev_end"])
    log.info("regime label counts: %s", pd.Series(labels).value_counts().to_dict())
    out = {"regime_label_counts": pd.Series(labels).value_counts().to_dict(), "fused_text_weight": 0.5,
           "note": "fused text weight fixed at 0.5 for clustering (the DEV-tuned analog fusion weight selected numeric only)"}
    out["clusters"] = cluster_regimes(sd, F, dv, V, cfg, labels)
    out["archetypes"] = archetypes(sd, F)
    out["h4"] = h4(sd, dv, labels, cfg)
    update_results("shock_regimes", out)
    log.info("k=%d stability ARI %.2f; H4 best=%s supported=%s", out["clusters"]["k"], out["clusters"]["stability_mean_pairwise_ari"],
             out["h4"]["best_text_model"], out["h4"]["h4_supported"])


def load_results():
    from ..reporting import read_results
    return read_results()


if __name__ == "__main__":
    main()
