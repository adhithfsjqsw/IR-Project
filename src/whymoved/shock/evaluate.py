"""H3 / H5 evaluation of analog retrieval. `python -m whymoved.shock.evaluate`."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import PROCESSED, get_logger, load_config
from ..reporting import save_table, update_results
from .analogs import ShockLib
from .daydocs import build_day_vectors, top_terms
from .features import GROUPS

log = get_logger("shock.eval")
PRIMARY = ["rand", "recent", "era", "vol", "num", "text1", "text2", "fused", "fused_t1", "rrf"]


def terciles(lib: ShockLib, dev_end: str) -> tuple[float, float]:
    d = lib.sd[(lib.sd.date <= dev_end) & lib.ok_outcome].fwd_ret_20
    return float(d.quantile(1 / 3)), float(d.quantile(2 / 3))


def tercile_of(x: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    return np.digitize(x, cuts)


def run_queries(lib: ShockLib, qs: np.ndarray, method: str, k: int, cuts, text_ok: bool, w=None, wf=0.5) -> pd.DataFrame:
    """Per-query metrics for one method."""
    rows = []
    ter = tercile_of(lib.sd.fwd_ret_20.to_numpy(), cuts)
    for q in qs:
        top = lib.topk(method, q, k, text_ok, w, wf)
        L = lib.library(q, text_ok or method.startswith(("text", "fused", "rrf")))
        if len(top) < min(k, 3):
            continue
        a = lib.sd.iloc[top]
        rows.append({"q": q, "episode": lib.ep[q], "p_at_k": float((ter[top] == ter[q]).mean()),
                     "chance": float((ter[L] == ter[q]).mean()), "pred_ret": float(a.fwd_ret_20.mean()),
                     "pred_dd": float(a.fwd_maxdd_20.mean()), "true_ret": float(lib.sd.fwd_ret_20.iloc[q]),
                     "true_dd": float(lib.sd.fwd_maxdd_20.iloc[q]),
                     "year_gap": float(np.abs(lib.year[top] - lib.year[q]).mean()),
                     "same_era": float((np.abs(lib.year[top] - lib.year[q]) <= 2).mean())})
    return pd.DataFrame(rows)


def baseline_forecast(lib: ShockLib, qs, kind: str, text_ok: bool) -> pd.DataFrame:
    rows = []
    for q in qs:
        L = lib.library(q, text_ok)
        o = lib.sd.iloc[L]
        f = np.mean if kind == "lib_mean" else np.median
        rows.append({"q": q, "episode": lib.ep[q], "pred_ret": f(o.fwd_ret_20), "pred_dd": f(o.fwd_maxdd_20),
                     "true_ret": lib.sd.fwd_ret_20.iloc[q], "true_dd": lib.sd.fwd_maxdd_20.iloc[q]})
    return pd.DataFrame(rows)


def stats(d: pd.DataFrame) -> dict[str, float]:
    s = {"mae_ret": float(np.abs(d.pred_ret - d.true_ret).mean()), "mae_dd": float(np.abs(d.pred_dd - d.true_dd).mean()),
         "sign_acc": float((np.sign(d.pred_ret) == np.sign(d.true_ret)).mean()),
         "spearman_ret": float(spearmanr(d.pred_ret, d.true_ret).correlation) if len(d) > 2 and d.pred_ret.nunique() > 1 else np.nan}
    if "p_at_k" in d:
        s["p_at_k"], s["chance"] = float(d.p_at_k.mean()), float(d.chance.mean())
    return s


METRICS = ["p_at_k", "mae_ret", "mae_dd", "sign_acc", "spearman_ret"]


def _metrics_arr(d: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    pr, tr = d["pred_ret"][idx], d["true_ret"][idx]
    out = {"mae_ret": float(np.abs(pr - tr).mean()), "mae_dd": float(np.abs(d["pred_dd"][idx] - d["true_dd"][idx]).mean()),
           "sign_acc": float((np.sign(pr) == np.sign(tr)).mean())}
    out["spearman_ret"] = float(spearmanr(pr, tr).correlation) if len(idx) > 2 and np.ptp(pr) > 0 else np.nan
    if "p_at_k" in d:
        out["p_at_k"] = float(d["p_at_k"][idx].mean())
    return out


def block_boot_all(frames: dict[str, pd.DataFrame], n: int = 1000, seed: int = 42, refs=("vol", "rand", "text1")):
    """Episode-block bootstrap (blocks = query episodes), shared resamples across methods and metrics.

    Returns (ci[method][metric] = (lo, hi), paired[(method, ref, metric)] = (mean diff, p)).
    """
    qs = sorted(set.intersection(*[set(f.q) for f in frames.values()]))
    arr = {}
    for m, f in frames.items():
        f = f.set_index("q").loc[qs]
        arr[m] = {c: f[c].to_numpy(float) for c in ("pred_ret", "pred_dd", "true_ret", "true_dd", "p_at_k") if c in f}
    ep = frames[next(iter(frames))].set_index("q").loc[qs].episode.to_numpy()
    groups = {e: np.where(ep == e)[0] for e in np.unique(ep)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    vals = {m: {k: np.full(n, np.nan) for k in METRICS} for m in frames}
    for b in range(n):
        idx = np.concatenate([groups[e] for e in rng.choice(keys, size=len(keys))])
        for m in frames:
            for k, v in _metrics_arr(arr[m], idx).items():
                vals[m][k][b] = v
    ci = {m: {k: (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))) if np.isfinite(v).any() else (np.nan, np.nan)
              for k, v in d.items()} for m, d in vals.items()}
    paired = {}
    for m in frames:
        for r in refs:
            if r in frames and m != r:
                for k in METRICS:
                    d = vals[m][k] - vals[r][k]
                    d = d[np.isfinite(d)]
                    if len(d):
                        paired[(m, r, k)] = (float(d.mean()), float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / n))))
    point = {m: _metrics_arr(arr[m], np.arange(len(qs))) for m in frames}
    return ci, paired, point, len(qs)


def tune_dev(lib: ShockLib, cfg: dict, cuts) -> dict:
    """Group weights for M_num and the fusion weight w, on DEV (expanding-window queries, same library rule)."""
    dev_q = np.where((lib.sd.date <= cfg["dev_end"]).to_numpy() & lib.ok_outcome)[0]
    dev_q = np.array([q for q in dev_q if len(lib.library(q)) >= 10])
    grid = [0, 0.5, 1, 2]
    best = (-1, None)
    for ws in itertools.product(grid, repeat=len(GROUPS)):
        if not any(ws):
            continue
        w = dict(zip(GROUPS, ws))
        p = run_queries(lib, dev_q, "num", cfg["k_analogs"], cuts, False, w).p_at_k.mean()
        if p > best[0] + 1e-9:
            best = (p, w)
    w = best[1]
    dev_t = np.array([q for q in dev_q if not lib.thin[q] and len(lib.library(q, True)) >= 10])
    wf_scores = {wf: run_queries(lib, dev_t, "fused", cfg["k_analogs"], cuts, True, w, wf).p_at_k.mean()
                 for wf in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]} if len(dev_t) else {0.5: np.nan}
    wf = max(wf_scores, key=lambda x: (np.nan_to_num(wf_scores[x], nan=-1), -abs(x - 0.5)))
    return {"group_weights": w, "dev_p_at_5_num": float(best[0]), "fusion_w": wf, "fusion_grid": wf_scores,
            "n_dev_queries": int(len(dev_q)), "n_dev_text_queries": int(len(dev_t))}


def main() -> dict:
    cfg = load_config()["shock"]
    sd = pd.read_parquet(PROCESSED / "shock_days.parquet")
    dv = build_day_vectors()
    lib = ShockLib(sd, dv, cfg["horizon"], load_config()["seed"])
    cuts = terciles(lib, cfg["dev_end"])
    tuned = tune_dev(lib, cfg, cuts)
    log.info("tuned: %s", {k: v for k, v in tuned.items() if k != "fusion_grid"})
    w, wf = tuned["group_weights"], tuned["fusion_w"]
    test_q = np.where((lib.sd.date >= cfg["test_start"]).to_numpy() & lib.ok_outcome)[0]
    results = {"tuned": tuned, "tercile_cuts": cuts, "subsets": {}}
    tables = []
    for subset, text_ok in (("all_days_numeric_methods", False), ("non_thin_all_methods", True)):
        qs = np.array([q for q in test_q if (not text_ok or not lib.thin[q]) and len(lib.library(q, text_ok)) >= cfg["k_analogs"]])
        methods = PRIMARY if text_ok else ["rand", "recent", "era", "vol", "num"]
        for k in (3, 5, 10):
            fr = {m: run_queries(lib, qs, m, k, cuts, text_ok, w, wf) for m in methods}
            for m in methods:
                fr[m] = fr[m][fr[m].q.isin(qs)]
            fr["lib_mean"], fr["lib_median"] = baseline_forecast(lib, qs, "lib_mean", text_ok), baseline_forecast(lib, qs, "lib_median", text_ok)
            n_q, n_e = len(qs), int(lib.sd.episode_id.iloc[qs].nunique())
            ci, paired, point, n_used = block_boot_all(fr, 1000, 42)
            for m, f in fr.items():
                row = {"subset": subset, "k": k, "method": m, "n_queries": n_used, "n_episodes": n_e, **point[m]}
                if "chance" in f:
                    row["chance"] = float(f.chance.mean())
                for metric in ("p_at_k", "mae_ret", "spearman_ret"):
                    lo, hi = ci[m][metric]
                    row[metric + "_lo"], row[metric + "_hi"] = lo, hi
                for ref in ("vol", "rand", "text1"):
                    for metric in ("p_at_k", "mae_ret", "spearman_ret"):
                        pr = paired.get((m, ref, metric))
                        if pr:
                            row[f"{metric}_diff_vs_{ref}"], row[f"{metric}_p_vs_{ref}"] = pr
                if "year_gap" in f:
                    row["mean_year_gap"], row["share_same_era_pm2y"] = float(f.year_gap.mean()), float(f.same_era.mean())
                tables.append(row)
            if k == cfg["k_analogs"]:
                results["subsets"][subset] = {"n_queries": n_q, "n_episodes": n_e}
    t = pd.DataFrame(tables)
    save_table(t, "shock_analogs")
    results["table"] = t.to_dict(orient="records")
    # H5: burst vs plain, k = 5, non-thin subset
    h5 = t[(t.subset == "non_thin_all_methods") & (t.k == cfg["k_analogs"]) & t.method.isin(["text1", "text2"])]
    results["h5"] = h5[["method", "p_at_k", "p_at_k_lo", "p_at_k_hi", "mae_ret", "spearman_ret", "p_at_k_diff_vs_text1",
                        "p_at_k_p_vs_text1", "n_queries", "n_episodes"]].to_dict(orient="records")
    # qualitative table for curated days that are eligible queries
    qual = []
    cur = pd.read_csv(PROCESSED.parent / "curated" / "episodes.csv")
    X1, X2 = dv.t1([d.strftime("%Y-%m-%d") for d in lib.sd.date]), dv.t2([d.strftime("%Y-%m-%d") for d in lib.sd.date])
    for d in cur.date:
        idx = np.where(lib.sd.date == pd.Timestamp(d))[0]
        if len(idx) == 0 or len(lib.library(idx[0])) < 5:
            continue
        q = idx[0]
        for m in ("num", "text2", "fused"):
            top = lib.topk(m, q, 5, m != "num", w, wf)
            for r, a in enumerate(top, 1):
                row = lib.sd.iloc[a]
                qual.append({"query": d, "method": m, "rank": r, "analog_date": row.date.strftime("%Y-%m-%d"),
                             "spy_ret": row.r_spy, "fwd_ret_20": row.fwd_ret_20, "fwd_maxdd_20": row.fwd_maxdd_20})
    qdf = pd.DataFrame(qual)
    save_table(qdf, "shock_qualitative")
    results["qualitative_n_queries"] = int(qdf["query"].nunique()) if len(qdf) else 0
    examples = {}
    for d in cur.date:
        hit = np.where(lib.sd.date == pd.Timestamp(d))[0]
        if len(hit) and not lib.thin[hit[0]] and len(examples) < 5:
            examples[d] = {"T1": top_terms(X1, dv, int(hit[0])), "T2": top_terms(X2, dv, int(hit[0]))}
    results["top_terms_examples"] = examples
    results["thin_share_of_shock_days"] = float(lib.thin.mean())
    update_results("shock_analogs", results)
    show = t[(t.k == cfg["k_analogs"])][["subset", "method", "n_queries", "n_episodes", "p_at_k", "chance", "mae_ret",
                                         "spearman_ret", "mean_year_gap"]]
    log.info("H3 (k=5):\n%s", show.round(3).to_string())
    return results


if __name__ == "__main__":
    main()
