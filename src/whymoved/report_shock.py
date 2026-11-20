"""Phase 15: shock-mode / search-benchmark sections for RESULTS.md and REPORT.md (numbers come from results.json)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import REPORTS


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    from .report import md
    return md(df, digits)


def _fmt_ci(r: dict, k: str) -> str:
    v, lo, hi = r.get(k), r.get(k + "_lo"), r.get(k + "_hi")
    if v is None or pd.isna(v):
        return ""
    return f"{v:.3f}" if lo is None or pd.isna(lo) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _g(x, k, default=float("nan")):
    v = x.get(k, default)
    return default if v is None else v


def shock_verdicts(R: dict) -> dict[str, str]:
    """Plain-language H3-H5 verdicts computed from saved numbers (no claims beyond what p-values support)."""
    out = {}
    sa = R.get("shock_analogs")
    if sa:
        t = pd.DataFrame(sa["table"])
        x = t[(t.k == 5) & (t.subset == "non_thin_all_methods")].set_index("method")
        best = max(["num", "text1", "text2", "fused", "fused_t1", "rrf"], key=lambda m: x.loc[m, "p_at_k"])
        b = x.loc[best]
        ok = all(_g(b, f"p_at_k_p_vs_{r}", 1) < 0.05 and _g(b, f"p_at_k_diff_vs_{r}", -1) > 0 for r in ("vol", "rand"))
        out["H3"] = (f"H3: best method by P@5 is `{best}` ({b.p_at_k:.3f}; empirical chance {b.chance:.3f}; n_queries={int(b.n_queries)}, "
                     f"n_episodes={int(b.n_episodes)}). Difference vs M_vol {_g(b, 'p_at_k_diff_vs_vol'):+.3f} (p={_g(b, 'p_at_k_p_vs_vol'):.3f}), "
                     f"vs random {_g(b, 'p_at_k_diff_vs_rand'):+.3f} (p={_g(b, 'p_at_k_p_vs_rand'):.3f}) -> "
                     + ("supported at p<0.05 against both baselines." if ok else
                        "NOT supported (null result): not significantly better than both baselines."))
        h5 = x.loc["text2"]
        out["H5"] = (f"H5: burst (T2) P@5 {h5.p_at_k:.3f} vs plain TF-IDF (T1) {x.loc['text1', 'p_at_k']:.3f}; difference "
                     f"{_g(h5, 'p_at_k_diff_vs_text1'):+.3f} (p={_g(h5, 'p_at_k_p_vs_text1'):.3f}) -> "
                     + ("burst weighting better." if _g(h5, "p_at_k_p_vs_text1", 1) < .05 and _g(h5, "p_at_k_diff_vs_text1", -1) > 0
                        else "NOT supported."))
        out["era"] = (f"Era diagnostics (k=5, non-thin subset): mean |year gap| of analogs: numeric {x.loc['num', 'mean_year_gap']:.1f}y, "
                      f"T1 {x.loc['text1', 'mean_year_gap']:.1f}y, T2 {x.loc['text2', 'mean_year_gap']:.1f}y, era-only baseline "
                      f"{x.loc['era', 'mean_year_gap']:.1f}y; share of analogs within +/-2 years: numeric {x.loc['num', 'share_same_era_pm2y']:.2f}, "
                      f"T1 {x.loc['text1', 'share_same_era_pm2y']:.2f}, T2 {x.loc['text2', 'share_same_era_pm2y']:.2f}. "
                      + ("Text similarity is dominated by vocabulary drift: text analogs come mostly from the query's own era."
                         if x.loc["text1", "share_same_era_pm2y"] > 2 * x.loc["num", "share_same_era_pm2y"]
                         else "Text analogs are not markedly more era-bound than numeric ones."))
    sr = R.get("shock_regimes")
    if sr:
        h = sr["h4"]
        m = h["models"][h["best_text_model"]]
        out["H4"] = (f"H4: best text model `{h['best_text_model']}` macro-F1 {m['macro_f1']:.3f} (95% CI {[round(v, 3) for v in m['f1_ci']]}) vs majority "
                     f"{h['models']['majority']['macro_f1']:.3f} and era-only {h['models']['era_only']['macro_f1']:.3f} "
                     f"(p vs majority {m['p_vs_majority']:.3f}, vs era-only {m['p_vs_era_only']:.3f}; n_queries={h['n_queries']}, "
                     f"n_episodes={h['n_episodes']}) -> " + ("supported." if h["h4_supported"] else "NOT supported."))
    return out


def shock_md(R: dict) -> list[str]:
    """All add-on tables for RESULTS.md."""
    L: list[str] = []
    sd = R.get("shock_detection")
    if sd:
        L += ["## Shock mode: detection", "", f"{sd['n_shock_days']} shock days in {sd['n_episodes']} episodes ({sd['n_dev']} in DEV <= 2014).", "",
              _md(pd.DataFrame({"year": list(sd["per_year"]), "shock_days": list(sd["per_year"].values())})), "",
              "### Face-validity (curated dates; a check only, not ground truth; user to verify against sources)", "",
              _md(pd.DataFrame(sd["face_validity"])), ""]
    cov = REPORTS / "tables" / "headline_coverage.csv"
    if cov.exists():
        L += ["### Headline coverage per year", "", _md(pd.read_csv(cov)), ""]
    sa = R.get("shock_analogs")
    if sa:
        v = shock_verdicts(R)
        tuned = {k: x for k, x in sa["tuned"].items() if k != "fusion_grid"}
        L += ["## Shock mode: analog retrieval (TEST 2015-2023)", "", v.get("H3", ""), "", v.get("H5", ""), "", v.get("era", ""), "",
              f"DEV-tuned: `{tuned}`. Tercile cut points (DEV): {sa['tercile_cuts']}. Thin-text share of shock days: "
              f"{sa['thin_share_of_shock_days']:.3f}.", ""]
        t = pd.DataFrame(sa["table"])
        for subset in t.subset.unique():
            for k in (3, 5, 10):
                x = t[(t.subset == subset) & (t.k == k)]
                rows = []
                for r in x.to_dict(orient="records"):
                    def dp(ref):
                        d, p = r.get(f"p_at_k_diff_vs_{ref}"), r.get(f"p_at_k_p_vs_{ref}")
                        return "" if d is None or pd.isna(d) else f"{d:+.3f} ({p:.2f})"
                    rows.append({"method": r["method"], "n_q": int(r["n_queries"]), "n_ep": int(r["n_episodes"]),
                                 "P@k [95% CI]": _fmt_ci(r, "p_at_k"), "chance": r.get("chance"), "MAE ret20": r["mae_ret"],
                                 "MAE maxdd": r["mae_dd"], "Spearman": r["spearman_ret"], "sign acc": r["sign_acc"],
                                 "d vs vol (p)": dp("vol"), "d vs rand (p)": dp("rand"),
                                 "mean |year gap|": r.get("mean_year_gap"), "same era +/-2y": r.get("share_same_era_pm2y")})
                L += [f"### {subset}, k={k}", "", _md(pd.DataFrame(rows)), ""]
        qp = REPORTS / "tables" / "shock_qualitative.csv"
        if qp.exists():
            L += ["### Qualitative analogs (descriptive; no causal claims)", "", _md(pd.read_csv(qp).head(60)), ""]
        L += ["Top terms for example days:", "", "```", str(sa.get("top_terms_examples")), "```", ""]
    sr = R.get("shock_regimes")
    if sr:
        c = sr["clusters"]
        v = shock_verdicts(R)
        L += ["## Shock mode: regimes, archetypes, H4", "", f"Regime label counts: {sr['regime_label_counts']}. {sr['note']}.", "",
              f"k chosen by DEV silhouette: {c['k']}; stability (mean pairwise ARI over 10 seeds): {c['stability_mean_pairwise_ari']:.3f}.", "",
              _md(pd.DataFrame(c["scan"])), "", _md(pd.DataFrame([{"method": k, **x} for k, x in c["evaluation"].items()])), "",
              _md(pd.DataFrame(c["profiles"]).assign(top_terms=lambda d: d.top_terms.map(", ".join))), "", v.get("H4", ""), "",
              _md(pd.DataFrame([{"model": m, "accuracy": x["accuracy"], "macro_F1": x["macro_f1"],
                                 "F1 CI": str([round(y, 3) for y in x["f1_ci"]]), "p vs majority": x.get("p_vs_majority"),
                                 "p vs era-only": x.get("p_vs_era_only")} for m, x in sr["h4"]["models"].items()])), ""]
        for n in ("pagerank", "hub", "authority"):
            L += [f"### Archetype days by {n} (descriptive: 'resembles many other shock days')", "",
                  _md(pd.DataFrame(sr["archetypes"][n]), 4), ""]
    tb = R.get("search_bench_tolerant")
    if tb:
        L += ["## Search benchmarks", "",
              f"### Tolerant retrieval ({tb['n_queries']} seeded corrupted queries over all S&P 500 names, tickers, aliases)", "",
              _md(pd.DataFrame(tb["table"])), "", f"Wildcards vs regex ground truth: `{tb['wildcards']}`", "", "Failure examples:", "",
              _md(pd.DataFrame(tb["failure_examples"])), ""]
    eb = R.get("search_bench_efficiency")
    if eb:
        L += ["### Efficiency (real FNSPID news sample; brute force = sparse-matrix reference with identical formulas)", "",
              f"{eb['n_docs']} docs, {eb['n_terms']} terms, build {eb['build_seconds']:.1f}s, pickle {eb['index_pickle_mb']:.1f} MB.", "",
              _md(pd.DataFrame([{"scorer": k, **x} for k, x in eb["latency"].items()])), "",
              "### Postings compression (gap + VByte)", "", _md(pd.DataFrame([eb["compression"]])), ""]
    return L
