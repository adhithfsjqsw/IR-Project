"""H2: market vs official-sector vs text-peer benchmarks. `python -m whymoved.moves.benchmarks`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from ..common import INTERIM, PROCESSED, get_logger, load_config, seed_all
from ..data.prices import wide_returns
from ..reporting import save_table, update_results
from .detect import model_panel

log = get_logger("h2")


def sector_mean_excl(rets: pd.DataFrame, comp: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Equal-weight mean return of the other members of each ticker's GICS sector (NaN-aware)."""
    sec = comp.set_index("ticker").gics_sector
    out = {}
    R = rets.drop(columns=["SPY"], errors="ignore")
    for t in tickers:
        mem = [m for m in sec[sec == sec[t]].index if m in R.columns and m != t]
        out[t] = R[mem].mean(axis=1, skipna=True)
    return pd.DataFrame(out)


def text_peer_mean(rets: pd.DataFrame, sectors: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Equal-weight mean return of each ticker's top-k text neighbours (self excluded by construction)."""
    pe = sectors.set_index("ticker").peers
    return pd.DataFrame({t: rets[[p for p in pe[t] if p in rets.columns]].mean(axis=1, skipna=True) for t in tickers})


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main() -> dict:
    cfg = load_config()
    seed_all(cfg["seed"])
    prices = pd.read_parquet(PROCESSED / "prices.parquet")
    rets = wide_returns(prices)
    comp = pd.read_parquet(PROCESSED / "companies.parquet")
    sectors = pd.read_parquet(PROCESSED / "sectors.parquet")
    tickers = json.loads((INTERIM / "attribution_tickers.json").read_text())
    gold = pd.read_parquet(PROCESSED / "gold.parquet")
    w, g, mo = cfg["estimation_window"], cfg["estimation_gap"], cfg["min_obs"]
    sec_f, txt_f = sector_mean_excl(rets, comp, tickers), text_peer_mean(rets, sectors, tickers)
    models = {"M0 market": None, "M1 market+GICS sector": {"sector": sec_f}, "M2 market+text peers": {"peers": txt_f}}
    panels = {}
    for name, f in models.items():
        p = model_panel(rets, tickers, f, w, g, mo, cfg["market_proxy"])
        p = p[(p.date >= cfg["test"]["start"]) & (p.date <= cfg["test"]["end"])]
        panels[name] = p.set_index(["ticker", "date"]).z
    D = pd.concat(panels, axis=1).dropna()  # identical sample for every model
    gk = {(r.ticker, r.date) for r in gold.itertuples()}
    D["y"] = [int((t, d.strftime("%Y-%m-%d")) in gk) for t, d in D.index]
    y = D.y.to_numpy()
    log.info("H2 sample: %d ticker-days, %d with an effective 8-K (%.2f%%)", len(D), y.sum(), 100 * y.mean())
    S = {n: D[n].abs().to_numpy() for n in models}
    rng = np.random.default_rng(cfg["seed"])
    B = 1000
    boots = rng.integers(0, len(D), size=(B, len(D)))
    res = {"n_days": int(len(D)), "n_positive": int(y.sum()), "base_rate": float(y.mean()), "models": {}}
    auc_b = {n: np.empty(B) for n in S}
    ap_b = {n: np.empty(B) for n in S}
    for b in range(B):
        i = boots[b]
        for n, s in S.items():
            auc_b[n][b] = fast_auc(y[i], s[i])
            ap_b[n][b] = average_precision_score(y[i], s[i]) if y[i].sum() else np.nan
    rows = []
    for n, s in S.items():
        top = np.argsort(-s)[:200]
        row = {"model": n, "roc_auc": fast_auc(y, s), "roc_auc_lo": float(np.nanquantile(auc_b[n], .025)),
               "roc_auc_hi": float(np.nanquantile(auc_b[n], .975)), "pr_auc": float(average_precision_score(y, s)),
               "pr_auc_lo": float(np.nanquantile(ap_b[n], .025)), "pr_auc_hi": float(np.nanquantile(ap_b[n], .975)),
               "top200_share_with_8k": float(y[top].mean()), "z_std": float(D[n].std())}
        rows.append(row)
    res["models"] = rows
    names = list(S)
    res["paired"] = {}
    for a, b_ in ((names[2], names[0]), (names[2], names[1]), (names[1], names[0])):
        for metric, arr in (("roc_auc", auc_b), ("pr_auc", ap_b)):
            d = arr[a] - arr[b_]
            p = 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))
            res["paired"][f"{a} - {b_} [{metric}]"] = {"mean_diff": float(np.nanmean(d)), "p": float(min(1.0, max(p, 1 / B)))}
    tab = pd.DataFrame(rows)
    save_table(tab, "h2_models")
    update_results("h2", res)
    log.info("H2:\n%s", tab.round(4).to_string())
    return res


if __name__ == "__main__":
    main()
