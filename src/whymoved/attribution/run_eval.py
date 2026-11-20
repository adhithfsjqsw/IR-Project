"""H1 evaluation: tune on DEV, report on TEST. `python -m whymoved.attribution.run_eval`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common import INTERIM, PROCESSED, get_logger, load_config, seed_all
from ..data.docs import trading_days
from ..ir.build import load_or_build
from ..reporting import save_table, update_results
from .ablations import ablation_table, systems_from_params, tune_on_dev
from .candidates import price_move_docs
from .evaluate import Evaluator, System
from .gold import gold_lookup

log = get_logger("run_eval")


def load_prior():
    """Source-reputation prior for A6 if reputation was built (reports/tables/publisher_reputation.csv)."""
    from ..reputation.sources import make_prior
    return make_prior()


def main() -> None:
    cfg = load_config()
    seed_all(cfg["seed"])
    idx = load_or_build()
    comp = pd.read_parquet(PROCESSED / "companies.parquet")
    prices = pd.read_parquet(PROCESSED / "prices.parquet")
    cal = trading_days(prices, cfg["market_proxy"])
    gold = gold_lookup(pd.read_parquet(PROCESSED / "gold.parquet"))
    moves = pd.read_parquet(PROCESSED / "moves.parquet")
    ev = Evaluator(idx, dict(zip(comp.ticker, comp.name)), gold, cal, cfg["bm25"]["k1"], cfg["bm25"]["b"],
                   cfg["seed"], price_move_docs(idx))
    d0, d1 = cfg["dev"]["start"], cfg["dev"]["end"]
    t0, t1 = cfg["test"]["start"], cfg["test"]["end"]
    dev = moves[(moves.date >= d0) & (moves.date <= d1)]
    test = moves[(moves.date >= t0) & (moves.date <= t1)]
    has_gold = lambda df: np.array([(r.ticker, r.date.strftime("%Y-%m-%d")) in gold for r in df.itertuples()])
    counts = {"dev_flagged": len(dev), "dev_with_gold": int(has_gold(dev).sum()),
              "test_flagged": len(test), "test_with_gold": int(has_gold(test).sum()),
              "test_without_gold": int((~has_gold(test)).sum())}
    log.info("counts: %s", counts)
    if counts["dev_with_gold"] < 5 or counts["test_with_gold"] < 5:
        log.warning("Too few gold-covered moves for a meaningful evaluation: %s", counts)

    tuned = tune_on_dev(ev, dev, cfg)
    log.info("tuned on DEV: %s", {k: v for k, v in tuned.items() if k != "grids"})
    prior = load_prior()
    systems = systems_from_params(tuned, prior)
    out = {"counts": counts, "tuned": tuned}
    per_hard = {}
    for mode in ("hard", "easy"):
        tab, per = ablation_table(ev, test, systems, mode, 1000)
        save_table(tab, f"ablation_{mode}")
        out[f"ablation_{mode}"] = tab.to_dict(orient="records")
        if mode == "hard":
            per_hard = per
        log.info("ablation (%s):\n%s", mode, tab[["system", "n_moves", "rr", "recall@1", "recall@10", "ndcg10"]]
                 .round(3).to_string())

    # optional: drop price-move-report headlines from the pool (final system, both modes)
    final = systems[-1]
    nopm = System(final.name + " (no price-report headlines)", method=final.method, direction_w=final.direction_w,
                  prf=final.prf, lam=final.lam, prior=final.prior, exclude_price_reports=True)
    out["price_report_exclusion"] = {m: {"with": ev.run(test, final, m).rr.mean(), "without": ev.run(test, nopm, m).rr.mean()}
                                     for m in ("hard", "easy")}
    out["n_price_report_docs"] = len(ev.pm)

    # "unexplained" heuristic: tau from DEV (90% of gold-covered dev moves have top-1 >= tau), final system, easy mode
    dev_scores = ev.run(dev, final, "easy").top1.to_numpy()
    tau = float(np.quantile(dev_scores, 0.10)) if len(dev_scores) else float("nan")
    t_all = ev.run(test, final, "easy", require_gold=False)
    t_all["has_gold"] = t_all.n_gold > 0
    out["unexplained"] = {"tau": tau, "heuristic": True,
                          "share_goldless_below_tau": float((t_all[~t_all.has_gold].top1 < tau).mean())
                          if (~t_all.has_gold).any() else None,
                          "share_gold_below_tau": float((t_all[t_all.has_gold].top1 < tau).mean())
                          if t_all.has_gold.any() else None,
                          "n_goldless": int((~t_all.has_gold).sum())}
    update_results("h1", out)
    per_hard[final.name].to_parquet(INTERIM / "h1_final_per_move.parquet")
    log.info("H1 done: %s", json.dumps(out["unexplained"]))


if __name__ == "__main__":
    main()
