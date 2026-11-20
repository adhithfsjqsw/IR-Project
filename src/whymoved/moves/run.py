"""Phase 3a: abnormal returns + moves.parquet + sanity checks. `python -m whymoved.moves.run`."""
from __future__ import annotations

import json

import pandas as pd

from ..common import INTERIM, PROCESSED, get_logger, load_config, seed_all
from ..data.docs import trading_days
from ..data.prices import wide_returns
from ..reporting import update_results
from .detect import flag_moves, model_panel

log = get_logger("moves")


def main() -> pd.DataFrame:
    cfg = load_config()
    seed_all(cfg["seed"])
    prices = pd.read_parquet(PROCESSED / "prices.parquet")
    tickers = json.loads((INTERIM / "attribution_tickers.json").read_text())
    rets = wide_returns(prices)
    panel = model_panel(rets, tickers, None, cfg["estimation_window"], cfg["estimation_gap"], cfg["min_obs"],
                        cfg["market_proxy"])
    panel = panel[(panel.date >= cfg["period"]["start"]) & (panel.date <= cfg["period"]["end"])]
    panel.to_parquet(INTERIM / "panel_m0.parquet", index=False)
    moves = flag_moves(panel, cfg["z_threshold"], cfg["min_abs_return"])
    moves.to_parquet(PROCESSED / "moves.parquet", index=False)

    z = panel.z.dropna()
    stats = {"z_mean": float(z.mean()), "z_std": float(z.std()), "n_ticker_days": int(len(z)),
             "n_flagged": int(len(moves)), "flags_per_ticker": moves.groupby("ticker").size().to_dict()}
    log.info("z mean=%.3f std=%.3f (n=%d); flagged=%d", stats["z_mean"], stats["z_std"], len(z), len(moves))
    gp = PROCESSED / "gold.parquet"
    if gp.exists():
        g = pd.read_parquet(gp)
        gold_days = {(r.ticker, r.date) for r in g.itertuples()}
        pd_days = panel.assign(d=panel.date.dt.strftime("%Y-%m-%d"))
        in_gold = pd_days.apply(lambda r: (r.ticker, r.d) in gold_days, axis=1)
        flagged_keys = {(r.ticker, r.date.strftime("%Y-%m-%d")) for r in moves.itertuples()}
        stats["share_8k_days_flagged"] = float(
            pd_days[in_gold].apply(lambda r: (r.ticker, r.d) in flagged_keys, axis=1).mean())
        stats["share_flags_with_8k"] = float(
            moves.apply(lambda r: (r.ticker, r.date.strftime("%Y-%m-%d")) in gold_days, axis=1).mean())
        log.info("8-K days flagged: %.3f; flagged moves with 8-K: %.3f", stats["share_8k_days_flagged"],
                 stats["share_flags_with_8k"])
    update_results("moves_sanity", stats)
    return moves


if __name__ == "__main__":
    main()
