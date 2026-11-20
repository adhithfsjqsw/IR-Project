"""Abnormal-move detection with the single-factor market model (SPY)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.prices import wide_returns
from .market_model import rolling_abnormal


def model_panel(rets: pd.DataFrame, tickers: list[str], factors: dict[str, pd.DataFrame] | None,
                window: int, gap: int, min_obs: int, market: str = "SPY") -> pd.DataFrame:
    """Rolling abnormal returns for each ticker. `factors[name]` is a date x ticker frame of extra regressors.

    Returns long frame: ticker, date, ret, abnormal, z, + model coefficients.
    """
    parts = []
    mkt = rets[market].to_numpy()
    for t in tickers:
        if t not in rets:
            continue
        cols = [mkt]
        for f in (factors or {}).values():
            cols.append(f[t].reindex(rets.index).to_numpy())
        r = rolling_abnormal(rets[t].to_numpy(), np.column_stack(cols), window, gap, min_obs)
        r.insert(0, "date", rets.index)
        r.insert(0, "ticker", t)
        r["ret"] = rets[t].to_numpy()
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def flag_moves(panel: pd.DataFrame, z_thr: float = 3.0, min_abs: float = 0.02) -> pd.DataFrame:
    """Flag |z| >= z_thr and |ret| >= min_abs. Adds direction (+1/-1)."""
    m = panel[(panel.z.abs() >= z_thr) & (panel.ret.abs() >= min_abs)].copy()
    m["direction"] = np.sign(m.ret).astype(int)
    m["model"] = "market_spy"
    return m[["ticker", "date", "ret", "abnormal", "z", "direction", "model"]].reset_index(drop=True)
