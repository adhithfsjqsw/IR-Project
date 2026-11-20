"""Market-state features for shock days. Every feature at day t uses only data up to and including the close of t."""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU"]
GROUPS = {
    "price_vol": ["r_spy", "r_spy_5d", "r_spy_20d", "drawdown", "rv20", "vol_ratio"],
    "vix": ["log_vix", "vix_d1", "vix_d5"],
    "sector": [f"rel_{s}" for s in SECTORS] + ["worst_sector", "dispersion"],
    "breadth": ["breadth"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def compute_features(m: pd.DataFrame, warmup: int = 260) -> pd.DataFrame:
    """m: long market frame (date, ticker, close, volume). Returns a date-indexed feature frame (warm-up rows dropped)."""
    close = m.pivot(index="date", columns="ticker", values="close").sort_index()
    vol = m.pivot(index="date", columns="ticker", values="volume").sort_index()
    spy, vix = close["SPY"], close["^VIX"]
    r = spy.pct_change()
    sec = close[SECTORS].pct_change()
    f = pd.DataFrame(index=close.index)
    f["r_spy"] = r
    f["r_spy_5d"] = spy / spy.shift(5) - 1
    f["r_spy_20d"] = spy / spy.shift(20) - 1
    f["drawdown"] = spy / spy.rolling(252, min_periods=252).max() - 1
    f["rv20"] = r.rolling(20).std() * np.sqrt(252)
    f["vol_ratio"] = vol["SPY"] / vol["SPY"].rolling(20).mean()
    f["log_vix"] = np.log(vix)
    f["vix_d1"] = np.log(vix).diff()
    f["vix_d5"] = np.log(vix).diff(5)
    for s in SECTORS:
        f[f"rel_{s}"] = sec[s] - r
    f["worst_sector"] = sec.min(axis=1)
    f["dispersion"] = sec.std(axis=1)
    f["breadth"] = (sec < 0).mean(axis=1)
    f["close"], f["vix"] = spy, vix
    return f.iloc[warmup:]


class Scaler:
    """Standardiser fitted on DEV rows only and then frozen."""

    def fit(self, X: pd.DataFrame) -> "Scaler":
        self.mean_, self.std_ = X.mean(), X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.mean_.index] - self.mean_) / self.std_


def group_weights(w: dict[str, float]) -> np.ndarray:
    """Per-feature weight vector (aligned with FEATURES) from group weights, normalised so groups are comparable
    (each group's total squared weight is w_g^2 regardless of its size)."""
    out = []
    for g, cols in GROUPS.items():
        out += [w.get(g, 1.0) / np.sqrt(len(cols))] * len(cols)
    return np.array(out)
