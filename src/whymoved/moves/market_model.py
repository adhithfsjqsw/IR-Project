"""Rolling OLS abnormal-return model (market model and its multi-factor variants)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_abnormal(y: np.ndarray, X: np.ndarray, window: int = 120, gap: int = 11,
                     min_obs: int = 100) -> pd.DataFrame:
    """For each day i, fit y = a + X b on the `window` days ending `gap` days before i (inclusive of i-gap).

    abnormal_i = y_i - (a + X_i b); sigma = residual std in the window (ddof = n_params);
    z = abnormal / sigma. Days with < min_obs valid window rows (or NaN y_i / X_i) get NaN.
    Returns columns: alpha, beta0.., sigma, expected, abnormal, z, n_obs.
    """
    T, k = X.shape
    cols = ["alpha"] + [f"beta{j}" for j in range(k)] + ["sigma", "expected", "abnormal", "z", "n_obs"]
    out = np.full((T, len(cols)), np.nan)
    for i in range(T):
        hi = i - gap + 1
        lo = hi - window
        if lo < 0 or np.isnan(y[i]) or np.isnan(X[i]).any():
            continue
        yw, Xw = y[lo:hi], X[lo:hi]
        ok = ~(np.isnan(yw) | np.isnan(Xw).any(axis=1))
        n = int(ok.sum())
        if n < min_obs:
            continue
        A = np.column_stack([np.ones(n), Xw[ok]])
        coef, *_ = np.linalg.lstsq(A, yw[ok], rcond=None)
        resid = yw[ok] - A @ coef
        sigma = float(np.sqrt(resid @ resid / max(1, n - A.shape[1])))
        exp = float(coef[0] + X[i] @ coef[1:])
        ab = float(y[i] - exp)
        out[i] = [*coef, sigma, exp, ab, ab / sigma if sigma > 0 else np.nan, n]
    return pd.DataFrame(out, columns=cols)
