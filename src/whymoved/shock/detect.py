"""Shock-day detection, episodes, forward outcomes, diagnostics. `python -m whymoved.shock.detect`."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import PROCESSED, REPORTS, get_logger, load_config
from ..reporting import save_table, update_results
from .data import load_market
from .features import FEATURES, Scaler, compute_features

log = get_logger("shock.detect")

CURATED = ["2000-04-14", "2001-09-17", "2008-09-29", "2008-10-15", "2010-05-06", "2011-08-08", "2015-08-24",
           "2018-02-05", "2018-12-24", "2020-03-09", "2020-03-12", "2020-03-16", "2022-06-13", "2022-09-13"]


def lagged_z(x: pd.Series, window: int) -> pd.Series:
    """x_t / std(x over the previous `window` days, lagged one day so day t is excluded)."""
    return x / x.shift(1).rolling(window).std()


def detect(f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Boolean shock flag and its components; |z_spy|>=z & |r|>=min_abs  OR  vix 1-day change z >= vix_z."""
    d = pd.DataFrame(index=f.index)
    d["z_spy"] = lagged_z(f.r_spy, cfg["vol_window"])
    d["vix_z"] = lagged_z(f.vix_d1, cfg["vol_window"])
    d["shock"] = ((d.z_spy.abs() >= cfg["z_spy"]) & (f.r_spy.abs() >= cfg["min_abs_ret"])) | (d.vix_z >= cfg["vix_z"])
    return d


def episodes(dates_pos: np.ndarray, gap: int) -> np.ndarray:
    """Episode ids for sorted trading-day positions: a gap > `gap` trading days starts a new episode."""
    if len(dates_pos) == 0:
        return np.array([], int)
    return np.r_[0, np.cumsum(np.diff(dates_pos) > gap)]


def forward_outcomes(f: pd.DataFrame, h: int = 20) -> pd.DataFrame:
    """Forward outcomes from the close of day t (SPY close-to-close; NaN if the window is incomplete)."""
    c, v = f.close.to_numpy(), f.vix.to_numpy()
    n = len(c)
    out = pd.DataFrame(index=f.index, columns=["fwd_ret_5", "fwd_ret_20", "fwd_maxdd_20", "fwd_vix_chg_20"], dtype=float)
    for i in range(n - h):
        out.iloc[i] = [c[i + 5] / c[i] - 1, c[i + h] / c[i] - 1, (c[i + 1:i + h + 1] / c[i]).min() - 1, v[i + h] / v[i] - 1]
    return out


def build_shock_days() -> pd.DataFrame:
    cfg = load_config()["shock"]
    f = compute_features(load_market())
    f = f[(f.index >= cfg["history_start"]) & (f.index <= cfg["end"])]
    det = detect(f, cfg)
    fo = forward_outcomes(f, cfg["horizon"])
    scaler = Scaler().fit(f.loc[f.index <= cfg["dev_end"], FEATURES].dropna())
    std = scaler.transform(f[FEATURES]).add_suffix("_std")
    pos = np.arange(len(f))
    s = det.shock.fillna(False).to_numpy()
    df = pd.concat([f, det, fo, std], axis=1)
    df["pos"] = pos
    df = df[s & df[FEATURES].notna().all(axis=1)].copy()
    df["direction"] = np.sign(df.r_spy).astype(int)
    df["episode_id"] = episodes(df.pos.to_numpy(), cfg["episode_gap"])
    df = df.reset_index(names="date")
    df.to_parquet(PROCESSED / "shock_days.parquet", index=False)
    f.assign(pos=pos).reset_index(names="date").to_parquet(PROCESSED / "shock_features_all.parquet", index=False)
    pd.to_pickle(scaler, PROCESSED / "shock_scaler.pkl")
    return df, f, det


def diagnostics(df: pd.DataFrame, f: pd.DataFrame, det: pd.DataFrame) -> dict:
    per_year = df.groupby(df.date.dt.year).size()
    log.info("shock days per year:\n%s", per_year.to_string())
    log.info("total shock days %d, episodes %d", len(df), df.episode_id.nunique())
    pos = {d: i for i, d in enumerate(f.index)}
    flagged = set(df.date)
    rows = []
    for d in CURATED:
        t = pd.Timestamp(d)
        if t not in pos:
            rows.append({"date": d, "flagged_exact": False, "flagged_pm2": False, "note": "not a trading day in data"})
            continue
        i = pos[t]
        near = [f.index[j] for j in range(max(0, i - 2), min(len(f), i + 3))]
        rows.append({"date": d, "flagged_exact": t in flagged, "flagged_pm2": any(x in flagged for x in near),
                     "z_spy": float(det.z_spy.get(t, np.nan)), "r_spy": float(f.r_spy.get(t, np.nan)),
                     "note": "face-validity check only; user to verify against sources"})
    fv = pd.DataFrame(rows)
    fv.to_csv(PROCESSED.parent / "curated" / "episodes.csv", index=False)
    save_table(fv, "shock_face_validity")
    misses = fv[~fv.flagged_pm2]
    log.info("face validity: %d/%d flagged within +/-2 days; misses: %s", fv.flagged_pm2.sum(), len(fv), misses.date.tolist())
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f.index, f.close, lw=0.8)
    ax.scatter(df.date, df.close, s=6, c="red")
    ax.set_yscale("log")
    ax.set_title("SPY with shock days")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "shock_spy.png", dpi=120)
    plt.close(fig)
    res = {"n_shock_days": int(len(df)), "n_episodes": int(df.episode_id.nunique()),
           "per_year": {int(k): int(v) for k, v in per_year.items()},
           "n_dev": int((df.date <= load_config()["shock"]["dev_end"]).sum()),
           "face_validity": fv.to_dict(orient="records")}
    update_results("shock_detection", res)
    return res


def main() -> pd.DataFrame:
    df, f, det = build_shock_days()
    diagnostics(df, f, det)
    return df


if __name__ == "__main__":
    main()
