"""Shock-mode market data: SPY, nine sector ETFs, ^VIX (adjusted close + volume), cached."""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from ..common import CACHE, PROCESSED, ensure_dirs, get_logger, load_config

log = get_logger("shock.data")


def load_market() -> pd.DataFrame:
    """Long frame: date, ticker, close, volume. Downloads once per ticker (cached)."""
    out = PROCESSED / "shock_market.parquet"
    if out.exists():
        return pd.read_parquet(out)
    ensure_dirs()
    cfg = load_config()["shock"]
    d = CACHE / "shock"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in cfg["tickers"] + [cfg["vix"]]:
        f = d / f"{t.replace('^', '')}.parquet"
        if not f.exists():
            for a in range(5):
                try:
                    df = yf.download(t, start=cfg["download_start"], end="2024-01-02", auto_adjust=True, progress=False)
                    if len(df):
                        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                        df[["Close", "Volume"]].to_parquet(f)
                        break
                except Exception as e:
                    log.warning("%s attempt %d: %s", t, a + 1, e)
                time.sleep(3 * 2 ** a)
            time.sleep(1)
        if not f.exists():
            raise RuntimeError(f"could not download {t}")
        x = pd.read_parquet(f).reset_index()
        x.columns = ["date", "close", "volume"]
        x["ticker"] = t
        frames.append(x)
    m = pd.concat(frames, ignore_index=True)
    m["date"] = pd.to_datetime(m.date).dt.tz_localize(None)
    m.to_parquet(out, index=False)
    log.info("market data:\n%s", m.groupby("ticker").date.agg(["min", "max", "count"]))
    return m


if __name__ == "__main__":
    load_market()
