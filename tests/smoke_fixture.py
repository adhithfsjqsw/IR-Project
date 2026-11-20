"""SYNTHETIC fixture for the end-to-end smoke test. Never used for reported results.

Writes a small fake world (12 tickers, random-walk prices with injected 8-K shocks, fake documents) into the directory
given by WHYMOVED_DATA, so that every pipeline stage can be executed without network access. Run via
`python tests/smoke_fixture.py` with WHYMOVED_DATA / WHYMOVED_REPORTS pointing at a temporary directory.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from whymoved.common import INTERIM, PROCESSED, ensure_dirs
from whymoved.data.calendar import effective_day
from whymoved.data.docs import build_docs, build_gold, trading_days

SECTORS = {"Tech": ["alpha", "chip", "cloud"], "Energy": ["oil", "drill", "pipeline"], "Health": ["drug", "trial", "clinic"]}
CAUSES = {"earnings": "Results of Operations quarterly revenue earnings per share guidance",
          "mna": "acquisition merger agreement definitive purchase price closing",
          "leadership": "chief executive officer resigned appointed board director",
          "financing": "senior notes offering credit facility proceeds indenture",
          "restructuring_impairment": "restructuring charges impairment severance workforce reduction",
          "governance": "bylaws amendment annual meeting stockholders vote proposal"}
ITEM = {"earnings": "2.02", "mna": "1.01", "leadership": "5.02", "financing": "2.03",
        "restructuring_impairment": "2.05", "governance": "5.07"}


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(0)
    tickers, rows = [], []
    for si, (sec, words) in enumerate(SECTORS.items()):
        for j in range(4):
            t = f"{sec[:2].upper()}{j}"
            tickers.append(t)
            rows.append({"ticker": t, "name": f"{words[j % 3].title()} {sec} Inc.", "gics_sector": sec,
                         "gics_sub_industry": sec + " sub", "cik": 1000 + len(tickers)})
    comp = pd.DataFrame(rows)
    comp["cik"] = comp.cik.astype("Int64")
    comp.to_parquet(PROCESSED / "companies.parquet", index=False)

    days = pd.bdate_range("2020-06-01", "2023-12-29")
    spy = rng.normal(0.0004, 0.009, len(days))
    secf = {s: rng.normal(0, 0.006, len(days)) for s in SECTORS}
    ret = {"SPY": spy}
    events = []
    for t, s in zip(comp.ticker, comp.gics_sector):
        r = 1.1 * spy + secf[s] + rng.normal(0, 0.007, len(days))
        for d in rng.choice(np.arange(150, len(days)), 14, replace=False):  # injected 8-K shock days
            r[d] += rng.choice([-1, 1]) * rng.uniform(0.06, 0.12)
            events.append((t, days[d], rng.choice(list(CAUSES)), int(np.sign(r[d]))))
        ret[t] = r
    px = pd.DataFrame(ret, index=days)
    adj = 100 * (1 + px).cumprod()
    long = adj.stack().rename("adj_close").to_frame().join(px.stack().rename("ret")).reset_index()
    long.columns = ["date", "ticker", "adj_close", "ret"]
    long.to_parquet(PROCESSED / "prices.parquet", index=False)
    cal = trading_days(long)

    k8 = []
    news = []
    for i, (t, d, cause, sgn) in enumerate(events):
        name = comp.set_index("ticker").loc[t, "name"]
        body = f"Item {ITEM[cause]} {CAUSES[cause]} {name} " + " ".join(rng.choice(CAUSES[cause].split(), 25))
        # filed 08:00 ET on the event day -> effective that day
        ts = pd.Timestamp(d.strftime("%Y-%m-%d") + " 13:00:00", tz="UTC")
        k8.append({"ticker": t, "cik": 1, "accession": f"0000-{i:04d}", "filing_date": d.strftime("%Y-%m-%d"),
                   "acceptance_ts_utc": ts.isoformat(), "items": ITEM[cause], "cause_label": cause,
                   "text": f"{ITEM[cause]} label. " + body, "body": body, "url": "http://x/" + str(i)})
        for k in range(3):  # reaction + same-event news
            news.append({"ticker": t, "published_ts_utc": ts + pd.Timedelta(hours=int(rng.integers(1, 5))),
                         "title": f"{name} {rng.choice(['shares jump', 'announces', 'update'])}",
                         "publisher": rng.choice(["wire.example", "blog.example"]),
                         "url": "http://n/" + str(i * 3 + k), "text": f"{name} " + " ".join(rng.choice(CAUSES[cause].split(), 20)),
                         "has_article": True})
    for t in comp.ticker:  # background noise news
        for d in pd.DatetimeIndex(rng.choice(days[150:], 60, replace=False)):
            news.append({"ticker": t, "published_ts_utc": pd.Timestamp(d.strftime("%Y-%m-%d") + " 15:00:00", tz="UTC"),
                         "title": f"{t} market commentary", "publisher": "wire.example", "url": "http://n/x",
                         "text": f"{t} general market commentary weather sunny", "has_article": True})
    news_df, k8_df = pd.DataFrame(news), pd.DataFrame(k8)
    docs = build_docs(news_df, k8_df, tickers, cal)
    docs.to_parquet(PROCESSED / "docs.parquet", index=False)
    build_gold(docs).to_parquet(PROCESSED / "gold.parquet", index=False)
    (INTERIM / "attribution_tickers.json").write_text(json.dumps(tickers))

    sec = comp.copy()
    sec["kmeans_cluster"], sec["hier_cluster"] = sec.gics_sector.astype("category").cat.codes, 0
    sec["x"], sec["y"], sec["peer_purity"] = rng.normal(size=len(sec)), rng.normal(size=len(sec)), 1.0
    sec["peers"] = [[o for o in comp.ticker[comp.gics_sector == s] if o != t][:3] for t, s in zip(comp.ticker, comp.gics_sector)]
    sec["peer_sims"] = [[0.5, 0.4, 0.3] for _ in range(len(sec))]
    sec.to_parquet(PROCESSED / "sectors.parquet", index=False)
    print("synthetic fixture written:", len(docs), "docs;", len(events), "events")


if __name__ == "__main__":
    main()
