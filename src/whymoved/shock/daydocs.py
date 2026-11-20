"""Day documents from FNSPID headlines (all tickers, 1999-2023) and the T1 / T2 (burst) day vectors."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from ..common import CACHE, PROCESSED, RAW, get_logger, load_config
from ..data.calendar import effective_days_vec
from ..data.fnspid_download import download_file
from ..ir.tokenize import stem, tokenize
from .data import load_market

log = get_logger("shock.daydocs")
FILES = ["Stock_news/All_external.csv", "Stock_news/nasdaq_exteral_data.csv"]
MONTHS = ("january february march april may june july august september october november december "
          "jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday "
          "mon tue tues wed thu thur thurs fri").split()


def calendar_days() -> list[str]:
    m = load_market()
    return sorted(m[m.ticker == "SPY"].date.dt.strftime("%Y-%m-%d").unique())


def scan_headlines() -> None:
    """Chunk-wise scan of both news files -> cached (day, title) parquet per chunk + symbol/publisher sidecars."""
    cfg = load_config()["shock"]
    cal = calendar_days()
    start, end = cfg["history_start"], cfg["end"]
    for fname in FILES:
        path = RAW / "fnspid" / fname
        if not path.exists():
            download_file(fname)
        tag = fname.split("/")[-1].split(".")[0]
        cdir = CACHE / "daydocs" / tag
        cdir.mkdir(parents=True, exist_ok=True)
        for i, ch in enumerate(tqdm(pd.read_csv(path, usecols=["Date", "Article_title", "Stock_symbol", "Publisher"],
                                                chunksize=500_000, dtype=str), desc=tag)):
            cf = cdir / f"c{i:05d}.parquet"
            if cf.exists():
                continue
            side = {"symbols": sorted(set(ch.Stock_symbol.dropna().str.lower())),
                    "publishers": sorted(set(ch.Publisher.dropna().str.lower()))}
            ch = ch.dropna(subset=["Date", "Article_title"])
            ts = pd.to_datetime(ch.Date, utc=True, errors="coerce")
            ok = ts.notna()
            ch, ts = ch[ok], ts[ok]
            if len(ch):
                eff = effective_days_vec(ts, cal)
                part = pd.DataFrame({"day": eff.to_numpy(), "title": ch.Article_title.to_numpy()}).dropna()
                part = part[(part.day >= start) & (part.day <= end)]
                part = part.assign(k=part.title.str.lower().str.strip()).drop_duplicates(["day", "k"]).drop(columns="k")
            else:
                part = pd.DataFrame({"day": [], "title": []})
            part.to_parquet(cf, index=False)
            (cdir / f"c{i:05d}.json").write_text(json.dumps(side))


def extra_stopwords() -> set[str]:
    """Tickers, company-name tokens, publisher names, month/weekday names (stemmed, to match the tokenizer output)."""
    words = set(MONTHS)
    for f in (CACHE / "daydocs").glob("*/*.json"):
        d = json.loads(f.read_text())
        for s in d["symbols"] + d["publishers"]:
            words.update(re.findall(r"[a-z0-9]+", s))
    comp = pd.read_parquet(PROCESSED / "companies.parquet")
    for n in comp.name:
        words.update(re.findall(r"[a-z0-9]+", n.lower()))
    return {stem(w) for w in words}


def build_day_docs() -> pd.DataFrame:
    """processed/day_docs.parquet: day, n_headlines, thin_text, text (<= max_headlines headlines kept)."""
    out = PROCESSED / "day_docs.parquet"
    if out.exists():
        return pd.read_parquet(out)
    cfg = load_config()["shock"]
    scan_headlines()
    parts = [pd.read_parquet(f) for f in sorted((CACHE / "daydocs").glob("*/*.parquet"))]
    h = pd.concat([p for p in parts if len(p)], ignore_index=True)
    h = h.assign(k=h.title.str.lower().str.strip()).drop_duplicates(["day", "k"]).drop(columns="k")
    log.info("unique (day, headline) pairs: %d", len(h))
    stop = extra_stopwords()
    # approximate headline-level IDF from a seeded 10% sample (used only to choose which headlines to keep)
    samp = h.sample(frac=0.1, random_state=load_config()["seed"]).title
    df = Counter(t for title in samp for t in set(tokenize(title)) - stop)
    ns = len(samp)
    idf = lambda t: np.log10(ns / df.get(t, 1))
    rows = []
    for day, g in tqdm(h.groupby("day"), desc="day docs"):
        titles = g.title.tolist()
        n = len(titles)
        if n > cfg["max_headlines"]:
            sc = [sum(idf(t) for t in set(tokenize(x)) - stop) for x in titles]
            keep = np.argsort(-np.array(sc), kind="stable")[: cfg["max_headlines"]]
            titles = [titles[i] for i in sorted(keep)]
        rows.append({"day": day, "n_headlines": n, "thin_text": n < cfg["min_headlines"], "text": " . ".join(titles)})
    d = pd.DataFrame(rows)
    d.to_parquet(out, index=False)
    cov = d.assign(year=d.day.str[:4]).groupby("year").agg(days=("day", "size"), median_headlines=("n_headlines", "median"),
                                                          thin_share=("thin_text", "mean"))
    log.info("headline coverage per year:\n%s", cov.to_string())
    cov.reset_index().to_csv(PROCESSED.parent.parent / "reports" / "tables" / "headline_coverage.csv", index=False)
    return d


@dataclass
class DayVectors:
    """Counts for every trading day + frozen DEV idf; T1 (tf-idf) and T2 (burst-weighted) views."""
    days: list[str]
    vocab: dict[str, int]
    counts: sp.csr_matrix
    idf: np.ndarray
    thin: np.ndarray
    baseline: int = 60

    @property
    def pos(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.days)}

    def _norm(self, X: sp.csr_matrix) -> sp.csr_matrix:
        n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        n[n == 0] = 1
        return sp.csr_matrix(sp.diags(1 / n) @ X)

    def t1(self, days: list[str]) -> sp.csr_matrix:
        X = self.counts[[self.pos[d] for d in days]].copy().astype(float)
        X.data = 1 + np.log10(X.data)
        return self._norm(X @ sp.diags(self.idf))

    def t2(self, days: list[str]) -> sp.csr_matrix:
        """tf / (mean tf over the previous `baseline` trading days + 1) * idf, L2-normalised. Day t is never in the baseline."""
        rows = []
        for d in days:
            i = self.pos[d]
            lo = max(0, i - self.baseline)
            base = np.asarray(self.counts[lo:i].sum(axis=0)).ravel() / self.baseline if i > lo else np.zeros(self.counts.shape[1])
            r = self.counts[i].toarray().ravel().astype(float)
            rows.append(sp.csr_matrix(r / (base + 1.0) * self.idf))
        return self._norm(sp.vstack(rows).tocsr())


def build_day_vectors(min_df: int = 3) -> DayVectors:
    """Vocabulary from all days with df >= min_df; idf FROZEN on DEV non-thin days (unseen terms get the max idf)."""
    cfg = load_config()["shock"]
    docs = build_day_docs().set_index("day")
    days = calendar_days()
    stop = extra_stopwords()
    toks = {d: [t for t in tokenize(docs.text[d]) if t not in stop] if d in docs.index else [] for d in tqdm(days, desc="tokens")}
    df_all = Counter(t for d in days for t in set(toks[d]))
    vocab = {t: i for i, t in enumerate(sorted(t for t, c in df_all.items() if c >= min_df))}
    rows, cols, vals = [], [], []
    for i, d in enumerate(days):
        for t, c in Counter(toks[d]).items():
            j = vocab.get(t)
            if j is not None:
                rows.append(i), cols.append(j), vals.append(c)
    C = sp.csr_matrix((vals, (rows, cols)), shape=(len(days), len(vocab)), dtype=float)
    thin = np.array([(d not in docs.index) or bool(docs.thin_text[d]) for d in days])
    dev = np.array([d <= cfg["dev_end"] for d in days]) & ~thin
    dfd = np.asarray((C[dev] > 0).sum(axis=0)).ravel()
    n = dev.sum()
    idf = np.log10(n / np.maximum(dfd, 1))  # df = 0 -> same as df = 1 (maximum idf)
    return DayVectors(days, vocab, C, idf, thin, cfg["burst_baseline_days"])


def top_terms(vec: sp.csr_matrix, dv: DayVectors, i: int, k: int = 10) -> list[str]:
    inv = {j: t for t, j in dv.vocab.items()}
    r = vec[i].toarray().ravel()
    return [inv[j] for j in np.argsort(-r)[:k] if r[j] > 0]
