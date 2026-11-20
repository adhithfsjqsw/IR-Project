"""Source reputation: how often a publisher's news agrees (by predicted cause) with a same-day 8-K. DEV period only."""
from __future__ import annotations

import pandas as pd

from ..common import PROCESSED, REPORTS, get_logger, load_config
from ..reporting import save_table, update_results

log = get_logger("reputation")
TABLE = REPORTS / "tables" / "publisher_reputation.csv"
SHRINK = 10


def compute_reputation() -> pd.DataFrame:
    """For news on ticker-days with an 8-K (DEV period), confirmed = classifier label matches an 8-K cause_label.

    reputation = (confirmed + SHRINK * global_rate) / (n + SHRINK).
    """
    from ..attribution.cause import predict_cause
    cfg = load_config()
    docs = pd.read_parquet(PROCESSED / "docs.parquet")
    dev = docs[(docs.effective_day >= cfg["dev"]["start"]) & (docs.effective_day <= cfg["dev"]["end"])]
    k = dev[dev.doc_type == "8k"].groupby(["ticker", "effective_day"]).cause_label.agg(set).rename("causes")
    n = dev[dev.doc_type == "news"].merge(k, left_on=["ticker", "effective_day"], right_index=True)
    if n.empty:
        log.warning("no news on 8-K days in DEV; reputation table is empty")
        return pd.DataFrame(columns=["publisher", "n", "confirmed", "reputation"])
    n = n.assign(pred=predict_cause(n.text.tolist()))
    n["confirmed"] = [p in c for p, c in zip(n.pred, n.causes)]
    g = n.confirmed.mean()
    t = n.groupby("publisher").agg(n=("confirmed", "size"), confirmed=("confirmed", "sum")).reset_index()
    t["reputation"] = (t.confirmed + SHRINK * g) / (t.n + SHRINK)
    t = t.sort_values("n", ascending=False)
    t.attrs["global_rate"] = float(g)
    save_table(t, "publisher_reputation")
    update_results("reputation", {"global_rate": float(g), "n_news_on_8k_days_dev": int(len(n)),
                                  "table": t.to_dict(orient="records")})
    return t


def make_prior():
    """Per-doc multiplicative prior (reputation / global rate for news; 1.0 for 8-Ks and unknown sources)."""
    from ..ir.build import load_or_build
    if not TABLE.exists():
        return None
    t = pd.read_csv(TABLE)
    if t.empty:
        return None
    g = float((t.confirmed.sum()) / max(1, t.n.sum()))
    rep = dict(zip(t.publisher, t.reputation / max(g, 1e-9)))
    idx = load_or_build()

    def prior(d: int) -> float:
        m = idx.meta[d]
        return rep.get(m.get("publisher"), 1.0) if m.get("doc_type") == "news" else 1.0
    return prior


if __name__ == "__main__":
    print(compute_reputation().head(20))
