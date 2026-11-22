"""RAG evaluation: does retrieval help an LLM name the cause of a move? `python -m whymoved.rag.evaluate [N]`.

Gold = cause label of the 8-K filed for that ticker-day (derived from 8-K item codes, independent of news text).
Conditions: none (LLM alone), news (RAG over news only), all (RAG over news + 8-K). Resumable JSONL cache.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ..common import REPORTS, get_logger
from .answer import answer_move
from .llm import make_backend

log = get_logger("rag_eval")
CACHE = REPORTS / "rag_eval_raw.jsonl"
CONDS = ("none", "news", "all")


def _labels(s, key) -> set[str]:
    return {s.idx.meta[d]["cause_label"] for d in s.gold[key] if s.idx.meta[d].get("cause_label")}


def sample_moves(s, n: int, seed: int, non_earnings: bool = False) -> list[tuple[str, str]]:
    """Default: random sample of test-period moves with a labelled gold 8-K. non_earnings: instead take moves whose
    gold label is NOT earnings, from dev + test (the LLM is never trained on either, so both are fair game)."""
    m = s.moves if non_earnings else s.moves[s.moves.date >= pd.Timestamp(s.cfg["test"]["start"])]
    keys = [(r.ticker, r.date.strftime("%Y-%m-%d")) for r in m.itertuples()
            if (r.ticker, r.date.strftime("%Y-%m-%d")) in s.gold and r.date.strftime("%Y-%m-%d") in s.pos]
    keys = [k for k in keys if _labels(s, k) and (not non_earnings or "earnings" not in _labels(s, k))]
    rng = np.random.default_rng(seed)
    return [keys[i] for i in sorted(rng.choice(len(keys), size=min(n, len(keys)), replace=False))]


def _load_cache() -> list[dict]:
    return [json.loads(x) for x in CACHE.read_text(encoding="utf-8").splitlines() if x.strip()] if CACHE.exists() else []


def run(n: int = 40, non_earnings: bool = False) -> None:
    from ..api.main import _explain, state
    s = state()
    cfg = s.cfg.get("rag", {})
    llm = make_backend(cfg)
    k, max_chars = cfg.get("top_k", 5), cfg.get("max_chars", 1200)
    done = {(r["ticker"], r["date"], r["cond"]) for r in _load_cache()}
    for ticker, date in sample_moves(s, n, s.cfg["seed"], non_earnings):
        gold_ids = s.gold[(ticker, date)]
        gold_labels = sorted({s.idx.meta[d]["cause_label"] for d in gold_ids if s.idx.meta[d].get("cause_label")})
        mv = s.moves[(s.moves.ticker == ticker) & (s.moves.date == pd.Timestamp(date))].iloc[0]
        move = {"ret": float(mv.ret), "abnormal": float(mv.abnormal), "z": float(mv.z)}
        hits = _explain(ticker, date, "easy").results
        for cond in CONDS:
            if (ticker, date, cond) in done:
                continue
            pool = [h for h in hits if cond == "all" or h.doc_type == "news"] if cond != "none" else []
            pool = pool[:k]
            srcs = [{"doc_type": h.doc_type, "effective_day": h.effective_day, "title": h.title,
                     "text": str(s.docs.iloc[h.doc_id].text)} for h in pool]
            t = time.time()
            out = answer_move(llm, ticker, s.names[ticker], date, move, srcs, max_chars, ask_cause=True,
                              allow_empty=True)
            rec = {"ticker": ticker, "date": date, "cond": cond, "gold": gold_labels, "pred": out["cause"],
                   "cited": out["cited"], "invalid": out["invalid"], "n_src": len(srcs),
                   "secs": round(time.time() - t, 1),
                   "gold_in_context": bool(set(gold_ids) & {h.doc_id for h in pool}), "answer": out["answer"],
                   "gold_positions": [i + 1 for i, h in enumerate(pool) if h.doc_id in gold_ids],
                   "non_earnings_run": non_earnings}
            rec["cited_gold"] = bool(set(rec["cited"]) & set(rec["gold_positions"]))
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log.info("%s %s %-4s gold=%s pred=%s (%.0fs)", ticker, date, cond, gold_labels, out["cause"], rec["secs"])


def summarize() -> dict:
    df = pd.DataFrame(_load_cache())
    df["correct"] = [p in g for p, g in zip(df.pred, df.gold)]
    df["abstain"] = df.pred.eq("unknown") | df.pred.isna()
    base = pd.Series([g[0] for g in df[df.cond == "none"].gold]).value_counts(normalize=True)
    res = {"n_moves": int((df.cond == "none").sum()),
           "majority_class_baseline": float(base.iloc[0]) if len(base) else None,
           "majority_class": base.index[0] if len(base) else None, "conditions": {}}
    for c in CONDS:
        d = df[df.cond == c]
        if d.empty:
            continue
        ans = d[~d.abstain]
        gi = d[d.gold_in_context]
        gn = d[~d.gold_in_context]
        res["conditions"][c] = {
            "n": int(len(d)), "accuracy": float(d.correct.mean()), "abstain_rate": float(d.abstain.mean()),
            "accuracy_when_answered": float(ans.correct.mean()) if len(ans) else None,
            "gold_doc_in_context": float(d.gold_in_context.mean()),
            "accuracy_gold_in_context": float(gi.correct.mean()) if len(gi) else None,
            "accuracy_gold_not_in_context": float(gn.correct.mean()) if len(gn) else None,
            "invalid_citation_rate": float((d.invalid.map(len) > 0).mean()),
            "answers_with_citation": float((d.cited.map(len) > 0).mean()) if c != "none" else None,
            "mean_secs": float(d.secs.mean())}
    # breakdowns that do not depend on one dominant class
    df["gold0"] = [g[0] for g in df.gold]
    df["is_earn"] = df.gold0.eq("earnings")
    res["by_gold_class"] = {}
    for c in CONDS:
        d = df[df.cond == c]
        if d.empty:
            continue
        rec = {k: float(v) for k, v in d.groupby("gold0").correct.mean().items()}
        res["by_gold_class"][c] = {"recall_per_class": rec, "macro_recall": float(np.mean(list(rec.values()))),
                                   "n_per_class": {k: int(v) for k, v in d.gold0.value_counts().items()},
                                   "accuracy_earnings": float(d[d.is_earn].correct.mean()) if d.is_earn.any() else None,
                                   "accuracy_non_earnings": float(d[~d.is_earn].correct.mean()) if (~d.is_earn).any() else None}
    if "cited_gold" in df:
        for c in ("news", "all"):
            d = df[(df.cond == c) & df.cited_gold.notna()]
            if len(d):
                res["conditions"][c]["cites_gold_doc_rate"] = float(d.cited_gold.astype(bool).mean())
                res["conditions"][c]["cites_gold_doc_n"] = int(len(d))
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 40
    run(n, non_earnings="--non-earnings" in sys.argv)
    r = summarize()
    (REPORTS / "rag_eval.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps(r, indent=2))
