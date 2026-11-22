"""Judged-relevance evaluation of move->document retrieval (easy mode, complete qrels).

The 8-K answer key marks only the filing as relevant, so news about the same event counts as a miss and the metric
underestimates every system. In easy mode the candidate pool for a move (the ticker's documents on days t-1 and t) is
small (median ~26), so we can judge EVERY document in it instead of pooling system outputs: the qrels are complete and
no system is penalised for retrieving something unjudged.

Judge = the local LLM (Ollama). Its labels are noisy and are validated, not trusted blindly:
  * recall on the known-relevant gold 8-Ks (LLM-judged, not auto-accepted),
  * false-positive rate on random documents of OTHER tickers,
  * a CSV of judgments for human spot-checking: reports/tables/judge_review_sample.csv.
Grades: gold 8-K = 2; LLM YES = 2; PARTIAL = 1; NO = 0. nDCG@10 uses the grades; P@5 and MRR count grade 2 only.

Workflow: judge pools for a dev and a test sample -> tune burst parameters on DEV -> report on TEST (paired bootstrap).
`python -m whymoved.attribution.judged [--dev N] [--test N] [--no-judge]`
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import pandas as pd

from ..common import PROCESSED, REPORTS, get_logger, load_config, seed_all
from ..data.docs import trading_days
from ..ir.build import load_or_build
from ..rag.llm import OllamaBackend
from ..reporting import save_table, update_results
from .candidates import candidate_pool, price_move_docs, window_days
from .evaluate import Evaluator, System
from .gold import gold_lookup
from .metrics import bootstrap_ci, paired_bootstrap_p

log = get_logger("judged")
CACHE = REPORTS / "judgments.jsonl"
GRADE = {"YES": 2, "PARTIAL": 1, "NO": 0}

SYSTEM = ("You judge document relevance for an information-retrieval evaluation. Be strict and literal. "
          "Answer with exactly one word.")


def prompt(name: str, ticker: str, date: str, mv: dict, doc: dict) -> str:
    return (f"Stock: {name} ({ticker}). On {date} it moved {mv['ret']:+.1%} (market-adjusted {mv['abnormal']:+.1%}).\n\n"
            f"Document ({doc['doc_type']}, dated {doc['day']}): {doc['title']}\n{doc['text'][:700]}\n\n"
            "Does this document report or directly discuss a company-specific event or news that could explain this "
            "move? Answer YES (clearly explains it), PARTIAL (related to the company's situation but not clearly the "
            "cause), or NO (unrelated, a generic market summary, or it only repeats the price move).")


def parse_label(raw: str) -> str | None:
    w = raw.strip().upper().replace("*", "").split()
    return w[0].strip(".,:;") if w and w[0].strip(".,:;") in GRADE else None


class Judge:
    def __init__(self, idx, docs: pd.DataFrame, names: dict[str, str], moves: pd.DataFrame) -> None:
        self.idx, self.docs, self.names = idx, docs, names
        self.mv = {(r.ticker, r.date.strftime("%Y-%m-%d")): {"ret": float(r.ret), "abnormal": float(r.abnormal)}
                   for r in moves.itertuples()}
        self.cache: dict[tuple[str, str, int], dict] = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[(r["ticker"], r["date"], r["doc_id"])] = r
        self.llm = OllamaBackend(model=load_config().get("rag", {}).get("model", "qwen2.5:7b"), temperature=0.0,
                                 max_tokens=4, timeout=120)

    def label(self, ticker: str, date: str, doc_id: int, kind: str = "pool") -> dict:
        doc_id = int(doc_id)  # numpy ints are not JSON-serializable
        key = (ticker, date, doc_id)
        if key in self.cache:
            return self.cache[key]
        row = self.docs.iloc[doc_id]
        doc = {"doc_type": self.idx.meta[doc_id]["doc_type"], "day": self.idx.meta[doc_id]["effective_day"],
               "title": str(row.title), "text": str(row.text)}
        raw = self.llm.generate(SYSTEM, prompt(self.names[ticker], ticker, date, self.mv[(ticker, date)], doc))
        rec = {"ticker": ticker, "date": date, "doc_id": doc_id, "label": parse_label(raw), "raw": raw[:20], "kind": kind,
               "doc_type": doc["doc_type"], "title": doc["title"][:120]}
        self.cache[key] = rec
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


def sample(moves: pd.DataFrame, pos: dict, lo: str, hi: str, n: int, seed: int) -> pd.DataFrame:
    m = moves[(moves.date >= lo) & (moves.date <= hi) & moves.date.map(lambda d: d.strftime("%Y-%m-%d") in pos)]
    return m.sample(n=min(n, len(m)), random_state=seed).sort_values(["date", "ticker"])


def grades_for(judge: Judge, ev: Evaluator, idx, ticker: str, day: str, gold: dict) -> dict[int, int]:
    """Complete qrels for one move: grade of every document in the easy-mode pool."""
    pool = candidate_pool(idx, window_days(ev.cal, ev.pos, day), ticker)
    g = gold.get((ticker, day), set())
    out = {}
    for d in sorted(pool):
        lab = judge.label(ticker, day, d)["label"]
        out[d] = 2 if d in g else GRADE.get(lab, 0)
    return out


def metrics(ranked: list[int], qrels: dict[int, int]) -> dict[str, float] | None:
    """nDCG@10 (graded), P@5 and MRR on grade-2 docs. None if the pool has nothing relevant."""
    if not any(v == 2 for v in qrels.values()):
        return None
    gains = [qrels.get(d, 0) for d in ranked[:10]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    first = next((i + 1 for i, d in enumerate(ranked) if qrels.get(d, 0) == 2), None)
    return {"ndcg10": dcg / idcg if idcg else 0.0, "p5": sum(1 for g in gains[:5] if g == 2) / 5.0,
            "mrr": 1.0 / first if first else 0.0}


def evaluate_systems(ev: Evaluator, moves: pd.DataFrame, systems: list[System], qrels: dict, n_rand: int = 200):
    """-> {system name: DataFrame(rows aligned on moves with relevant docs)}."""
    rng = np.random.default_rng(0)
    out = {}
    for s in systems:
        rows = []
        r = ev.run(moves, s, "easy", require_gold=False, return_top=True, k=200)
        for rec in r.itertuples():
            q = qrels.get((rec.ticker, rec.date))
            if q is None:
                continue
            if s.method == "random":
                pool = list(q)
                ms = [metrics(list(rng.permutation(pool)), q) for _ in range(n_rand)]
                m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]} if ms and ms[0] else None
            else:
                m = metrics(list(rec.ranked), q)
            if m:
                rows.append({"ticker": rec.ticker, "date": rec.date, **m})
        out[s.name] = pd.DataFrame(rows)
    return out


def table(per: dict[str, pd.DataFrame], ref: str, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for name, df in per.items():
        row = {"system": name, "n_moves": len(df)}
        for k in ("ndcg10", "p5", "mrr"):
            mean, lo, hi = bootstrap_ci(df[k].to_numpy(), n_boot)
            row[k], row[k + "_lo"], row[k + "_hi"] = mean, lo, hi
        if name != ref and len(df) == len(per[ref]):
            d, p = paired_bootstrap_p(df.ndcg10.to_numpy(), per[ref].ndcg10.to_numpy(), n_boot)
            row["ndcg_diff_vs_ref"], row["p_vs_ref"] = d, p
        rows.append(row)
    return pd.DataFrame(rows)


def validate_judge(judge: Judge, idx, ev: Evaluator, gold: dict, moves: pd.DataFrame, n: int = 30) -> dict:
    """LLM recall on known-relevant gold 8-Ks and false-positive rate on random docs of other tickers (same day)."""
    rng = np.random.default_rng(1)
    gm = [(r.ticker, r.date.strftime("%Y-%m-%d")) for r in moves.itertuples()
          if (r.ticker, r.date.strftime("%Y-%m-%d")) in gold and r.date.strftime("%Y-%m-%d") in ev.pos]
    gm = [gm[i] for i in rng.choice(len(gm), size=min(n, len(gm)), replace=False)]
    rec = [judge.label(t, d, next(iter(gold[(t, d)])), "validate-gold")["label"] for t, d in gm]
    fp = []
    for t, d in gm:
        others = [x for x in idx.day_docs.get(d, []) if idx.meta[x].get("ticker") not in (t, None)]
        fp.append(judge.label(t, d, int(rng.choice(others)), "validate-random")["label"] if others else None)
    f = lambda xs, ok: float(np.mean([x in ok for x in xs if x is not None])) if any(x is not None for x in xs) else None
    return {"n": len(gm), "gold_judged_yes": f(rec, {"YES"}), "gold_judged_yes_or_partial": f(rec, {"YES", "PARTIAL"}),
            "random_other_ticker_judged_yes": f(fp, {"YES"}), "random_other_ticker_judged_yes_or_partial": f(fp, {"YES", "PARTIAL"}),
            "unparsed_rate": float(np.mean([x is None for x in rec + fp]))}


def main(n_dev: int = 20, n_test: int = 35, do_judge: bool = True) -> None:
    cfg = load_config()
    seed_all(cfg["seed"])
    idx = load_or_build()
    comp = pd.read_parquet(PROCESSED / "companies.parquet")
    names = dict(zip(comp.ticker, comp.name))
    cal = trading_days(pd.read_parquet(PROCESSED / "prices.parquet"), cfg["market_proxy"])
    gold = gold_lookup(pd.read_parquet(PROCESSED / "gold.parquet"))
    moves = pd.read_parquet(PROCESSED / "moves.parquet")
    docs = pd.read_parquet(PROCESSED / "docs.parquet")
    ev = Evaluator(idx, names, gold, cal, cfg["bm25"]["k1"], cfg["bm25"]["b"], cfg["seed"], price_move_docs(idx))
    R = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))["h1"]
    tuned, extras = R["tuned"], R["extras_tuned"]
    dev = sample(moves, ev.pos, cfg["dev"]["start"], cfg["dev"]["end"], n_dev, cfg["seed"])
    test = sample(moves, ev.pos, cfg["test"]["start"], cfg["test"]["end"], n_test, cfg["seed"])
    judge = Judge(idx, docs, names, moves)

    def build_qrels(df: pd.DataFrame) -> dict:
        q, t0 = {}, time.time()
        for i, r in enumerate(df.itertuples(), 1):
            day = r.date.strftime("%Y-%m-%d")
            q[(r.ticker, day)] = grades_for(judge, ev, idx, r.ticker, day, gold)
            log.info("judged %d/%d moves (%s %s, pool %d) %.0fs elapsed", i, len(df), r.ticker, day,
                     len(q[(r.ticker, day)]), time.time() - t0)
        return q

    if not do_judge:
        judge.llm = None  # cache only; any miss raises
    qd, qt = build_qrels(dev), build_qrels(test)
    q_all = {**qd, **qt}
    val = validate_judge(judge, idx, ev, gold, moves)
    log.info("judge validation: %s", val)
    rel = {k: sum(1 for v in q.values() if v == 2) for k, q in q_all.items()}
    cover = {"n_dev": len(dev), "n_test": len(test), "docs_judged": int(sum(len(q) for q in q_all.values())),
             "mean_pool": float(np.mean([len(q) for q in q_all.values()])),
             "moves_with_relevant": int(sum(1 for v in rel.values() if v > 0)),
             "mean_relevant_per_move": float(np.mean(list(rel.values()))),
             "share_of_relevant_that_are_not_the_8k": None}
    all_g = [(k, d, g) for k, q in q_all.items() for d, g in q.items() if g == 2]
    cover["share_of_relevant_that_are_not_the_8k"] = float(np.mean([d not in gold.get(k, set()) for k, d, g in all_g])) if all_g else None

    # ---- tune burst on DEV (judged nDCG@10, easy mode) ----
    base = dict(direction_w=tuned["direction_w"], prf=tuned["prf"], lam=tuned["lambda"])
    grid = [(w, n) for w in (0.3, 0.6, 1.0, 1.5) for n in (5, 10)]
    scores = {}
    for w, n in grid:
        per = evaluate_systems(ev, dev, [System("B", burst_w=w, burst_n=n, **base)], qd)["B"]
        scores[(w, n)] = float(per.ndcg10.mean()) if len(per) else float("nan")
    a5 = float(evaluate_systems(ev, dev, [System("A5", **base)], qd)["A5"].ndcg10.mean())
    bw, bn = max(grid, key=lambda g: scores[g])
    log.info("DEV judged nDCG@10: A5 %.3f; burst grid best %s = %.3f", a5, (bw, bn), scores[(bw, bn)])

    # ---- tune BM25 (k1, b) and the time-decay strength on DEV (judged nDCG@10). Adopted only if it clearly beats A5. ----
    seed, pm = cfg["seed"], price_move_docs(idx)
    grid3 = [(k1, b, lam) for k1 in (0.9, 1.2, 1.8) for b in (0.3, 0.5, 0.75, 0.9) for lam in (0.5, 1.0, 2.0, 4.0)]
    sc3, evs = {}, {}
    for k1, b, lam in grid3:
        evs.setdefault((k1, b), Evaluator(idx, names, gold, cal, k1, b, seed, pm))
        per = evaluate_systems(evs[(k1, b)], dev, [System("T", direction_w=tuned["direction_w"], prf=tuned["prf"], lam=lam)], qd)["T"]
        sc3[(k1, b, lam)] = float(per.ndcg10.mean()) if len(per) else float("nan")
    bk, bb, bl = max(grid3, key=lambda g: sc3[g])
    margin = 0.02
    dev_selected = sc3[(bk, bb, bl)] > a5 + margin
    log.info("DEV judged nDCG@10: A5 %.3f; best (k1=%s,b=%s,lam=%s) = %.3f -> %s", a5, bk, bb, bl, sc3[(bk, bb, bl)],
             "beats A5 on DEV by the margin (must be confirmed on TEST)" if dev_selected else f"no DEV gain > +{margin}")

    # ---- report on TEST ----
    systems = [System("A0 random", method="random"), System("A2 BM25 name-only"), System("A5 + time prior", **base),
               System("B1 A5 + burst terms", burst_w=bw, burst_n=bn, **base),
               System("E1 A5 + event lexicon (8-K style)", event_w=extras["event_w"], **base)]
    per = evaluate_systems(ev, test, systems, qt)
    per.update(evaluate_systems(evs[(bk, bb)], test, [System("B2 A5 w/ DEV-tuned k1,b,lambda", direction_w=tuned["direction_w"],
                                                              prf=tuned["prf"], lam=bl)], qt))
    tab = table(per, "A5 + time prior")
    save_table(tab, "judged_easy")
    b2 = tab[tab.system.str.startswith("B2")].iloc[0]
    confirmed = bool(dev_selected and b2.ndcg_diff_vs_ref > 0 and b2.p_vs_ref < 0.05)
    log.info("TEST judged (easy):\n%s", tab[["system", "n_moves", "ndcg10", "p5", "mrr"]].round(3).to_string())

    # human review sample
    rng = np.random.default_rng(2)
    recs = [r for r in judge.cache.values() if r["kind"] == "pool"]
    rows = [recs[i] for i in rng.choice(len(recs), size=min(60, len(recs)), replace=False)]
    save_table(pd.DataFrame([{k: r[k] for k in ("ticker", "date", "doc_type", "title", "label")} for r in rows]),
               "judge_review_sample")
    update_results("judged", {"coverage": cover, "judge_validation": val, "burst_tuned": {"burst_w": bw, "burst_n": bn,
                   "dev_ndcg10": scores[(bw, bn)], "dev_ndcg10_A5": a5, "grid": {str(k): v for k, v in scores.items()}},
                   "bm25_tuned": {"k1": bk, "b": bb, "lambda": bl, "dev_ndcg10": sc3[(bk, bb, bl)], "dev_selected": bool(dev_selected), "confirmed_on_test": confirmed,
                                  "required_margin": margin, "grid": {str(k): v for k, v in sc3.items()}},
                   "table_easy": tab.to_dict(orient="records")})


if __name__ == "__main__":
    a = sys.argv[1:]
    get = lambda f, d: int(a[a.index(f) + 1]) if f in a else d
    main(get("--dev", 20), get("--test", 35), "--no-judge" not in a)
