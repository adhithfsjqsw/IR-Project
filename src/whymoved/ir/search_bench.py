"""Search benchmarks: tolerant entity retrieval (9.1), efficiency (9.2), postings compression (9.3).

`python -m whymoved.ir.search_bench`. The efficiency benchmark indexes REAL FNSPID news text (a seeded sample), not 8-Ks.
"""
from __future__ import annotations

import random
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from ..common import INTERIM, PROCESSED, get_logger, load_config
from ..reporting import save_table, update_results
from .tolerant import EntityIndex, kgrams, levenshtein, normalize_name, soundex

log = get_logger("search_bench")

# ---------------- corruption generator (seeded, rule-based) ----------------
PHONETIC_RULES = [("ph", "f"), ("f", "ph"), ("c", "k"), ("k", "c"), ("s", "z"), ("z", "s"), ("ie", "ei"), ("ei", "ie"),
                  ("ck", "k"), ("x", "ks"), ("ee", "ea"), ("ea", "ee"), ("oo", "u"), ("y", "i"), ("i", "y"), ("th", "t"),
                  ("tt", "t"), ("ll", "l"), ("ss", "s"), ("a", "e"), ("e", "a"), ("o", "u"), ("u", "o"), ("ia", "ea"),
                  ("ch", "k"), ("gh", "g"), ("mb", "m"), ("pp", "p")]


def edit(word: str, rng: random.Random) -> str:
    """One random edit: substitution, deletion, insertion or adjacent transposition."""
    if len(word) < 2:
        return word
    i = rng.randrange(len(word))
    op = rng.choice(["sub", "del", "ins", "swap"])
    c = rng.choice("abcdefghijklmnopqrstuvwxyz")
    if op == "sub":
        return word[:i] + c + word[i + 1:]
    if op == "del":
        return word[:i] + word[i + 1:]
    if op == "ins":
        return word[:i] + c + word[i:]
    i = min(i, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def phonetic(word: str, rng: random.Random) -> str:
    """Apply one applicable phonetic substitution rule at a random position (and drop a silent h/doubled letter)."""
    cands = [(m.start(), a, b) for a, b in PHONETIC_RULES for m in re.finditer(re.escape(a), word)]
    if not cands:
        return word
    i, a, b = rng.choice(cands)
    return word[:i] + b + word[i + len(a):]


def corrupt(q: str, kind: str, rng: random.Random) -> str:
    if kind == "edit1":
        return edit(q, rng)
    if kind == "edit2":
        return edit(edit(q, rng), rng)
    return phonetic(q, rng)


def make_queries(comp: pd.DataFrame, seed: int, aliases: dict[str, list[str]]) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for r in comp.itertuples():
        base = {"name": normalize_name(r.name), "ticker": r.ticker.lower()}
        for i, a in enumerate(aliases.get(r.ticker, [])):
            base[f"alias{i}"] = a
        for src, q in base.items():
            if len(q) < 4:
                continue
            for kind in ("edit1", "edit2", "phonetic"):
                c = corrupt(q, kind, rng)
                if c != q:
                    rows.append({"truth": r.ticker, "clean": q, "query": c, "kind": kind, "source": src})
    return pd.DataFrame(rows)


def derive_aliases(comp: pd.DataFrame) -> dict[str, list[str]]:
    """Rule-based aliases: acronym of multi-word names, joined-word form, first two words."""
    out = {}
    for r in comp.itertuples():
        n = normalize_name(r.name)
        w = n.split()
        al = set()
        if len(w) >= 2:
            al.add("".join(x[0] for x in w))
            al.add("".join(w))
            al.add(" ".join(w[:2]))
        out[r.ticker] = sorted(a for a in al if len(a) >= 3 and a != n)
    return out


# ---------------- resolvers ----------------
def r_exact(ent: EntityIndex, q: str) -> list[str]:
    return sorted(ent.terms.get(normalize_name(q), []))


def r_soundex(ent: EntityIndex, q: str) -> list[str]:
    forms = set()
    for w in normalize_name(q).split():
        forms |= ent.sound.get(soundex(w), set())
    ranked = sorted(forms, key=lambda f: (levenshtein(normalize_name(q), f), f))  # tie order only; selection is by Soundex
    return [t for f in ranked for t in sorted(ent.terms[f])]


def _cands(ent: EntityIndex, q: str) -> dict[str, float]:
    qn = normalize_name(q)
    qg = {k: kgrams(qn, k) for k in (2, 3)}
    forms = {f for k in (2, 3) for g in qg[k] for f in ent.gram[k].get(g, ())}
    return {f: max(len(qg[k] & kgrams(f, k)) / len(qg[k] | kgrams(f, k)) for k in (2, 3)) for f in forms}


def r_jaccard(ent: EntityIndex, q: str) -> list[str]:
    c = _cands(ent, q)
    return [t for f in sorted(c, key=lambda f: (-c[f], f)) for t in sorted(ent.terms[f])]


def r_kgram_lev(ent: EntityIndex, q: str) -> list[str]:
    qn = normalize_name(q)
    c = {f: j for f, j in _cands(ent, q).items() if j >= 0.2}
    sc = {f: 0.7 * (1 - levenshtein(qn, f) / max(len(qn), len(f), 1)) + 0.3 * j for f, j in c.items()}
    return [t for f in sorted(sc, key=lambda f: (-sc[f], f)) for t in sorted(ent.terms[f])]


def evaluate(ent: EntityIndex, queries: pd.DataFrame, fn) -> pd.DataFrame:
    rows = []
    for r in queries.itertuples():
        t0 = time.perf_counter()
        seen, ranked = set(), []
        for t in fn(ent, r.query):
            if t not in seen:
                seen.add(t)
                ranked.append(t)
        dt = time.perf_counter() - t0
        rank = ranked.index(r.truth) + 1 if r.truth in ranked else 0
        rows.append({"kind": r.kind, "rank": rank, "lat_ms": dt * 1000, "query": r.query, "truth": r.truth, "top": ranked[:1]})
    return pd.DataFrame(rows)


def summarize(name: str, d: pd.DataFrame) -> list[dict]:
    out = []
    for kind, g in [("all", d), *d.groupby("kind")]:
        out.append({"method": name, "kind": kind, "n": len(g), "acc@1": float((g["rank"] == 1).mean()),
                    "acc@3": float(((g["rank"] >= 1) & (g["rank"] <= 3)).mean()),
                    "mrr": float(np.where(g["rank"] > 0, 1 / np.maximum(g["rank"], 1), 0).mean()),
                    "median_latency_ms": float(g.lat_ms.median())})
    return out


def wildcard_bench(ent: EntityIndex, seed: int, n: int = 300) -> dict:
    rng = random.Random(seed)
    forms = sorted(f for f in ent.terms if len(f) >= 5 and " " not in f)
    res = {"prefix": [], "suffix": [], "middle": []}
    for f in rng.sample(forms, min(n, len(forms))):
        pats = {"prefix": f[: rng.randint(2, 4)] + "*", "suffix": "*" + f[-rng.randint(2, 4):],
                "middle": f[:rng.randint(1, 2)] + "*" + f[-rng.randint(1, 2):]}
        for kind, p in pats.items():
            rx = re.compile("^" + ".*".join(re.escape(x) for x in p.split("*")) + "$")
            truth = {x for x in ent.terms if rx.match(x)}
            got = ent.wildcard(p)
            tp = len(truth & got)
            res[kind].append((tp / max(1, len(got)) if got else 1.0, tp / max(1, len(truth))))
    return {k: {"precision": float(np.mean([a for a, _ in v])), "recall": float(np.mean([b for _, b in v])), "n": len(v)}
            for k, v in res.items()}


def main() -> None:
    cfg = load_config()
    comp = pd.read_parquet(PROCESSED / "companies.parquet")
    recs = comp.to_dict(orient="records")
    aliases = derive_aliases(comp)
    qs = make_queries(comp, cfg["seed"], aliases)
    log.info("tolerant benchmark: %d queries over %d companies", len(qs), len(comp))
    assert qs.equals(make_queries(comp, cfg["seed"], aliases)), "corruption generator must be deterministic"
    base = EntityIndex(recs, first_word_alias=False)
    with_alias = EntityIndex(recs, first_word_alias=True, extra_aliases=aliases)
    runs = {"exact": (base, r_exact), "soundex_only": (base, r_soundex), "kgram_jaccard_only": (base, r_jaccard),
            "kgram+levenshtein": (base, r_kgram_lev), "kgram+lev+alias": (with_alias, r_kgram_lev)}
    rows, fails = [], []
    for name, (ent, fn) in runs.items():
        d = evaluate(ent, qs, fn)
        rows += summarize(name, d)
        if name == "kgram+lev+alias":
            fails = d[d["rank"] != 1].head(8)
    tol = pd.DataFrame(rows)
    save_table(tol, "tolerant_retrieval")
    log.info("tolerant retrieval:\n%s", tol[tol.kind == "all"].round(3).to_string())
    failures = []
    for r in fails.itertuples():
        reason = ("truth not retrieved: too few shared k-grams after corruption" if r.rank == 0
                  else f"truth ranked {r.rank}: outranked by another company's form")
        failures.append({"query": r.query, "truth": r.truth, "top1": r.top[0] if r.top else None, "rank": int(r.rank),
                         "likely_reason": reason})
    wc = wildcard_bench(with_alias, cfg["seed"])
    log.info("wildcards: %s", wc)
    update_results("search_bench_tolerant", {"n_queries": int(len(qs)), "table": tol.to_dict(orient="records"),
                                              "failure_examples": failures, "wildcards": wc})
    efficiency()


# ---------------- 9.2 efficiency + 9.3 compression ----------------
def efficiency(n_docs: int = 40_000, n_queries: int = 1000) -> None:
    import os
    import scipy.sparse as sp

    from .index import PositionalIndex
    from .scoring import bm25_scores, tfidf_scores, top_k
    from .tokenize import tokenize
    from .vbyte import compression_report
    cfg = load_config()
    news = pd.read_parquet(INTERIM / "news_all.parquet", columns=["text", "title"])
    news = news.sample(n=min(n_docs, len(news)), random_state=cfg["seed"])
    t0 = time.perf_counter()
    idx = PositionalIndex()
    for t in news.text:
        idx.add_document(t)
    idx.finalize()
    build_s = time.perf_counter() - t0
    path = PROCESSED / "bench_index.pkl"
    idx.save(path)
    size_mb = os.path.getsize(path) / 1e6
    path.unlink()
    rng = np.random.default_rng(cfg["seed"])
    titles = news.title.fillna("").tolist()
    queries = []
    while len(queries) < n_queries:
        toks = [t for t in tokenize(titles[int(rng.integers(len(titles)))]) if idx.df(t) > 0]
        if len(toks) >= 2:
            queries.append({t: 1.0 for t in toks[: int(rng.integers(2, 5))]})
    # brute-force reference: sparse matrices with the SAME formulas
    terms = {t: i for i, t in enumerate(idx.postings)}
    r_, c_, v_tf, v_bm = [], [], [], []
    avgdl = idx.avgdl
    for t, pl in idx.postings.items():
        idf_t = np.log10(idx.N / len(pl))
        idf_b = np.log(1 + (idx.N - len(pl) + 0.5) / (len(pl) + 0.5))
        for d, pos in pl.items():
            tf = len(pos)
            r_.append(d), c_.append(terms[t])
            v_tf.append((1 + np.log10(tf)) * idf_t / (idx.norms[d] or 1.0))
            v_bm.append(idf_b * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * idx.doc_len[d] / avgdl)))
    M_tf = sp.csr_matrix((v_tf, (r_, c_)), shape=(idx.N, len(terms)))
    M_bm = sp.csr_matrix((v_bm, (r_, c_)), shape=(idx.N, len(terms)))
    res = {}
    for name, fn, M in (("tfidf", lambda q: tfidf_scores(idx, q)[0], M_tf), ("bm25", lambda q: bm25_scores(idx, q)[0], M_bm)):
        lat, ref_lat, agree = [], [], 0
        for q in queries:
            t0 = time.perf_counter()
            sc = fn(q)
            top = [d for d, _ in top_k(sc, 10)]
            lat.append((time.perf_counter() - t0) * 1000)
            qv = sp.csr_matrix(([w * (np.log10(idx.N / idx.df(t)) if name == "tfidf" else 1.0) for t, w in q.items()],
                                ([0] * len(q), [terms[t] for t in q])), shape=(1, len(terms)))
            t0 = time.perf_counter()
            s = np.asarray((M @ qv.T).todense()).ravel()
            ref = list(np.argsort(-s, kind="stable")[:10])
            ref_lat.append((time.perf_counter() - t0) * 1000)
            # correctness: the brute-force leader must be a top-scoring doc for our TAAT scorer (ties allowed; the
            # brute-force scores omit the constant query-norm factor, so compare via our scores)
            agree += int(np.isclose(sc.get(ref[0], 0.0), max(sc.values())) if sc and s.max() > 0 else 1)
        res[name] = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                     "brute_force_p50_ms": float(np.percentile(ref_lat, 50)), "brute_force_p95_ms": float(np.percentile(ref_lat, 95)),
                     "top1_score_agreement": agree / len(queries)}
    comp = compression_report(idx)
    out = {"n_docs": int(idx.N), "n_terms": len(idx.postings), "n_queries": n_queries, "build_seconds": build_s,
           "index_pickle_mb": size_mb, "latency": res, "compression": comp}
    save_table(pd.DataFrame([{"scorer": k, **v} for k, v in res.items()]), "efficiency")
    update_results("search_bench_efficiency", out)
    log.info("efficiency: %s", out)


if __name__ == "__main__":
    main()
