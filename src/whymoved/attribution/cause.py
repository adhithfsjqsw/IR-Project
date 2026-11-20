"""Phase 4: cause-type classifier on 8-K texts (NB / Rocchio / kNN, from scratch). `python -m whymoved.attribution.cause`."""
from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd

from ..data.edgar import ITEM_LABELS
from ..common import PROCESSED, get_logger, load_config, seed_all
from ..reporting import save_table, update_results
from ..sectors.classify import KNNClassifier, MultinomialNB, RocchioClassifier, scores
from ..sectors.vectors import TfidfVectorizer

log = get_logger("cause")
LABELS = ["earnings", "mna", "leadership", "financing", "restructuring_impairment", "governance", "other"]
MODEL_PATH = PROCESSED / "cause_model.pkl"


_ITEM_RE = re.compile(r"Item\s+\d\.\d{2}\.?", re.I)
_LABEL_RE = re.compile("|".join(re.escape(v) for v in ITEM_LABELS.values()), re.I)


def deleak(body: str) -> str:
    """Remove 'Item d.dd' markers and the official item-title strings: cause_label is a deterministic function
    of the items, so leaving them in would make the task trivial (and would not transfer to news)."""
    return _LABEL_RE.sub(" ", _ITEM_RE.sub(" ", body or ""))


def _fit_all(vec: TfidfVectorizer, texts: list[str], y: np.ndarray, k: int = 5) -> dict:
    C = vec.counts(texts)
    X = vec.tfidf_from_counts(C)
    return {"nb": MultinomialNB(1.0).fit(C, y), "rocchio": RocchioClassifier().fit(X, y),
            "knn": KNNClassifier(k).fit(X, y)}


def _predict(vec: TfidfVectorizer, models: dict, texts: list[str]) -> dict[str, np.ndarray]:
    C = vec.counts(texts)
    X = vec.tfidf_from_counts(C)
    return {"nb": models["nb"].predict(C), "rocchio": models["rocchio"].predict(X), "knn": models["knn"].predict(X)}


def split_8k(docs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train: <= 2021-12-31; dev: 2022-01-01..2022-06-30; test: 2022-07-01 .. end of period."""
    k = docs[(docs.doc_type == "8k") & (docs.body.fillna("").str.len() > 0)].copy()
    k["text"] = k.body.map(deleak)
    d = k.effective_day
    return k[d <= "2021-12-31"], k[(d >= "2022-01-01") & (d <= "2022-06-30")], k[d >= "2022-07-01"]


def train_and_eval() -> dict:
    cfg = load_config()
    seed_all(cfg["seed"])
    docs = pd.read_parquet(PROCESSED / "docs.parquet")
    tr, dv, te = split_8k(docs, cfg)
    log.info("8-K split: train=%d dev=%d test=%d", len(tr), len(dv), len(te))
    # dev: choose kNN k and vocabulary bigram setting
    best = None
    for ngram in (1, 2):
        vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_max=ngram).fit(tr.text.tolist())
        for k in (3, 5, 9):
            m = _fit_all(vec, tr.text.tolist(), tr.cause_label.to_numpy(), k)
            f1 = scores(dv.cause_label.to_numpy(), _predict(vec, m, dv.text.tolist())["knn"], LABELS)["macro_f1"]
            if best is None or f1 > best[0] + 1e-9:
                best = (f1, ngram, k)
    _, ngram, k = best
    log.info("dev-chosen ngram_max=%d knn k=%d (dev kNN macro-F1=%.3f)", ngram, k, best[0])
    # final: refit on train+dev, evaluate on test
    trd = pd.concat([tr, dv])
    vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_max=ngram).fit(trd.text.tolist())
    models = _fit_all(vec, trd.text.tolist(), trd.cause_label.to_numpy(), k)
    preds = _predict(vec, models, te.text.tolist())
    res = {"split": {"train": len(tr), "dev": len(dv), "test": len(te)}, "ngram_max": ngram, "knn_k": k,
           "class_counts_test": te.cause_label.value_counts().to_dict(), "models": {}}
    rows = []
    for name, p in preds.items():
        s = scores(te.cause_label.to_numpy(), p, LABELS)
        res["models"][name] = s
        rows.append({"model": name, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                     **{f"f1_{l}": s["per_class"][l]["f1"] for l in LABELS}})
    # majority-class baseline
    maj = np.array([trd.cause_label.mode()[0]] * len(te))
    s = scores(te.cause_label.to_numpy(), maj, LABELS)
    res["models"]["majority"] = s
    rows.append({"model": "majority", "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                 **{f"f1_{l}": s["per_class"][l]["f1"] for l in LABELS}})
    save_table(pd.DataFrame(rows), "cause_f1")
    update_results("cause", res)
    best_name = max(("nb", "rocchio", "knn"), key=lambda n: res["models"][n]["macro_f1"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"vec": vec, "models": models, "best": best_name}, f)
    log.info("cause test results:\n%s", pd.DataFrame(rows).round(3).to_string())
    return res


def load_classifier():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_cause(texts: list[str], bundle: dict | None = None) -> np.ndarray:
    """Predict cause labels for arbitrary texts (UI badges / reputation) using the best model."""
    b = bundle or load_classifier()
    return _predict(b["vec"], b["models"], texts)[b["best"]]


if __name__ == "__main__":
    train_and_eval()
