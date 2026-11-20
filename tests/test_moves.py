import numpy as np
import pandas as pd

from whymoved.attribution.candidates import PRICE_MOVE_RE
from whymoved.attribution.evaluate import Evaluator, System
from whymoved.attribution.metrics import bootstrap_ci, paired_bootstrap_p, rank_metrics
from whymoved.ir.index import PositionalIndex
from whymoved.moves.detect import flag_moves, model_panel
from whymoved.moves.market_model import rolling_abnormal


def _returns(n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, n)
    r = 0.0002 + 1.2 * m + rng.normal(0, 0.008, n)
    return m, r


def test_rolling_ols_recovers_beta_and_z_is_standard():
    m, r = _returns(1500)
    out = rolling_abnormal(r, m[:, None], window=120, gap=11, min_obs=100)
    ok = out.dropna()
    assert abs(ok.beta0.mean() - 1.2) < 0.05
    assert abs(ok.z.mean()) < 0.1 and 0.9 < ok.z.std() < 1.1
    assert out.iloc[:120].z.isna().all()  # not enough history yet


def test_window_excludes_last_gap_days_and_event_day():
    m, r = _returns(300)
    base = rolling_abnormal(r, m[:, None])
    r2 = r.copy()
    r2[250:] += 0.5  # shock inside the gap before day 255 and the event itself must not move day 255's fit
    shocked = rolling_abnormal(r2, m[:, None])
    # day 255 window ends at day 244 -> the coefficients are identical
    assert np.isclose(base.alpha[255], shocked.alpha[255]) and np.isclose(base.sigma[255], shocked.sigma[255])


def test_flagging_requires_z_and_min_return():
    m, r = _returns(400)
    r[350] += 0.10
    rets = pd.DataFrame({"SPY": m, "AAA": r}, index=pd.bdate_range("2021-01-01", periods=400))
    panel = model_panel(rets, ["AAA"], None, 120, 11, 100)
    f = flag_moves(panel, 3.0, 0.02)
    assert rets.index[350] in set(f.date) and (f.direction[f.date == rets.index[350]] == 1).all()
    assert (f.ret.abs() >= 0.02).all() and (f.z.abs() >= 3).all()


def test_rank_metrics():
    r = rank_metrics([5, 3, 9], {9})
    assert r["rr"] == 1 / 3 and r["recall@1"] == 0 and r["recall@5"] == 1
    assert np.isclose(r["ndcg10"], 1 / np.log2(4))
    assert rank_metrics([1], {2})["rr"] == 0


def test_bootstrap():
    x = np.r_[np.ones(50), np.zeros(50)]
    mean, lo, hi = bootstrap_ci(x, 500)
    assert lo < 0.5 < hi and mean == 0.5
    d, p = paired_bootstrap_p(np.ones(50), np.zeros(50), 500)
    assert d == 1.0 and p < 0.01
    assert paired_bootstrap_p(x, x)[1] > 0.5


def test_price_report_regex():
    assert PRICE_MOVE_RE.search("Apple shares jump after earnings")
    assert PRICE_MOVE_RE.search("Stock falls 5% on guidance")
    assert not PRICE_MOVE_RE.search("Apple announces new CEO")


def test_evaluator_end_to_end_on_toy_index():
    idx = PositionalIndex()
    docs = [("AAA", "8k", "2021-01-05", "Acme Corp announces results of operations record quarterly revenue"),
            ("AAA", "news", "2021-01-05", "Weather today is sunny"),
            ("BBB", "8k", "2021-01-05", "Beta Inc announces departure of chief executive"),
            ("AAA", "news", "2021-01-01", "Acme Corp mentioned in unrelated story")]
    for t, ty, d, txt in docs:
        idx.add_document(txt, {"ticker": t, "doc_type": ty, "effective_day": d, "title": txt})
    idx.finalize()
    cal = ["2021-01-01", "2021-01-04", "2021-01-05"]
    ev = Evaluator(idx, {"AAA": "Acme Corp", "BBB": "Beta Inc"}, {("AAA", "2021-01-05"): {0}}, cal)
    mv = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2021-01-05")], "direction": [1]})
    for mode in ("hard", "easy"):
        res = ev.run(mv, System("A2"), mode)
        assert len(res) == 1 and res.rr.iloc[0] == 1.0
    assert len(ev.run(mv.assign(ticker="BBB"), System("A2"), "hard")) == 0  # no gold -> skipped


def test_paired_bootstrap_p_never_exceeds_one():
    x = np.zeros(40)
    assert paired_bootstrap_p(x, x)[1] <= 1.0
