import numpy as np
import pandas as pd
import scipy.sparse as sp

from whymoved.shock.analogs import ShockLib
from whymoved.shock.daydocs import DayVectors
from whymoved.shock.features import FEATURES


def _lib(n=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", periods=n * 8)
    idx = np.sort(rng.choice(len(days), n, replace=False))
    sd = pd.DataFrame({"date": days[idx], "pos": idx, "episode_id": np.r_[0, np.cumsum(np.diff(idx) > 10)],
                       "fwd_ret_20": rng.normal(0, 0.05, n), "fwd_maxdd_20": -rng.random(n) * 0.1})
    for f in FEATURES:
        sd[f + "_std"] = rng.normal(size=n)
    C = sp.csr_matrix(rng.poisson(0.3, size=(len(days), 30)).astype(float))
    dv = DayVectors([d.strftime("%Y-%m-%d") for d in days], {str(i): i for i in range(30)}, C, np.ones(30),
                    np.zeros(len(days), bool))
    return ShockLib(sd, dv)


def test_library_rule_forward_window_complete_and_other_episode():
    lib = _lib()
    for q in range(lib.n):
        L = lib.library(q)
        assert (lib.pos[L] + 20 <= lib.pos[q]).all()
        assert (lib.ep[L] != lib.ep[q]).all()
        assert q not in L


def test_no_method_returns_future_or_same_episode_days():
    lib = _lib()
    q = lib.n - 1
    for m in ["rand", "recent", "era", "vol", "num", "text1", "text2", "fused", "rrf"]:
        top = lib.topk(m, q, 5)
        assert len(top) == 5 and (lib.pos[top] + 20 <= lib.pos[q]).all() and (lib.ep[top] != lib.ep[q]).all()


def test_recent_is_most_recent_eligible():
    lib = _lib()
    q = lib.n - 1
    L = lib.library(q)
    assert lib.topk("recent", q, 1)[0] == L[np.argmax(lib.pos[L])]


def test_thin_days_excluded_from_text_views():
    lib = _lib()
    lib.thin[:10] = True
    q = lib.n - 1
    assert not lib.thin[lib.topk("text2", q, 5)].any()
