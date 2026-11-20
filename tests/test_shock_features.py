import numpy as np
import pandas as pd

from whymoved.shock.detect import detect, episodes, forward_outcomes, lagged_z
from whymoved.shock.features import FEATURES, SECTORS, Scaler, compute_features


def _market(n=700, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2005-01-03", periods=n)
    rows = []
    for t in ["SPY", "^VIX", *SECTORS]:
        c = 100 * np.cumprod(1 + rng.normal(0, 0.01, n)) if t != "^VIX" else 20 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))
        rows.append(pd.DataFrame({"date": d, "ticker": t, "close": c, "volume": rng.integers(1e6, 2e6, n).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def test_features_do_not_use_future_data():
    m = _market()
    full = compute_features(m, warmup=260)
    t = full.index[200]
    cut = compute_features(m[m.date <= t], warmup=260)
    assert np.allclose(full.loc[t, FEATURES].astype(float), cut.loc[t, FEATURES].astype(float), equal_nan=True)


def test_detection_z_excludes_current_day():
    s = pd.Series(np.r_[np.random.default_rng(0).normal(0, 0.01, 100), 0.2])
    z = lagged_z(s, 60)
    assert z.iloc[-1] > 10  # the spike is not in its own baseline
    s2 = s.copy()
    s2.iloc[-1] = 0.0
    assert np.isclose(lagged_z(s2, 60).iloc[-2], z.iloc[-2])


def test_scaler_fit_on_dev_only_and_frozen():
    dev = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    sc = Scaler().fit(dev)
    out = sc.transform(pd.DataFrame({"a": [100.0]}))
    assert np.isclose(sc.mean_["a"], 2.0) and out.a.iloc[0] > 90


def test_episode_grouping():
    pos = np.array([1, 3, 10, 30, 35, 80])
    assert episodes(pos, 10).tolist() == [0, 0, 0, 1, 1, 2]


def test_forward_outcomes():
    f = pd.DataFrame({"close": np.arange(100.0, 160.0), "vix": 20.0}, index=pd.bdate_range("2020-01-01", periods=60))
    fo = forward_outcomes(f, 20)
    assert np.isclose(fo.fwd_ret_5.iloc[0], 105 / 100 - 1) and np.isclose(fo.fwd_ret_20.iloc[0], 0.2)
    assert fo.fwd_ret_20.iloc[-1] != fo.fwd_ret_20.iloc[-1]  # NaN: incomplete window
