import numpy as np
import scipy.sparse as sp

from whymoved.shock.daydocs import DayVectors


def _dv(counts):
    days = [f"2020-01-{i + 1:02d}" for i in range(len(counts))]
    C = sp.csr_matrix(np.array(counts, float))
    return DayVectors(days, {"a": 0, "b": 1}, C, np.ones(C.shape[1]), np.zeros(len(days), bool), baseline=3)


def test_burst_baseline_excludes_day_t():
    # term a is steady (1/day) then spikes on the last day; term b steady
    dv = _dv([[1, 1]] * 6 + [[50, 1]])
    base_changed = _dv([[1, 1]] * 6 + [[500, 1]])
    v, v2 = dv.t2([dv.days[-1]]), base_changed.t2([dv.days[-1]])
    assert v[0, 0] > v[0, 1]  # burst on a
    # changing day t's own count must not alter the baseline: ratio a/b scales ~ linearly with the count
    assert v2[0, 0] / v2[0, 1] > 5 * v[0, 0] / v[0, 1] * 0.9


def test_burst_downweights_persistent_terms_vs_t1():
    dv = _dv([[30, 1]] * 6 + [[30, 5]])
    t1, t2 = dv.t1([dv.days[-1]]), dv.t2([dv.days[-1]])
    assert t2[0, 1] / t2[0, 0] > t1[0, 1] / t1[0, 0]


def test_vectors_unit_norm():
    dv = _dv([[1, 2], [3, 0], [0, 4], [1, 1]])
    for X in (dv.t1(dv.days[:3]), dv.t2(dv.days[:3])):
        assert np.allclose(np.asarray(X.multiply(X).sum(axis=1)).ravel(), 1.0)
