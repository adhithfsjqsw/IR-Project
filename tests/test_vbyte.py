import numpy as np

from whymoved.ir.index import PositionalIndex
from whymoved.ir.vbyte import compress_postings, compression_report, decompress_postings, gaps, ungaps, vb_decode, vb_encode


def test_vbyte_roundtrip_random_lists():
    rng = np.random.default_rng(0)
    for _ in range(200):
        nums = rng.integers(0, 2**rng.integers(1, 32), size=rng.integers(0, 50)).tolist()
        assert vb_decode(vb_encode(nums)) == nums
    assert vb_decode(vb_encode([0, 127, 128, 16383, 16384, 2**31])) == [0, 127, 128, 16383, 16384, 2**31]


def test_gap_roundtrip_and_postings():
    ids = sorted(np.random.default_rng(1).choice(10**6, 500, replace=False).tolist())
    assert ungaps(gaps(ids)) == ids
    pl = {3: [1, 5, 9], 10: [0], 4000: [2, 3]}
    assert decompress_postings(compress_postings(pl)) == pl


def test_index_compression_roundtrip_and_ratio():
    idx = PositionalIndex()
    rng = np.random.default_rng(2)
    for _ in range(300):
        idx.add_document(" ".join(rng.choice([f"w{i}" for i in range(50)], 40)))
    r = compression_report(idx)
    assert r["all_lists_roundtrip_ok"] and r["positional_ratio"] > 1.5
