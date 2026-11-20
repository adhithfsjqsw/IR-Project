"""Gap encoding + variable-byte (VByte) compression of postings lists."""
from __future__ import annotations

from typing import Iterable

from .index import PositionalIndex


def vb_encode_number(n: int) -> bytes:
    """7 data bits per byte, high bit set on the LAST byte of a number."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(n & 0x7F)
        n >>= 7
    out.reverse()
    out[-1] |= 0x80
    return bytes(out)


def vb_encode(nums: Iterable[int]) -> bytes:
    return b"".join(vb_encode_number(n) for n in nums)


def vb_decode(data: bytes) -> list[int]:
    out, n = [], 0
    for b in data:
        if b & 0x80:
            out.append((n << 7) | (b & 0x7F))
            n = 0
        else:
            n = (n << 7) | b
    return out


def gaps(sorted_ids: list[int]) -> list[int]:
    return [sorted_ids[0]] + [b - a for a, b in zip(sorted_ids, sorted_ids[1:])] if sorted_ids else []


def ungaps(g: list[int]) -> list[int]:
    out, acc = [], 0
    for x in g:
        acc += x
        out.append(acc)
    return out


def compress_postings(plist: dict[int, list[int]]) -> bytes:
    """Layout per list: n_docs, then per doc: doc-gap, n_positions, position gaps (all VByte)."""
    docs = sorted(plist)
    parts = [vb_encode_number(len(docs))]
    for dg, d in zip(gaps(docs), docs):
        pos = plist[d]
        parts.append(vb_encode_number(dg) + vb_encode_number(len(pos)) + vb_encode(gaps(pos)))
    return b"".join(parts)


def decompress_postings(data: bytes) -> dict[int, list[int]]:
    nums = vb_decode(data)
    n, i, doc, out = nums[0], 1, 0, {}
    for _ in range(n):
        doc += nums[i]
        m = nums[i + 1]
        out[doc] = ungaps(nums[i + 2:i + 2 + m])
        i += 2 + m
    return out


def compression_report(idx: PositionalIndex) -> dict:
    """Round-trip every postings list; ratio vs raw 32-bit ints (doc ids + counts + positions)."""
    raw = comp = 0
    for plist in idx.postings.values():
        blob = compress_postings(plist)
        assert decompress_postings(blob) == {d: list(p) for d, p in plist.items()}, "round-trip mismatch"
        raw += 4 * (1 + sum(2 + len(p) for p in plist.values()))
        comp += len(blob)
    # doc-id-only view (what a non-positional index would store)
    raw_ids = sum(4 * len(p) for p in idx.postings.values())
    comp_ids = sum(len(vb_encode(gaps(sorted(p)))) for p in idx.postings.values())
    return {"n_lists": len(idx.postings), "positional_raw_bytes": raw, "positional_vbyte_bytes": comp,
            "positional_ratio": raw / max(1, comp), "docid_raw_bytes": raw_ids, "docid_vbyte_bytes": comp_ids,
            "docid_ratio": raw_ids / max(1, comp_ids), "all_lists_roundtrip_ok": True}
