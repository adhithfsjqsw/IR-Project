"""Merge result fragments into reports/results.json; write CSV tables."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .common import REPORTS, ensure_dirs

RESULTS = REPORTS / "results.json"


def _clean(o: Any) -> Any:
    """Recursively make a structure strict-JSON safe: NaN/inf -> None, numpy scalars -> python."""
    import math

    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def update_results(key: str, value: Any) -> None:
    """Set results[key] = value in reports/results.json (strict JSON: NaN/inf are stored as null)."""
    ensure_dirs()
    cur = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    cur[key] = _clean(value)
    RESULTS.write_text(json.dumps(cur, indent=2, default=str, allow_nan=False), encoding="utf-8")


def read_results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def save_table(df: pd.DataFrame, name: str) -> None:
    ensure_dirs()
    df.to_csv(REPORTS / "tables" / f"{name}.csv", index=False)
