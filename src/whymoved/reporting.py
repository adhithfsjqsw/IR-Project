"""Merge result fragments into reports/results.json; write CSV tables."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .common import REPORTS, ensure_dirs

RESULTS = REPORTS / "results.json"


def update_results(key: str, value: Any) -> None:
    """Set results[key] = value (JSON-serialisable) in reports/results.json."""
    ensure_dirs()
    cur = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    cur[key] = value
    RESULTS.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")


def read_results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def save_table(df: pd.DataFrame, name: str) -> None:
    ensure_dirs()
    df.to_csv(REPORTS / "tables" / f"{name}.csv", index=False)
