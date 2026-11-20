"""Shared helpers: config, paths, seeding, logging."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW, CACHE, INTERIM, PROCESSED = (DATA / n for n in ("raw", "cache", "interim", "processed"))
REPORTS = ROOT / "reports"


def load_config() -> dict[str, Any]:
    """Read config.yaml from the repo root."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_all(seed: int = 42) -> None:
    """Seed python and numpy RNGs."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def ensure_dirs() -> None:
    for d in (RAW, CACHE, INTERIM, PROCESSED, REPORTS / "figures", REPORTS / "tables"):
        d.mkdir(parents=True, exist_ok=True)
