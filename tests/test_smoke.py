"""End-to-end smoke test on SYNTHETIC data in a temp dir (checks the plumbing of every stage, not any scientific result)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["tests/smoke_fixture.py"],
    ["-m", "whymoved.ir.build"],
    ["-m", "whymoved.moves.run"],
    ["-m", "whymoved.attribution.cause"],
    ["-m", "whymoved.reputation.sources"],
    ["-m", "whymoved.attribution.run_eval"],
    ["-m", "whymoved.moves.benchmarks"],
    ["-m", "whymoved.report"],
    ["tests/smoke_api.py"],
]


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    env = {**os.environ, "WHYMOVED_DATA": str(tmp_path / "data"), "WHYMOVED_REPORTS": str(tmp_path / "reports"),
           "PYTHONPATH": str(ROOT / "src")}
    for step in STEPS:
        r = subprocess.run([sys.executable, *step], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"step {step} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    rep = (tmp_path / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert "H1" in rep and "not investment advice" in rep.lower()
    assert (tmp_path / "reports" / "results.json").exists()
