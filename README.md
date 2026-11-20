# WhyMoved

Detect abnormal stock moves, retrieve the news / SEC 8-K documents that most likely explain them, and evaluate the
retrieval against 8-K filings as an independent answer key. A second module clusters S&P 500 companies by the text of
their business descriptions and uses text-based peers as a benchmark for "abnormal". Built for a university IR course;
the core IR components are implemented from scratch (see `src/whymoved/ir`, `src/whymoved/sectors`).

*Not investment advice.*

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
export EDGAR_USER_AGENT="Your Name your.email@example.com"       # required by the SEC; never committed
```

Python 3.14 was used (the spec says 3.11; no 3.14-only features are used — see `DECISIONS.md`).
Node 18+ is needed only for the UI.

## Run

| Step | Command | What it does | Approx. runtime* |
|---|---|---|---|
| tests | `make test` | 34 tests: IR core, classifiers, k-means, PageRank/HITS vs networkx, calendar, market model, plus a 1-minute SYNTHETIC end-to-end smoke test (temp dir; never touches real outputs) | ~1 min |
| data | `make data` | companies, descriptions, prices, FNSPID (≈23 GB download), EDGAR 8-Ks, docs, gold | 40–90 min first run, then seconds |
| index | `make index` | positional inverted index → `data/processed/index.pkl` | 1–2 min |
| moves | `make moves` | rolling market model, flagged moves, sanity checks | seconds |
| eval | `make eval` | H1: dev tuning + ablation table with bootstrap CIs | 5–15 min |
| classify | `make classify` | cause classifier (NB / Rocchio / kNN) | < 1 min |
| sectors | `make sectors` | clusters vs GICS, classification CV, hidden peers, PageRank/HITS, figures | 1–3 min |
| integrate | `make integrate` | H2: M0/M1/M2 benchmark comparison | 1–2 min |
| report | `make report` | `reports/RESULTS.md`, `reports/REPORT.md` | seconds |
| api | `make api` | FastAPI on :8000 (`/docs` for the schema) | |
| ui | `make ui` | Vite dev server on :5173 | |

\*Machine-dependent; measured on a laptop with ~20 MB/s download. Everything is cached under `data/cache`; re-runs do not
re-download. All randomness uses seed 42. On Windows without `make`, run the `python -m ...` commands from the Makefile
with `PYTHONPATH=src`.

Order matters: `data → index → moves → eval` and `classify` / `sectors` before `integrate` (H2 needs text peers).
`reputation` (A6 ablation): `python -m whymoved.reputation.sources` after `classify`, before `eval`.

## Environment variables

- `EDGAR_USER_AGENT` (required for 8-K collection): e.g. `"Jane Doe jane@example.com"`.
- `VITE_API` (optional, UI): API base URL, default `http://localhost:8000`.

## Repo map

```
config.yaml            all thresholds / seeds / periods
DECISIONS.md PLAN.md   decisions + deviations, plan status
src/whymoved/
  data/        sp500, descriptions, prices, fnspid(+download), edgar, calendar (effective-day rule), docs (docs+gold), run
  ir/          tokenize, index (positional), scoring (TF-IDF, BM25), tolerant (k-gram, Levenshtein, Soundex, wildcard),
               query (direction expansion), feedback (Rocchio), search, build
  moves/       market_model (rolling OLS), detect, benchmarks (H2), run
  attribution/ candidates, gold, evaluate, ablations, run_eval (H1), metrics (bootstrap), cause (classifier)
  sectors/     vectors (TF-IDF), kmeans, hierarchical, classify (NB/Rocchio/kNN), hidden_peers, graph (PageRank/HITS), run
  reputation/  sources
  api/         FastAPI app + schemas
frontend/      Vite + React + TypeScript UI (Explain, Peers, Sectors, Evaluation)
reports/       results.json, RESULTS.md, REPORT.md, figures/, tables/
tests/         pytest suite
```
