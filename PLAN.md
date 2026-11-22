# PLAN

Status verified by a full re-run (13 pipeline steps, all exit 0) and the test suite (62+ tests) on 2026-11-21.

| Phase | Content | Status |
|---|---|---|
| 1 | Data: companies, descriptions, prices, FNSPID news, EDGAR 8-K, docs, gold | DONE (1,575 gold ticker-days) |
| 2 | IR core + unit tests | DONE, tests pass |
| 3 | Moves + H1 ablations | DONE (425 flagged moves; H1 supported, see RESULTS.md) |
| 4 | Cause classifier | DONE |
| 5 | Sector module | DONE |
| 6 | H2 integration + reputation | DONE (M2 > M0, M2 ~ M1) |
| 7 | FastAPI + React UI | DONE; API endpoints checked on real data; UI builds (not exercised in a browser) |
| 8 | Report + handover | DONE (`make report`) |

## Add-on status
| Phase | Status |
|---|---|
| 9 Search hardening (tolerant bench, efficiency, VByte) | DONE; `/api/search` checked on real data |
| 10 Shock data/detection/day docs | DONE |
| 11 Analogs H3/H5 | DONE, null results, era-confounded text |
| 12 Regimes/archetypes/H4 | DONE, H4 not supported |
| 13 Peer events (H6) | NOT DONE: needs all-constituent news/8-K; `/api/peers/events` not implemented |
| 14 API + UI | shock + search endpoints and pages done; peer-event section and search feedback buttons missing |
| 15 Report | RESULTS.md/REPORT.md cover 9-12 |
| 16 RAG | DONE: `src/whymoved/rag`, `GET /api/rag`, UI panel, evaluation in `reports/rag_eval.json` |
