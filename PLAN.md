# PLAN

| Phase | Content | Status |
|---|---|---|
| 1 | Data: companies, descriptions, prices, FNSPID news, EDGAR 8-K, docs, gold | companies/descriptions/prices/news DONE; **8-K step blocked on `EDGAR_USER_AGENT`** |
| 2 | IR core + unit tests | done, tests pass |
| 3 | Moves + H1 ablations | moves DONE (425 flagged); H1 code smoke-tested, awaiting 8-K gold |
| 4 | Cause classifier | code smoke-tested, awaiting 8-K texts |
| 5 | Sector module | DONE, results + figures saved |
| 6 | H2 integration + reputation | code smoke-tested, awaiting 8-K targets |
| 7 | FastAPI + React UI | built; API smoke-tested on synthetic fixture; UI builds; real-data check pending |
| 8 | Report + handover | generator done (partial real results); final run pending |

Cut order (if over budget): reputation, centrality UI panels, H2, React UI -> static HTML, 25 tickers.

## Add-on status
| Phase | Status |
|---|---|
| 9 Search hardening (tolerant bench, efficiency, VByte) | DONE; `/api/search` written but untested on real docs (needs base index) |
| 10 Shock data/detection/day docs | DONE (187 shock days, 98 episodes, 13/14 curated hit) |
| 11 Analogs H3/H5 | DONE, null results, era-confounded text |
| 12 Regimes/archetypes/H4 | DONE, H4 not supported |
| 13 Peer events (H6) | NOT DONE: needs base 8-K data + all-constituent news/8-K; `/api/peers/events` not implemented |
| 14 API + UI | shock + search endpoints and pages done; peer-event section and search feedback buttons missing |
| 15 Report | RESULTS.md/REPORT.md extended for 9-12 |
