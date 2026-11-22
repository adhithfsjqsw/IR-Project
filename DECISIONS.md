# DECISIONS

Every non-obvious decision, with the reason. Newest at the bottom of each section.

## Environment
- **Python 3.14.3, not 3.11.** Only 3.14 is installed on the build machine; all dependencies installed cleanly
  on it (see `requirements.txt`). No code depends on 3.14-specific features.
- Bare `python` is a Microsoft Store stub on this machine; the Makefile uses `.venv/Scripts/python.exe`.

## Data
- **FNSPID file choice.** The spec expects "the news file(s) under Stock_news/". Repo `Zihan1004/FNSPID` has two:
  `All_external.csv` (5.7 GB) and `nasdaq_exteral_data.csv` (23 GB). I downloaded `All_external.csv` first
  and verified by scanning all 13.1 M rows that it **ends in mid-2020** (year histogram in the log), so it cannot
  cover the 2021–2023 period. I deleted it and use `nasdaq_exteral_data.csv`, which has the same expected columns
  (plus a leading `Unnamed: 0` index), real UTC timestamps with time of day, and article text through Dec 2023.
  The reduced repo was *not* needed, and has date-only timestamps for many rows (which would break the
  before/after-16:00 effective-day rule).
- **Publisher is blank in the Nasdaq file.** When `Publisher` is empty the publisher is set to the URL's domain
  (almost always `nasdaq.com`). Consequence: the source-reputation module has little publisher diversity
  and is reported as such (limitation: FNSPID comes from very few sources).
- **Ticker selection (40).** Round-robin by within-sector rank of 2021–2023 news count, max 4 per sector,
  top 40 overall by (rank, count). Written to `data/interim/attribution_tickers.json`.
- **8-K text.** `text` = item labels + first 3000 chars after the first `Item d.dd`; `body` keeps the same
  text *without* the label prefix (used by the cause classifier).
- **EDGAR needs `EDGAR_USER_AGENT`.** It must be a real declared contact (SEC policy). It is read only from the
  environment and never invented or hard-coded; the pipeline stops with a clear message if unset.
- Prices: yfinance `auto_adjust=True` close; chunks of 50; tickers silently dropped by a chunk are retried one by
  one. Tickers with no data in the window (e.g. recently listed names) are reported in the log.

## IR
- Phrase queries run on the stop-word-stripped token stream, so "interest in the rate" matches the phrase
  "lending rate" style queries by adjacency of surviving tokens. This is a deliberate, documented simplification.
- sklearn's English stop-word list contains "interest" (and e.g. "bill", "fire", "system"); we keep the list
  as the spec mandates.
- BM25 `idf = ln(1 + (N-df+.5)/(df+.5))` (never negative). Unit tests inject this idf into `rank_bm25` so
  the comparison tests the scoring code, not an idf variant.
- TF-IDF uses log10 everywhere (spec). sklearn's sublinear tf uses ln, so the TF-IDF test builds the reference by hand
  from sklearn `CountVectorizer` counts.

## Moves / evaluation
- Estimation window = 120 trading days whose last day is t-11. sigma uses ddof = n_params.
- Relevance for H1 is *strict*: only the 8-K(s) effective on day t for that ticker are relevant.
- Documents that never match any query term are not in the ranking (reciprocal rank 0), a conservative choice.
- Hyper-parameters (direction weight, Rocchio alpha/beta/t, time-prior lambda) are tuned **sequentially on DEV
  (2021) MRR in hard mode**; a component is switched off if it does not beat the previous stage on DEV.
- "Unexplained" threshold tau: 10th percentile of top-1 scores of gold-covered DEV moves (easy mode, final system).
  Heuristic only.

## Cause classifier
- Because `cause_label` is a deterministic function of the filing's item numbers, the classifier text has
  every `Item d.dd` marker and the official item titles removed (otherwise F1 would be trivially ~1 and the model
  would not transfer to news). Split: train <= 2021, dev = 2022 H1, test = 2022 H2 onward.

## Sector module
- TF-IDF vectoriser is implemented from scratch (project tokenizer, 1–2 grams, min_df 2, max_df 0.5, sublinear).
- Classification CV fits the vectoriser inside each training fold.

## Data findings logged during the build
- **pandas empty-frame bug (fixed, guarded).** Assigning a longer Series to an *empty* DataFrame makes pandas adopt that
  Series' index, which produced phantom NaN-ticker rows with out-of-period dates in the first FNSPID filter. The
  filter now masks the Series with the same mask, skips empty chunks, and `build_news_all` asserts that every kept row has a valid
  ticker and an in-period timestamp.
- **Uneven FNSPID coverage.** After filtering, the news counts show no 2021 news for AAPL, MSFT, TSLA and none before 2023 for AMZN
  (see `data/` log, "news counts per ticker per year"). The ticker-selection rule (highest news count in the period) is kept as
  specified, so DEV tuning has less news for those tickers; their 8-Ks (once collected) are still in the pool. Reported as a limitation.
- Prices: 8 current constituents have no price history in the window (recent listings: FDXF, GEV, HONA, Q, RDDT, SNDK, SOLV, VYLR);
  they are excluded from H2 sector means and from text-peer means automatically (NaN-aware).
- Moves: market-model z over 30,120 ticker-days has mean -0.006 and std 1.049; 425 moves flagged (120 DEV, 305 TEST).

## Add-on (shock mode, peers, search hardening)
- **Ordering deviation.** The add-on says to start after base gates 1-6 pass. Those gates are blocked on `EDGAR_USER_AGENT`
  (not set in this session). Add-on phases 9.1-9.3, 10, 11, 12 need only prices + FNSPID headlines, so they are built now;
  Phase 13 (peer events) and the 8-K parts of 9.4 wait for the base gate.
- Sector ETFs: XLRE and XLC are ignored (short history). Shock module has its own DEV (<=2014) / TEST (2015+) split.
- **Shock detector** thresholds are the config defaults (no adjustment needed: 187 days). 2022-06-13 is missed by the lagged-60-day z (2022 volatility was already high); not tuned to force a hit.
- **Fusion weight** tuned on DEV came out as w = 1.0 (numeric only), so `fused` equals `num` on TEST. Regime clustering therefore uses a fixed 0.5 text weight, stated in results.
- **Era/thin text:** headline volume is ~35/day in 1999 and thousands/day around 2011 (`reports/tables/headline_coverage.csv`); only ~3% of 2023 and a few 1999-2000 days are `thin_text`.
- **Search feedback buttons** are not on the Search page (the base `/api/feedback` is explain-specific); shock analogs do have explicit Rocchio feedback.
- **Bug fixed:** bootstrap p-values could exceed 1 when many resampled differences are exactly 0; now capped at 1 everywhere.
- Shock analogs are only available for detected shock days (404 otherwise), not arbitrary dates.

## Post-evaluation additions (RAG, baseline diagnosis, extras)
- **Easy-mode "random beats BM25" is not a bug.** Exact expected random MRR is H_n/n averaged over pools: 0.251 in easy mode
  (median pool 26 docs), measured 0.27. Inside a ticker-filtered pool a company-name query carries no information, so BM25
  name-only ranks by name repetition, which favours news over formal 8-Ks (gold 8-Ks match 59% of query terms, other pool docs 72%).
  The report now prints the analytic expectation next to the A0 row.
- **A0..A6 are unchanged** (re-run reproduced them exactly). New variants E1-E3 are tuned on DEV only and reported separately:
  E1 event-lexicon expansion, E3 small-m Rocchio, E2 an 8-K document-type prior.
- **E1/E2 are benchmark-aware.** The gold is always an 8-K, so features that identify 8-K-style text raise MRR without showing
  better explanatory retrieval (hard-mode top-1 is an 8-K for 2% of moves under A5 but 30% under E1). E2 reaches MRR 1.000 in easy
  mode, which is the clearest sign of this. Treated as diagnostics; the headline H1 comparison stays A5 vs A2.
- **Title-field boosting was not implemented:** gold 8-K titles are item codes ("8-K 2.02,9.01"), so it can only help news.
- **RAG** (`src/whymoved/rag`, `GET /api/rag`): BM25 top-k -> local Ollama (qwen2.5:7b) with a cite-only-the-sources prompt.
  Evaluated by `python -m whymoved.rag.evaluate` against the 8-K-derived cause label (conditions: none / news / all). The gold
  label is ~87-93% "earnings", so overall accuracy is read against the majority-class baseline and split by earnings vs non-earnings.
- **Not done:** Phase 13 (peer events, H6) needs all-constituent news/8-K data.

## Retrieval-improvement round (judged relevance)
- **Why:** the 8-K-only key marks one document per move, so news about the same event counts as a miss, and 8-K-style features
  (E1/E2) inflate it. `attribution/judged.py` judges EVERY document in an easy-mode pool with the local LLM (complete qrels,
  no pooling bias) and evaluates any system instantly from `reports/judgments.jsonl`. The judge is an LLM and unverified by a human:
  it called 83% of the known-relevant gold 8-Ks relevant and 0% of random other-ticker documents (n=30 each); a spot-check
  sample is in `reports/tables/judge_review_sample.csv`.
- **Findings (27 test moves with a relevant doc):** A5 beats BM25 name-only (nDCG@10 0.656 vs 0.541, p=0.006) and random (0.471);
  ~88% of judged-relevant documents are not the 8-K. So A5 is above random under judged relevance, unlike under the 8-K-only
  easy-mode metric.
- **Tried and NOT adopted:** (1) burst-term queries (`ir/burst.py`): no gain on DEV or TEST (0.648 vs 0.656, p=0.85);
  (2) re-tuned BM25 k1/b/lambda: +0.023 on DEV but -0.007 on TEST (overfit to 20 dev moves); (3) E1 event lexicon: +0.057, not
  significant, i.e. most of its 8-K-benchmark gain does not transfer. Defaults and the API are unchanged.
- **Limits:** easy mode only (hard mode would need judging other tickers' pools); 27 test moves is small; the time prior is the
  only component that clearly helps.
