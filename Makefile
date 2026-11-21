PY=.venv/Scripts/python.exe
export PYTHONPATH=src
setup:
	python -m venv .venv && $(PY) -m pip install -r requirements.txt
data:
	$(PY) -m whymoved.data.run
index:
	$(PY) -m whymoved.ir.build
moves:
	$(PY) -m whymoved.moves.run
eval:
	$(PY) -m whymoved.attribution.run_eval
classify:
	$(PY) -m whymoved.attribution.cause
sectors:
	$(PY) -m whymoved.sectors.run
integrate:
	$(PY) -m whymoved.moves.benchmarks
report:
	$(PY) -m whymoved.report
api:
	$(PY) -m uvicorn whymoved.api.main:app --port 8000
ui:
	cd frontend && npm install && npm run dev
test:
	$(PY) -m pytest -q
all: data index moves eval classify sectors integrate report

# ---- add-on targets ----
search_bench:
	$(PY) -m whymoved.ir.search_bench
shock_data:
	$(PY) -m whymoved.shock.data && $(PY) -m whymoved.shock.detect && $(PY) -m whymoved.shock.daydocs
shock_eval:
	$(PY) -m whymoved.shock.evaluate
shock_regimes:
	$(PY) -m whymoved.shock.regimes
peer_events:
	@echo "Phase 13 (peer events) is not implemented yet: it needs the base 8-K data (EDGAR_USER_AGENT)." && exit 1
all_addon: search_bench shock_data shock_eval shock_regimes report
