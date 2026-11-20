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
