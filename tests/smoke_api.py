"""Exercise the API against the synthetic fixture (run by test_smoke.py in a subprocess)."""
import pandas as pd
from fastapi.testclient import TestClient

from whymoved.api.main import app

c = TestClient(app)
r = c.get("/api/entities", params={"q": "Alfa"})
assert r.status_code == 200 and r.json(), r.text
t = r.json()[0]["ticker"]
mv = c.get("/api/moves", params={"ticker": "TE0"}).json()
assert mv, "no moves"
m = next((x for x in mv if x["has_8k"]), mv[0])
e = c.get("/api/explain", params={"ticker": m["ticker"], "date": m["date"], "mode": "easy"})
assert e.status_code == 200, e.text
body = e.json()
assert body["results"] and body["results"][0]["rank"] == 1
f = c.post("/api/feedback", json={"ticker": m["ticker"], "date": m["date"], "relevant": [body["results"][0]["doc_id"]]})
assert f.status_code == 200 and f.json()["results"]
h = c.get("/api/explain", params={"ticker": m["ticker"], "date": m["date"], "mode": "hard"})
assert h.status_code == 200
assert c.get("/api/peers", params={"ticker": "TE0"}).json()["peers"]
assert c.get("/api/sectors/clusters").json()
assert c.get("/api/eval/summary").json().get("h1")
assert c.get("/api/explain", params={"ticker": "NOPE", "date": m["date"]}).status_code == 404
print("api smoke ok")
