"""Exercise the shock router on REAL shock artifacts (needs data/processed/shock_*.parquet and results.json)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from whymoved.api.shock import router

app = FastAPI()
app.include_router(router)
c = TestClient(app)
days = c.get("/api/shock/days", params={"start": "2020-01-01", "end": "2020-12-31"}).json()
assert days and days[0]["date"].startswith("2020")
d = next(x["date"] for x in days if x["date"] == "2020-03-16")
day = c.get("/api/shock/day", params={"date": d}).json()
assert day["features_raw"] and day["sector_returns"] and day["headlines"] is not None
for v in ("numeric", "text", "burst", "fused"):
    a = c.get("/api/shock/analogs", params={"date": d, "view": v, "k": 5}).json()
    assert len(a["analogs"]) == 5 and a["caution"].startswith("Analogs are descriptive"), v
    q = d
    assert all(x["date"] < q for x in a["analogs"]), "analog from the future"
    assert len(a["fan"]["median"]) == 26
fb = c.post("/api/shock/feedback", json={"date": d, "view": "fused", "relevant": [a["analogs"][0]["date"]], "nonrelevant": [a["analogs"][1]["date"]]})
assert fb.status_code == 200 and len(fb.json()["analogs"]) == 5
assert c.get("/api/shock/analogs", params={"date": "2020-03-17"}).status_code == 404
assert c.get("/api/shock/archetypes").json()
assert c.get("/api/shock/clusters").json()["k"]
assert c.get("/api/eval/shock").json()["shock_analogs"]
assert c.get("/api/shock/timeline").json()
print("shock api smoke ok")
