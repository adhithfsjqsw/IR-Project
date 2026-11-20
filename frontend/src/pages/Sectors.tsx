import { useState } from "react";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { API, ClusterPt, get } from "../api";
import { useAsync } from "../hooks";
import { Status } from "../Status";

const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#4b5563", "#ea580c", "#0d9488", "#9333ea"];
interface HP { ticker: string; name: string; sector: string; purity: number; why: string }
interface Cent { ticker: string; name: string; sector: string; pagerank: number; hub: number; authority: number }

export default function Sectors() {
  const [by, setBy] = useState<"gics_sector" | "kmeans_cluster">("gics_sector");
  const pts = useAsync<ClusterPt[]>(() => get("/api/sectors/clusters"), []);
  const hp = useAsync<HP[]>(() => get("/api/sectors/hidden_peers"), []);
  const cent = useAsync<Cent[]>(() => get("/api/graph/centrality"), []);
  const groups = pts.data ? [...new Set(pts.data.map((p) => String(p[by])))].sort() : [];
  return (
    <div>
      <h2>Text clusters vs GICS</h2>
      <div className="row">
        <label><input type="radio" checked={by === "gics_sector"} onChange={() => setBy("gics_sector")} /> colour by GICS sector</label>
        <label><input type="radio" checked={by === "kmeans_cluster"} onChange={() => setBy("kmeans_cluster")} /> colour by text cluster (k=11)</label>
      </div>
      <Status loading={pts.loading} error={pts.error} />
      {pts.data && (
        <div style={{ height: 460 }}>
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid />
              <XAxis type="number" dataKey="x" name="SVD1" />
              <YAxis type="number" dataKey="y" name="SVD2" />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={({ payload }) => {
                  const d = payload?.[0]?.payload as ClusterPt | undefined;
                  return d ? <div className="tip">{d.ticker} – {d.name}<br />{d.gics_sector}</div> : null;
                }}
              />
              <Legend />
              {groups.map((g, i) => (
                <Scatter key={g} name={g.slice(0, 22)} data={pts.data!.filter((p) => String(p[by]) === g)} fill={COLORS[i % COLORS.length]} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
      <h3>Dendrogram (average linkage, cosine; readable subset)</h3>
      <img className="fig" src={`${API}/api/figures/dendrogram.png`} alt="dendrogram" />
      <h3>Hidden peers (lowest peer-sector purity)</h3>
      <Status loading={hp.loading} error={hp.error} />
      {hp.data && (
        <table>
          <thead><tr><th>Ticker</th><th>Sector</th><th>Purity</th><th>Why (shared terms with cross-sector neighbours)</th></tr></thead>
          <tbody>
            {hp.data.map((h) => (
              <tr key={h.ticker}><td>{h.ticker}</td><td>{h.sector}</td><td>{(h.purity * 100).toFixed(0)}%</td><td>{h.why || "—"}</td></tr>
            ))}
          </tbody>
        </table>
      )}
      <h3>Graph centrality (directed kNN graph)</h3>
      <Status loading={cent.loading} error={cent.error} />
      {cent.data && (
        <div className="cols3">
          {(["pagerank", "hub", "authority"] as const).map((k) => (
            <table key={k}>
              <thead><tr><th colSpan={2}>Top 20 by {k}</th></tr></thead>
              <tbody>
                {[...cent.data!].sort((a, b) => b[k] - a[k]).slice(0, 20).map((c) => (
                  <tr key={c.ticker}><td>{c.ticker}</td><td>{c[k].toFixed(4)}</td></tr>
                ))}
              </tbody>
            </table>
          ))}
        </div>
      )}
    </div>
  );
}
