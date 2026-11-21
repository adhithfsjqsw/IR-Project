import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get } from "../api";
import { useAsync } from "../hooks";
import { Status } from "../Status";

// results.json is schemaless by design (written by the pipeline), so rows are loosely typed.
type Row = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

function withCi(rows: Row[], key: string) {
  return rows.map((r) => ({
    name: String(r.system ?? r.model).slice(0, 26),
    v: r[key],
    err: [r[key] - r[key + "_lo"], r[key + "_hi"] - r[key]],
  }));
}

function CiChart({ rows, keyName, title }: { rows: Row[]; keyName: string; title: string }) {
  return (
    <div>
      <h4>{title}</h4>
      <div style={{ height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={withCi(rows, keyName)} margin={{ bottom: 70 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-35} textAnchor="end" interval={0} fontSize={10} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="v" fill="#2563eb"><ErrorBar dataKey="err" width={4} /></Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Table({ rows, cols }: { rows: Row[]; cols: string[] }) {
  const f = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : String(v ?? ""));
  return (
    <table>
      <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{f(r[c])}</td>)}</tr>)}</tbody>
    </table>
  );
}

const ABL = ["system", "n_moves", "rr", "recall@1", "recall@5", "recall@10", "ndcg10", "mrr_diff_vs_A2", "p_vs_A2"];

export default function Evaluation() {
  const s = useAsync<Row>(() => get("/api/eval/summary"), []);
  const d = s.data;
  const sh = useAsync<Row>(() => get("/api/eval/shock"), []);
  const e = sh.data;
  return (
    <div>
      <h2>Evaluation (TEST 2022-2023, tuned on DEV 2021)</h2>
      <Status loading={s.loading} error={s.error} />
      {d?.h1 && (
        <>
          <h3>H1 — retrieval ablations (hard mode: all tickers, no ticker filter)</h3>
          <p className="muted">
            Flagged test moves {d.h1.counts.test_flagged}; with a gold 8-K {d.h1.counts.test_with_gold}; without {d.h1.counts.test_without_gold}.
            Error bars: 95% bootstrap CI over moves. "rr" is the reciprocal rank (its mean is MRR).
          </p>
          <div className="cols">
            <CiChart rows={d.h1.ablation_hard} keyName="rr" title="MRR" />
            <CiChart rows={d.h1.ablation_hard} keyName="recall@10" title="Recall@10" />
          </div>
          <Table rows={d.h1.ablation_hard} cols={ABL} />
          <h3>H1 — easy mode (ticker filter)</h3>
          <Table rows={d.h1.ablation_easy} cols={ABL} />
        </>
      )}
      {d?.h2 && (
        <>
          <h3>H2 — benchmark for abnormal returns (|z| scoring "an 8-K is effective")</h3>
          <p className="muted">{d.h2.n_days} ticker-days, {d.h2.n_positive} positives (base rate {(d.h2.base_rate * 100).toFixed(1)}%).</p>
          <div className="cols">
            <CiChart rows={d.h2.models} keyName="roc_auc" title="ROC-AUC" />
            <CiChart rows={d.h2.models} keyName="pr_auc" title="PR-AUC" />
          </div>
          <Table rows={d.h2.models} cols={["model", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "pr_auc_lo", "pr_auc_hi", "top200_share_with_8k"]} />
        </>
      )}
      {d?.cause && (
        <>
          <h3>Cause classifier (test)</h3>
          <Table
            rows={Object.entries(d.cause.models).map(([k, v]: [string, any]) => ({ model: k, accuracy: v.accuracy, macro_f1: v.macro_f1 }))}
            cols={["model", "accuracy", "macro_f1"]}
          />
        </>
      )}
      {d?.sectors && (
        <>
          <h3>Sector clustering vs GICS sector</h3>
          <Table
            rows={[{ method: "k-means k=11", ...d.sectors.kmeans_k11.sector }, { method: "hierarchical k=11", ...d.sectors.hierarchical_k11.sector }]}
            cols={["method", "nmi", "ari", "purity"]}
          />
          <h4>GICS sector classification from descriptions (5-fold CV)</h4>
          <Table
            rows={Object.entries(d.sectors.sector_classification).map(([k, v]: [string, any]) => ({ model: k, accuracy: v.accuracy, macro_f1: v.macro_f1 }))}
            cols={["model", "accuracy", "macro_f1"]}
          />
        </>
      )}
      <h2>Add-on evaluation (shock mode and search)</h2>
      <Status loading={sh.loading} error={sh.error} />
      {e?.shock_analogs && (<>
        <h3>H3 / H5 — analog retrieval (TEST 2015-2023, k=5, non-thin subset)</h3>
        <p className="muted">{e.shock_analogs.subsets?.non_thin_all_methods?.n_queries} queries from {e.shock_analogs.subsets?.non_thin_all_methods?.n_episodes} independent episodes. 95% CIs by episode-block bootstrap.</p>
        <Table rows={e.shock_analogs.table.filter((r: Row) => r.k === 5 && r.subset === "non_thin_all_methods")}
          cols={["method", "p_at_k", "p_at_k_lo", "p_at_k_hi", "chance", "mae_ret", "spearman_ret", "p_at_k_diff_vs_vol", "p_at_k_p_vs_vol", "mean_year_gap", "share_same_era_pm2y"]} />
      </>)}
      {e?.shock_regimes?.h4 && (<>
        <h3>H4 — text to regime classification</h3>
        <Table rows={Object.entries(e.shock_regimes.h4.models).map(([m, v]: [string, any]) => ({ model: m, accuracy: v.accuracy, macro_f1: v.macro_f1, p_vs_majority: v.p_vs_majority, p_vs_era_only: v.p_vs_era_only }))}
          cols={["model", "accuracy", "macro_f1", "p_vs_majority", "p_vs_era_only"]} />
      </>)}
      {e?.search_bench_tolerant && (<>
        <h3>Tolerant retrieval benchmark</h3>
        <Table rows={e.search_bench_tolerant.table.filter((r: Row) => r.kind === "all")} cols={["method", "n", "acc@1", "acc@3", "mrr", "median_latency_ms"]} />
      </>)}
    </div>
  );
}
