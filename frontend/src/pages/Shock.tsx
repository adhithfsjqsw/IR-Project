import { useState } from "react";
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { get, post } from "../api";
import { useAsync } from "../hooks";
import { Status } from "../Status";

interface Pt { date: string; close: number }
interface Day { date: string; r_spy: number; z_spy: number; direction: number; episode_id: number; thin_text: boolean; cluster: number | null }
interface DayInfo {
  date: string; flags: { z_spy: number; vix_z: number | null }; features_raw: Record<string, number>; sector_returns: Record<string, number>;
  headlines: string[]; top_terms_t1: string[]; top_terms_t2: string[]; thin_text: boolean; n_headlines: number; cluster: number | null;
  outcomes: Record<string, number | null>;
}
interface Analog {
  date: string; score: number; r_spy: number; year: number; thin_text: boolean; headlines: string[];
  view_scores: Record<string, number>; outcomes: Record<string, number>; path: { offset: number; value: number }[];
}
interface AnalogResp {
  library_size: number; n_independent_episodes: number; caution: string; analogs: Analog[];
  query_path: { offset: number; value: number }[]; fan: { offsets: number[]; q25: number[]; median: number[]; q75: number[] };
}

const pct = (x: number | null | undefined) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
const heat = (r: number) => { const a = Math.min(1, Math.abs(r) / 0.04); return r < 0 ? `rgba(220,38,38,${a})` : `rgba(22,163,74,${a})`; };

export default function Shock() {
  const [date, setDate] = useState<string | null>(null);
  const [view, setView] = useState("fused");
  const [k, setK] = useState(5);
  const [fb, setFb] = useState<{ rel: string[]; non: string[] }>({ rel: [], non: [] });
  const [fbRes, setFbRes] = useState<AnalogResp | null>(null);
  const [fbErr, setFbErr] = useState<string | null>(null);
  const line = useAsync<Pt[]>(() => get("/api/shock/timeline?step=3"), []);
  const days = useAsync<Day[]>(() => get("/api/shock/days"), []);
  const info = useAsync<DayInfo>(date ? () => get(`/api/shock/day?date=${date}`) : null, [date]);
  const an = useAsync<AnalogResp>(date ? () => get(`/api/shock/analogs?date=${date}&view=${view}&k=${k}`) : null, [date, view, k]);
  const shown = fbRes ?? an.data;
  const closeAt = new Map((line.data ?? []).map((p) => [p.date, p.close]));
  const marks = (days.data ?? []).map((d) => ({ ...d, close: closeAt.get(d.date) ?? [...closeAt.entries()].find(([x]) => x >= d.date)?.[1] })).filter((d) => d.close);
  const pick = (d: string) => { setDate(d); setFbRes(null); setFb({ rel: [], non: [] }); };
  const send = async (rel: string[], non: string[]) => {
    setFb({ rel, non }); setFbErr(null);
    try { setFbRes(await post<AnalogResp>("/api/shock/feedback", { date, view, k, relevant: rel, nonrelevant: non })); } catch (e) { setFbErr(String((e as Error).message)); }
  };
  const pathData = shown ? shown.query_path.map((p, i) => ({ offset: p.offset, query: p.value, ...Object.fromEntries(shown.analogs.map((a, j) => [`a${j}`, a.path[i]?.value])) })) : [];
  const fanData = shown ? shown.fan.offsets.map((o, i) => ({ offset: o, q25: shown.fan.q25[i], median: shown.fan.median[i], q75: shown.fan.q75[i], query: shown.query_path[i]?.value })) : [];
  return (
    <div>
      <h2>Shock mode</h2>
      <p className="warn">Analogs are descriptive, not a forecast. Few independent episodes; text similarity is partly driven by era (see Evaluation).</p>
      <Status loading={line.loading || days.loading} error={line.error ?? days.error} />
      {line.data && (
        <div style={{ height: 280 }}><ResponsiveContainer>
          <ComposedChart data={line.data}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" tickFormatter={(d) => d.slice(0, 4)} minTickGap={50} />
            <YAxis scale="log" domain={["auto", "auto"]} /><Tooltip /><Legend />
            <Line dataKey="close" dot={false} stroke="#2563eb" name="SPY (adj. close)" />
            <Scatter data={marks} dataKey="close" name="shock day (click)" fill="#dc2626" onClick={(p: { date: string }) => pick(p.date)} />
          </ComposedChart></ResponsiveContainer></div>)}
      {days.data && !date && <p className="muted">{days.data.length} shock days detected. Click a red marker, or pick a date: {" "}
        <select onChange={(e) => e.target.value && pick(e.target.value)} defaultValue=""><option value="">…</option>
          {days.data.map((d) => <option key={d.date} value={d.date}>{d.date} ({pct(d.r_spy)})</option>)}</select></p>}
      {date && <>
        <h3>{date}</h3>
        <Status loading={info.loading} error={info.error} />
        {info.data && <>
          <div className="row">{["r_spy", "drawdown", "rv20", "vix_d1"].map((f) => <span key={f} className="badge">{f}: {info.data!.features_raw[f]?.toFixed(3)}</span>)}
            <span className="badge">VIX {Math.exp(info.data.features_raw.log_vix).toFixed(1)}</span>
            <span className="badge">z_spy {info.data.flags.z_spy.toFixed(1)}</span><span className="badge">breadth {info.data.features_raw.breadth.toFixed(2)}</span></div>
          <div className="row">{Object.entries(info.data.sector_returns).map(([s, r]) => <span key={s} className="badge" style={{ background: heat(r), color: "#111" }}>{s} {pct(r)}</span>)}</div>
          {info.data.thin_text ? <p className="muted">Text view unavailable (fewer than 20 headlines).</p> :
            <details><summary>{info.data.n_headlines} headlines; top terms</summary>
              <p><b>T1:</b> {info.data.top_terms_t1.join(", ")}<br /><b>T2 (burst):</b> {info.data.top_terms_t2.join(", ")}</p>
              <ul>{info.data.headlines.map((h, i) => <li key={i}>{h}</li>)}</ul></details>}
        </>}
        <div className="row">
          {["numeric", "text", "burst", "fused"].map((v) => <button key={v} className={v === view ? "on" : ""} style={v === view ? { borderColor: "var(--accent)" } : {}} onClick={() => { setView(v); setFbRes(null); }}>{v}</button>)}
          <label>k <select value={k} onChange={(e) => { setK(+e.target.value); setFbRes(null); }}>{[3, 5, 10].map((x) => <option key={x}>{x}</option>)}</select></label>
          {fbRes && <button onClick={() => { setFbRes(null); setFb({ rel: [], non: [] }); }}>Reset feedback</button>}
        </div>
        <Status loading={an.loading} error={an.error ?? fbErr} />
        {shown && <>
          <p className="muted">Library: {shown.library_size} eligible past shock days (forward window complete, different episode). Top-{shown.analogs.length} analogs span {shown.n_independent_episodes} independent episodes.</p>
          <div className="cols">
            <div><h4>Normalised SPY paths (t−5…t+20)</h4><div style={{ height: 260 }}><ResponsiveContainer><ComposedChart data={pathData}>
              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="offset" /><YAxis domain={["auto", "auto"]} /><Tooltip />
              {shown.analogs.map((a, j) => <Line key={a.date} dataKey={`a${j}`} dot={false} stroke="#94a3b8" name={a.date} />)}
              <Line dataKey="query" dot={false} stroke="#dc2626" strokeWidth={2.5} name="query day" /></ComposedChart></ResponsiveContainer></div></div>
            <div><h4>Analog fan (median, quartiles)</h4><div style={{ height: 260 }}><ResponsiveContainer><ComposedChart data={fanData}>
              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="offset" /><YAxis domain={["auto", "auto"]} /><Tooltip /><Legend />
              <Line dataKey="q25" dot={false} stroke="#94a3b8" strokeDasharray="4 3" /><Line dataKey="q75" dot={false} stroke="#94a3b8" strokeDasharray="4 3" />
              <Line dataKey="median" dot={false} stroke="#2563eb" strokeWidth={2} /><Line dataKey="query" dot={false} stroke="#dc2626" name="query day" /></ComposedChart></ResponsiveContainer></div></div>
          </div>
          {shown.analogs.map((a) => (
            <article key={a.date}>
              <div className="meta"><b>{a.date}</b> <span className={a.r_spy < 0 ? "down" : "up"}>SPY {pct(a.r_spy)}</span>
                <span className="badge">+20d {pct(a.outcomes.fwd_ret_20)}</span><span className="badge">max DD {pct(a.outcomes.fwd_maxdd_20)}</span>
                <span className="muted"> sim {a.score.toFixed(2)} · numeric {a.view_scores.numeric.toFixed(2)} · text {a.view_scores.text.toFixed(2)} · burst {a.view_scores.burst.toFixed(2)}</span></div>
              {a.thin_text ? <p className="muted">no text view</p> : <ul>{a.headlines.slice(0, 3).map((h, i) => <li key={i}>{h}</li>)}</ul>}
              <button onClick={() => send([...fb.rel, a.date], fb.non)}>Relevant ({fb.rel.includes(a.date) ? "✓" : "+"})</button>{" "}
              <button onClick={() => send(fb.rel, [...fb.non, a.date])}>Not relevant ({fb.non.includes(a.date) ? "✓" : "+"})</button>
            </article>))}
        </>}
      </>}
    </div>
  );
}
