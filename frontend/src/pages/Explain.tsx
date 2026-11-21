import { useState } from "react";
import { Explain as ExplainT, Move, get, post } from "../api";
import { useAsync } from "../hooks";
import { Status } from "../Status";
import TickerBox from "./TickerBox";

function Highlight({ text, terms }: { text: string; terms: string[] }) {
  if (!terms.length) return <>{text}</>;
  const esc = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const rx = new RegExp("\\b(" + esc.join("|") + ")\\b", "gi");
  return (
    <>
      {text.split(rx).map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>))}
    </>
  );
}

export default function Explain({ ticker, setTicker }: { ticker: string; setTicker: (t: string) => void }) {
  const [start, setStart] = useState("2022-01-01");
  const [end, setEnd] = useState("2023-12-29");
  const [mode, setMode] = useState<"easy" | "hard">("easy");
  const [sel, setSel] = useState<Move | null>(null);
  const [override, setOverride] = useState<ExplainT | null>(null);
  const [fbErr, setFbErr] = useState<string | null>(null);
  const moves = useAsync<Move[]>(
    ticker ? () => get(`/api/moves?ticker=${ticker}&start=${start}&end=${end}`) : null,
    [ticker, start, end],
  );
  const exp = useAsync<ExplainT>(
    sel ? () => get(`/api/explain?ticker=${sel.ticker}&date=${sel.date}&mode=${mode}`) : null,
    [sel, mode],
  );
  const shock = useAsync<{ date: string }[]>(sel ? () => get(`/api/shock/days?start=${sel.date}&end=${sel.date}`) : null, [sel]);
  const shown = override ?? exp.data;

  const moreLikeThis = async (docId: number) => {
    if (!sel) return;
    setFbErr(null);
    try {
      setOverride(await post<ExplainT>("/api/feedback", { ticker: sel.ticker, date: sel.date, mode, relevant: [docId] }));
    } catch (e) {
      setFbErr(String((e as Error).message));
    }
  };

  return (
    <div className="cols">
      <section>
        <h2>Abnormal moves</h2>
        <TickerBox value={ticker} onPick={(t) => { setTicker(t); setSel(null); setOverride(null); }} />
        <div className="row">
          <label>From <input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
          <label>To <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
        </div>
        {!ticker && <p className="muted">Pick one of the attribution tickers.</p>}
        <Status loading={moves.loading} error={moves.error} />
        {moves.data && moves.data.length === 0 && <p className="muted">No flagged moves in this range.</p>}
        {moves.data && moves.data.length > 0 && (
          <table>
            <thead><tr><th>Date</th><th>Return</th><th>z</th><th>8-K</th></tr></thead>
            <tbody>
              {moves.data.map((m) => (
                <tr key={m.date} className={sel?.date === m.date ? "sel" : ""} onClick={() => { setSel(m); setOverride(null); }}>
                  <td>{m.date}</td>
                  <td className={m.ret >= 0 ? "up" : "down"}>{(m.ret * 100).toFixed(1)}%</td>
                  <td>{m.z.toFixed(1)}</td>
                  <td>{m.has_8k ? "●" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section>
        <h2>Likely explanations {sel && <small>{sel.ticker} · {sel.date}</small>}</h2>
        {!sel && <p className="muted">Select a move.</p>}
        {sel && (
          <div className="row">
            <label>Pool{" "}
              <select value={mode} onChange={(e) => { setMode(e.target.value as "easy" | "hard"); setOverride(null); }}>
                <option value="easy">this ticker only</option>
                <option value="hard">all tickers (hard)</option>
              </select>
            </label>
            {override && <button onClick={() => setOverride(null)}>Reset feedback</button>}
          </div>
        )}
        {shock.data && shock.data.length > 0 && <p className="warn">Market-wide shock day: SPY moved abnormally on {sel?.date}. This move is likely partly market-driven; see the Shock tab for similar past days.</p>}
        <Status loading={exp.loading} error={exp.error ?? fbErr} />
        {shown?.unexplained && (
          <p className="warn">
            Heuristic flag: the top score is below the dev-set threshold (τ = {shown.tau?.toFixed(2)}); this move may have no
            public explanation in the indexed documents.
          </p>
        )}
        {shown && shown.results.length === 0 && <p className="muted">No documents matched in the candidate window.</p>}
        {shown?.results.map((r) => (
          <article key={r.doc_id} className={r.is_gold ? "gold" : ""}>
            <div className="meta">
              <span className={`badge ${r.doc_type}`}>{r.doc_type === "8k" ? "8-K" : "news"}</span>
              {r.cause_label && <span className="badge" title={r.cause_source}>{r.cause_label}</span>}
              {r.is_gold && <span className="badge goldb">gold 8-K</span>}
              <span className="muted">#{r.rank} · score {r.score.toFixed(2)} · {r.effective_day} · {r.publisher}</span>
            </div>
            <h3>{r.url ? <a href={r.url} target="_blank" rel="noreferrer">{r.title}</a> : r.title}</h3>
            <p><Highlight text={r.snippet} terms={r.matched_terms} /></p>
            <button onClick={() => moreLikeThis(r.doc_id)}>More like this</button>
          </article>
        ))}
      </section>
    </div>
  );
}
