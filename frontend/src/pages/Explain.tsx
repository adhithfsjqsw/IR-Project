import { useState, useEffect } from "react";
import { Explain as ExplainT, Move, Rag, get, post } from "../api";
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

/** Renders the LLM answer with [n] citation markers linked to the numbered sources below it. */
function CitedAnswer({ text, n }: { text: string; n: number }) {
  return (
    <p>
      {text.split(/(\[\d+\])/g).map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (!m) return <span key={i}>{p}</span>;
        const k = Number(m[1]);
        return k >= 1 && k <= n ? <a key={i} href={`#rag-src-${k}`}>{p}</a> : <span key={i} className="err" title="no such source">{p}</span>;
      })}
    </p>
  );
}

export default function Explain({ ticker, setTicker }: { ticker: string; setTicker: (t: string) => void }) {
  const [ragOn, setRagOn] = useState(false);
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
  // open the largest move automatically so the explanations appear right after a ticker is chosen
  useEffect(() => {
    if (!sel && moves.data && moves.data.length > 0) setSel([...moves.data].sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moves.data]);
  const shown = override ?? exp.data;
  // on demand only: one call takes tens of seconds on a local model
  const rag = useAsync<Rag>(ragOn && sel ? () => get(`/api/rag?ticker=${sel.ticker}&date=${sel.date}&mode=${mode}`) : null, [ragOn, sel, mode]);

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
                <tr key={m.date} className={sel?.date === m.date ? "sel" : ""} onClick={() => { setSel(m); setOverride(null); setRagOn(false); }}>
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
              <select value={mode} onChange={(e) => { setMode(e.target.value as "easy" | "hard"); setOverride(null); setRagOn(false); }}>
                <option value="easy">this ticker only</option>
                <option value="hard">all tickers (hard)</option>
              </select>
            </label>
            {override && <button onClick={() => setOverride(null)}>Reset feedback</button>}
          </div>
        )}
        {sel && (
          <div className="rag">
            <div className="row">
              <button onClick={() => setRagOn(true)} disabled={rag.loading}>
                {rag.loading ? "Generating…" : rag.data ? "Regenerate AI explanation" : "Generate AI explanation"}
              </button>
              <span className="muted">Local LLM over the top retrieved documents; can take up to a minute.</span>
            </div>
            <Status loading={rag.loading} error={rag.error} />
            {rag.data && (
              <>
                <CitedAnswer text={rag.data.answer} n={rag.data.sources.length} />
                {rag.data.invalid_citations.length > 0 && (
                  <p className="err">Warning: the model cited sources that do not exist ({rag.data.invalid_citations.join(", ")}).</p>
                )}
                {rag.data.cited.length === 0 && <p className="warn">The answer cites no source; treat it as unsupported.</p>}
                <p className="muted">Generated by {rag.data.model}; not investment advice. Verify against the sources:</p>
                <ol className="ragsrc">
                  {rag.data.sources.map((r, i) => (
                    <li key={r.doc_id} id={`rag-src-${i + 1}`} className={rag.data!.cited.includes(i + 1) ? "cited" : ""}>
                      <span className={`badge ${r.doc_type}`}>{r.doc_type === "8k" ? "8-K" : "news"}</span>
                      {r.url ? <a href={r.url} target="_blank" rel="noreferrer">{r.title}</a> : r.title}{" "}
                      <span className="muted">· {r.effective_day}</span>
                    </li>
                  ))}
                </ol>
              </>
            )}
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
