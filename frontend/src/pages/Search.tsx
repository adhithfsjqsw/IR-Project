import { useState } from "react";
import { get } from "../api";
import { useAsync } from "../hooks";
import { Status } from "../Status";

interface Hit {
  doc_id: number; rank: number; score: number; contributions: Record<string, number>; doc_type: string;
  publisher: string | null; ticker: string; effective_day: string; title: string; snippet: string;
  matched_terms: string[]; cause_label: string | null; url: string | null;
}
interface Resp { query: string; n_candidates: number; expanded_wildcards: Record<string, string[]>; did_you_mean: string[]; results: Hit[] }

function Highlight({ text, terms }: { text: string; terms: string[] }) {
  if (!terms.length) return <>{text}</>;
  const esc = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const rx = new RegExp("\\b(" + esc.join("|") + ")\\b", "gi");
  return <>{text.split(rx).map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>))}</>;
}

export default function Search() {
  const [q, setQ] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [mode, setMode] = useState("bm25");
  const [ticker, setTicker] = useState("");
  const [docType, setDocType] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [open, setOpen] = useState<number | null>(null);
  const params = new URLSearchParams({ q: submitted, mode, k: "20" });
  if (ticker) params.set("ticker", ticker.toUpperCase());
  if (docType) params.set("doc_type", docType);
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  const res = useAsync<Resp>(submitted ? () => get(`/api/search?${params.toString()}`) : null, [submitted, mode, ticker, docType, start, end]);
  return (
    <div>
      <h2>News search</h2>
      <form onSubmit={(e) => { e.preventDefault(); setSubmitted(q.trim()); }} className="row">
        <input style={{ flex: 1, minWidth: 260 }} value={q} onChange={(e) => setQ(e.target.value)}
          placeholder='e.g.  "interest rate" AND fed*   or   lawsuit NOT apple' />
        <button type="submit">Search</button>
      </form>
      <p className="muted">Syntax: plain terms, "exact phrase", AND / OR / NOT, parentheses, wildcard terms such as <code>nvd*</code> (matched against stemmed index terms).</p>
      <div className="row">
        <label>Scoring <select value={mode} onChange={(e) => setMode(e.target.value)}><option value="bm25">BM25</option><option value="tfidf">TF-IDF cosine</option></select></label>
        <label>Ticker <input size={6} value={ticker} onChange={(e) => setTicker(e.target.value)} /></label>
        <label>Type <select value={docType} onChange={(e) => setDocType(e.target.value)}><option value="">any</option><option value="news">news</option><option value="8k">8-K</option></select></label>
        <label>From <input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
        <label>To <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
      </div>
      <Status loading={res.loading} error={res.error} />
      {res.data && <>
        <p className="muted">{res.data.n_candidates} candidate documents.</p>
        {res.data.did_you_mean.length > 0 && <p>Did you mean: {res.data.did_you_mean.map((d) => <button key={d} onClick={() => { setTicker(d.split(" ")[0]); }}>{d}</button>)}</p>}
        {Object.entries(res.data.expanded_wildcards).map(([k, v]) => <p key={k} className="muted">{k} → {v.slice(0, 12).join(", ") || "no matches"}</p>)}
        {res.data.results.length === 0 && <p className="muted">No results.</p>}
        {res.data.results.map((h) => (
          <article key={h.doc_id}>
            <div className="meta"><span className={`badge ${h.doc_type}`}>{h.doc_type === "8k" ? "8-K" : "news"}</span>
              {h.cause_label && <span className="badge">{h.cause_label}</span>}
              <span className="muted">#{h.rank} · {h.score.toFixed(3)} · {h.ticker} · {h.effective_day} · {h.publisher}</span></div>
            <h3>{h.url ? <a href={h.url} target="_blank" rel="noreferrer">{h.title}</a> : h.title}</h3>
            <p><Highlight text={h.snippet} terms={h.matched_terms} /></p>
            <button onClick={() => setOpen(open === h.doc_id ? null : h.doc_id)}>{open === h.doc_id ? "Hide" : "Per-term scores"}</button>
            {open === h.doc_id && <table><tbody>{Object.entries(h.contributions).sort((a, b) => b[1] - a[1]).map(([t, v]) => <tr key={t}><td>{t}</td><td>{v.toFixed(3)}</td></tr>)}</tbody></table>}
          </article>))}
      </>}
    </div>
  );
}
