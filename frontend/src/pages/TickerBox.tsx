import { useState } from "react";
import { Entity, get } from "../api";
import { useAsync } from "../hooks";

/** Typo-tolerant ticker box backed by /api/entities (k-gram + Levenshtein + Soundex + wildcards). */
export default function TickerBox({ value, onPick }: { value: string; onPick: (t: string) => void }) {
  const [q, setQ] = useState(value);
  const res = useAsync<Entity[]>(q.trim().length > 1 ? () => get(`/api/entities?q=${encodeURIComponent(q)}`) : null, [q]);
  return (
    <div className="tickerbox">
      <input value={q} placeholder="Company or ticker (typos and NVD* work)" onChange={(e) => setQ(e.target.value)} />
      {res.loading && <span className="muted"> …</span>}
      {res.error && <span className="err"> {res.error}</span>}
      {res.data && (
        <ul>
          {res.data.map((e) => (
            <li key={e.ticker}>
              <button onClick={() => { onPick(e.ticker); setQ(e.ticker); }}>
                <b>{e.ticker}</b> {e.name} <span className="muted">{e.sector} · {e.score.toFixed(2)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
