import { Peers as PeersT, get } from "../api";
import { useAsync } from "../hooks";
import { Status } from "../Status";
import TickerBox from "./TickerBox";

export default function Peers({ ticker, setTicker }: { ticker: string; setTicker: (t: string) => void }) {
  const p = useAsync<PeersT>(ticker ? () => get(`/api/peers?ticker=${ticker}`) : null, [ticker]);
  return (
    <div>
      <h2>Text-based peers</h2>
      <TickerBox value={ticker} onPick={setTicker} />
      <Status loading={p.loading} error={p.error} />
      {p.data && (
        <>
          <p>
            <b>{p.data.name}</b> — official GICS sector: <b>{p.data.official_sector}</b> ({p.data.sub_industry}) · text cluster{" "}
            {p.data.kmeans_cluster} · peer-sector purity {(p.data.peer_purity * 100).toFixed(0)}%
          </p>
          <table>
            <thead><tr><th>Peer</th><th>Name</th><th>Official sector</th><th>Cosine</th><th>Text vs official</th></tr></thead>
            <tbody>
              {p.data.peers.map((x) => (
                <tr key={x.ticker}>
                  <td>{x.ticker}</td><td>{x.name}</td><td>{x.sector}</td><td>{x.similarity.toFixed(3)}</td>
                  <td className={x.same_sector ? "" : "warn"}>{x.same_sector ? "same sector" : "cross-sector (hidden peer)"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
