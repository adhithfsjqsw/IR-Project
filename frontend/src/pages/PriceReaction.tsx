import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get } from "../api";
import { useAsync } from "../hooks";

interface Point { date: string; rel: number; ret: number | null; market_ret: number | null; abnormal: number | null; is_event: boolean }
interface Price {
  ticker: string; event_day: string; market: string; event_ret: number | null; event_abnormal: number | null;
  after_3d_abnormal: number | null; flagged_move: boolean; points: Point[];
}

const pct = (x: number | null, d = 1) => (x == null ? "n/a" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`);
const cls = (x: number | null) => (x == null ? "muted" : x >= 0 ? "up" : "down");

/** The ticker's closing price 5 trading days either side of the news day, rebased to 100 the day before. */
export default function PriceReaction({ ticker, day }: { ticker: string; day: string }) {
  const res = useAsync<Price>(() => get(`/api/price?ticker=${ticker}&date=${day}`), [ticker, day]);
  if (res.loading) return <p className="muted">Loading price…</p>;
  if (res.error || !res.data) return <p className="muted">No price data for {ticker} on {day}.</p>;
  const d = res.data;
  return (
    <div className="pricebox">
      <div className="pricestats">
        <span>{d.ticker} on {d.event_day}: <b className={cls(d.event_ret)}>{pct(d.event_ret)}</b></span>
        <span>vs {d.market}: <b className={cls(d.event_abnormal)}>{pct(d.event_abnormal)}</b></span>
        <span>3-day vs {d.market}: <b className={cls(d.after_3d_abnormal)}>{pct(d.after_3d_abnormal)}</b></span>
        {d.flagged_move && <span className="badge goldb">flagged abnormal move</span>}
      </div>
      <div style={{ height: 150 }}>
        <ResponsiveContainer>
          <LineChart data={d.points} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
            <XAxis dataKey="date" tickFormatter={(v: string) => v.slice(5)} tick={{ fontSize: 11 }} stroke="var(--muted)" />
            <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} width={40} stroke="var(--muted)" />
            <Tooltip formatter={(v: number) => v.toFixed(2)} labelFormatter={(l: string) => l} />
            <ReferenceLine x={d.event_day} stroke="var(--accent)" strokeDasharray="4 3" label={{ value: "news", fontSize: 11, fill: "var(--accent)", position: "top" }} />
            <Line type="monotone" dataKey="rel" name="price (100 = day before)" stroke="var(--accent)" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="muted" style={{ fontSize: 12, margin: "2px 0 0" }}>
        Market-adjusted = stock return minus {d.market}. Timing fits the news only if the ticker tag is right and the story broke that day; this is correlation, not proof of cause.
      </p>
    </div>
  );
}
