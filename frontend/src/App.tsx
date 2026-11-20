import { useState } from "react";
import Explain from "./pages/Explain";
import Peers from "./pages/Peers";
import Sectors from "./pages/Sectors";
import Evaluation from "./pages/Evaluation";
import { Status } from "./Status";

const TABS = ["Explain", "Peers", "Sectors", "Evaluation"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Explain");
  const [ticker, setTicker] = useState("");
  return (
    <div className="app">
      <header>
        <h1>WhyMoved</h1>
        <nav>
          {TABS.map((t) => (
            <button key={t} className={t === tab ? "on" : ""} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "Explain" && <Explain ticker={ticker} setTicker={setTicker} />}
        {tab === "Peers" && <Peers ticker={ticker} setTicker={setTicker} />}
        {tab === "Sectors" && <Sectors />}
        {tab === "Evaluation" && <Evaluation />}
      </main>
      <footer>Not investment advice. Retrieval results are likely causes, not proven causality.</footer>
    </div>
  );
}

export { Status };
