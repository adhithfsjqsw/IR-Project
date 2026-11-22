import { useState } from "react";
import Explain from "./pages/Explain";
import Peers from "./pages/Peers";
import Sectors from "./pages/Sectors";
import Evaluation from "./pages/Evaluation";
import Search from "./pages/Search";
import Shock from "./pages/Shock";
import { Status } from "./Status";

const TABS = ["Explain", "Search", "Shock", "Peers", "Sectors", "Evaluation"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Explain");
  const [ticker, setTicker] = useState("");
  return (
    <div className="app">
      <header>
        <h1>WhyMoved<small>Market-move retrieval &amp; explanation</small></h1>
        <nav>
          {TABS.map((t) => (
            <button key={t} className={t === tab ? "on" : ""} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "Explain" && <Explain ticker={ticker} setTicker={setTicker} />}
        {tab === "Search" && <Search />}
        {tab === "Shock" && <Shock />}
        {tab === "Peers" && <Peers ticker={ticker} setTicker={setTicker} />}
        {tab === "Sectors" && <Sectors />}
        {tab === "Evaluation" && <Evaluation />}
      </main>
      <footer>Not investment advice. Retrieval results are likely causes, not proven causality.</footer>
    </div>
  );
}

export { Status };
