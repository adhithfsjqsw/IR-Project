"""Build a grounded prompt from retrieved documents, call the LLM, and check the citations it produced."""
from __future__ import annotations

import re

CAUSES = ["earnings", "mna", "leadership", "financing", "restructuring_impairment", "governance", "other"]

SYSTEM = (
    "You are a financial-news analyst. You explain why a stock made an unusual move, using ONLY the numbered "
    "sources provided. Rules: (1) every factual claim must cite its source like [1] or [2][3]; (2) never use "
    "outside knowledge or invent facts, numbers or dates; (3) if the sources do not explain the move, say so "
    "plainly instead of guessing; (4) sources may be unrelated to the move, so ignore irrelevant ones; "
    "(5) be concise: at most 5 sentences. Not investment advice."
)
SYSTEM_NO_SOURCES = (
    "You are a financial analyst. Explain why a stock made an unusual move on a given day, using only what you "
    "know. If you do not know, say so. Be concise: at most 5 sentences."
)
CAUSE_INSTR = ("\n\nAfter your explanation, add a final line exactly like `CAUSE: <label>` where <label> is one of: "
               + ", ".join(CAUSES) + ", unknown. Use `unknown` if the evidence does not show the cause.")


def move_header(ticker: str, name: str, date: str, move: dict | None) -> str:
    if move:
        return (f"{name} ({ticker}) on {date}: return {move['ret']:+.2%}, market-adjusted {move['abnormal']:+.2%} "
                f"(z-score {move['z']:+.1f}).")
    return f"{name} ({ticker}) on {date}."


def build_prompt(ticker: str, name: str, date: str, move: dict | None, sources: list[dict], max_chars: int,
                 ask_cause: bool = False) -> str:
    head = move_header(ticker, name, date, move)
    tail = "Why did this stock move?" + (" Cite sources." if sources else "") + (CAUSE_INSTR if ask_cause else "")
    if not sources:
        return f"{head}\n\n{tail}"
    blocks = []
    for i, s in enumerate(sources, 1):
        kind = "SEC 8-K" if s["doc_type"] == "8k" else "news"
        blocks.append(f"[{i}] ({kind}, {s['effective_day']}) {s['title']}\n{s['text'][:max_chars]}")
    return f"{head}\n\nSources:\n\n" + "\n\n".join(blocks) + f"\n\n{tail}"


def check_citations(answer: str, n_sources: int) -> dict:
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    return {"cited": [c for c in cited if 1 <= c <= n_sources],
            "invalid": [c for c in cited if not 1 <= c <= n_sources]}


def parse_cause(answer: str) -> str | None:
    m = re.search(r"CAUSE:\s*`?([a-z_]+)", answer, re.I)
    return m.group(1).lower() if m and m.group(1).lower() in CAUSES + ["unknown"] else None


def strip_cause(answer: str) -> str:
    return re.sub(r"\n?\s*`?CAUSE:.*$", "", answer, flags=re.I | re.S).strip()


def answer_move(llm, ticker: str, name: str, date: str, move: dict | None, sources: list[dict], max_chars: int = 1200,
                ask_cause: bool = False, allow_empty: bool = False):
    if not sources and not allow_empty:
        return {"answer": "No documents were retrieved for this move, so it cannot be explained from the data.",
                "cited": [], "invalid": [], "prompt_sources": 0, "cause": None}
    prompt = build_prompt(ticker, name, date, move, sources, max_chars, ask_cause)
    raw = llm.generate(SYSTEM if sources else SYSTEM_NO_SOURCES, prompt)
    ans = strip_cause(raw) if ask_cause else raw
    return {"answer": ans, **check_citations(ans, len(sources)), "prompt_sources": len(sources),
            "cause": parse_cause(raw) if ask_cause else None}
