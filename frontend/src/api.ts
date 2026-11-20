export const API: string = (import.meta as any).env?.VITE_API ?? "http://localhost:8000";

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export interface Entity { ticker: string; name: string; sector: string; score: number }
export interface Move { ticker: string; date: string; ret: number; abnormal: number; z: number; direction: number; has_8k: boolean }
export interface Explanation {
  doc_id: number; rank: number; score: number; doc_type: string; publisher: string | null; effective_day: string;
  title: string; snippet: string; cause_label: string | null; cause_source: string; matched_terms: string[];
  is_gold: boolean; url: string | null;
}
export interface Explain {
  ticker: string; date: string; mode: string; unexplained: boolean; tau: number | null;
  results: Explanation[]; query_terms: Record<string, number>;
}
export interface Peer { ticker: string; name: string; sector: string; similarity: number; same_sector: boolean }
export interface Peers {
  ticker: string; name: string; official_sector: string; sub_industry: string; kmeans_cluster: number;
  peer_purity: number; peers: Peer[];
}
export interface ClusterPt {
  ticker: string; name: string; gics_sector: string; kmeans_cluster: number; hier_cluster: number; x: number; y: number;
}
