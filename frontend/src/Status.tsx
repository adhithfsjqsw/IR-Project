/** Loading / error indicator used by every data-driven view. */
export function Status({ loading, error }: { loading: boolean; error: string | null }) {
  if (loading) return <p className="muted">Loading…</p>;
  if (error) return <p className="err">Error: {error}</p>;
  return null;
}
