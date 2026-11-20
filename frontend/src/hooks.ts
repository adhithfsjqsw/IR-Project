import { useEffect, useState } from "react";

export interface Async<T> { data: T | null; loading: boolean; error: string | null }

/** Runs `fn` whenever `deps` change and exposes loading / error states. Pass null to stay idle. */
export function useAsync<T>(fn: (() => Promise<T>) | null, deps: unknown[]): Async<T> {
  const [s, set] = useState<Async<T>>({ data: null, loading: !!fn, error: null });
  useEffect(() => {
    if (!fn) {
      set({ data: null, loading: false, error: null });
      return;
    }
    let live = true;
    set((p) => ({ ...p, loading: true, error: null }));
    fn()
      .then((d) => live && set({ data: d, loading: false, error: null }))
      .catch((e) => live && set({ data: null, loading: false, error: String(e.message ?? e) }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}
