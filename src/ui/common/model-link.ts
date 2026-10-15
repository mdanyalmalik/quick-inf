// Links to a model: <base>/<model id>/ opens the app with that model selected, as if it had been clicked. The
// address bar follows the selected model, so it can be copied and shared.

const BASE = import.meta.env.BASE_URL;

/** The model id a path names (`/qwen3-0.6b/`, with or without the final slash), or null. */
export function modelIdFromPath(pathname: string, base = BASE): string | null {
  if (!pathname.startsWith(base)) return null;
  const rest = pathname.slice(base.length).replace(/\/$/, '');
  if (rest === '' || rest.includes('/')) return null;
  try {
    return decodeURIComponent(rest);
  } catch {
    return null;
  }
}

/** The path for a model (the app's start page for none). */
export function modelPath(modelId: string | null, base = BASE): string {
  return modelId === null ? base : `${base}${encodeURIComponent(modelId)}/`;
}

/** Points the address bar at a model without adding a history entry; keeps any query and hash. */
export function showModelPath(modelId: string | null): void {
  const path = modelPath(modelId);
  if (location.pathname === path) return;
  history.replaceState(history.state, '', path + location.search + location.hash);
}
