export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 KB';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

/** Rough remaining-time text, e.g. "about 2 min left". */
export function formatRemaining(seconds: number): string {
  if (seconds < 10) return 'a few seconds left';
  if (seconds < 60) return `about ${Math.round(seconds / 5) * 5} s left`;
  return `about ${Math.round(seconds / 60)} min left`;
}

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}
