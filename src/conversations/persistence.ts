// Tracks whether saving conversations is failing (e.g. storage blocked or full),
// so the UI can say chats aren't being kept instead of failing silently.

let lastError: string | null = null;
const listeners = new Set<() => void>();

export function reportSaveError(err: unknown): void {
  const message = err instanceof Error ? err.message || err.name : String(err);
  console.warn('Saving conversations failed:', err);
  if (message === lastError) return;
  lastError = message;
  for (const l of listeners) l();
}

export function reportSaveOk(): void {
  if (lastError === null) return;
  lastError = null;
  for (const l of listeners) l();
}

export const saveStatus = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  /** The latest save error, or null while saving works. */
  getError: (): string | null => lastError,
};

/** Runs a write, recording success or failure. Never throws. */
export async function save(write: () => Promise<void>): Promise<void> {
  try {
    await write();
    reportSaveOk();
  } catch (err) {
    reportSaveError(err);
  }
}
