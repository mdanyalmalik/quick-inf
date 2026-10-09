// Whether chats have changed since the last backup file was saved (see backup.ts),
// for the "unsaved chats" indicators.
//
// Kept in localStorage, which every tab shares: a change in one tab shows as
// unsaved in the others. Only additions and edits
// count as changes; deleting chats can't lose anything a backup holds.

const CHANGED_KEY = 'quick-inf:chats-changed-at';
const BACKED_UP_KEY = 'quick-inf:chats-backed-up-at';

export interface BackupStatus {
  /** When the last backup file was saved, if ever. */
  backedUpAt: number | null;
  /** Chats were added or edited after the last backup (or there's never been one). */
  unsaved: boolean;
}

const listeners = new Set<() => void>();
let status: BackupStatus = read();

function storage(): Storage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null; // blocked (e.g. some private modes)
  }
}

function readNumber(key: string): number | null {
  const value = Number(storage()?.getItem(key) ?? NaN);
  return Number.isFinite(value) ? value : null;
}

function read(): BackupStatus {
  const changedAt = readNumber(CHANGED_KEY);
  const backedUpAt = readNumber(BACKED_UP_KEY);
  return { backedUpAt, unsaved: changedAt !== null && (backedUpAt === null || changedAt > backedUpAt) };
}

function refresh(): void {
  const next = read();
  if (next.backedUpAt === status.backedUpAt && next.unsaved === status.unsaved) return;
  status = next;
  for (const l of listeners) l();
}

function write(key: string, value: number): void {
  try {
    storage()?.setItem(key, String(value));
  } catch {
    // Storage full or blocked: the indicators just won't track this change.
  }
  refresh();
}

if (typeof window !== 'undefined') {
  // Another tab changed chats or saved a backup.
  window.addEventListener('storage', (e) => {
    if (e.key === CHANGED_KEY || e.key === BACKED_UP_KEY || e.key === null) refresh();
  });
}

export const backupStatus = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  get: (): BackupStatus => status,
};

/** Records that chats were added or edited (call when the change happens, before it's written). */
export function markChatsChanged(): void {
  // Strictly later than a backup saved in the same millisecond.
  write(CHANGED_KEY, Math.max(Date.now(), (readNumber(BACKED_UP_KEY) ?? 0) + 1));
}

/** Records that a backup holding every chat as of `at` was saved. */
export function markBackedUp(at: number): void {
  write(BACKED_UP_KEY, at);
}

/**
 * Chats saved before change tracking existed (or in a browser where it was lost)
 * have no record of changes; count them as not backed up.
 */
export function noteExistingChats(count: number): void {
  if (count > 0 && readNumber(CHANGED_KEY) === null && readNumber(BACKED_UP_KEY) === null) markChatsChanged();
}
