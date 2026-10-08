// Cross-tab change notices for conversations. Each tab keeps an in-memory copy
// of what it shows; when another tab writes, it says what changed and this tab
// re-reads that part from IndexedDB.

export type SyncNotice = { kind: 'sessions' } | { kind: 'messages'; sessionId: string };

const channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel('quick-inf:conversations');

export function broadcast(notice: SyncNotice): void {
  channel?.postMessage(notice);
}

export function onBroadcast(handler: (notice: SyncNotice) => void): void {
  channel?.addEventListener('message', (e: MessageEvent<SyncNotice>) => handler(e.data));
}
