// Chat messages, kept in memory per open session and saved to IndexedDB.
//
// - A session's messages load the first time it's opened (`load`), not at startup.
// - New and finished messages are written at once. A streaming reply changes on
//   every token, so it is saved at most once per CHECKPOINT_MS instead.
// - A tab generating a reply holds a Web Lock for that session. A saved reply
//   still marked "streaming" whose lock is free was cut off (tab closed,
//   crash): on load it is marked interrupted, keeping the text saved so far.

import { dbDeleteMany, dbGetAllByIndex, dbPut, dbPutMany, sessionRange } from '../store/db';
import { markChatsChanged } from './backup-status';
import { save } from './persistence';
import type { MessageRecord } from './records';
import { broadcast, onBroadcast } from './sync';

const CHECKPOINT_MS = 1000;

export type MessageDraft = Pick<MessageRecord, 'id' | 'role' | 'content'> & Partial<MessageRecord>;

/** Lock held while a tab is generating a reply in this session. */
export function generationLockName(sessionId: string): string {
  return `quick-inf:generating:${sessionId}`;
}

class MessageStore {
  private readonly bySession = new Map<string, MessageRecord[]>();
  private readonly loads = new Map<string, Promise<MessageRecord[]>>();
  /** Pending checkpoint per message id. */
  private readonly timers = new Map<string, { sessionId: string; timer: ReturnType<typeof setTimeout> }>();
  private readonly listeners = new Set<() => void>();

  constructor() {
    onBroadcast((notice) => {
      // Another tab changed this session: re-read it if we're showing it and not writing to it ourselves.
      if (notice.kind === 'messages' && this.bySession.has(notice.sessionId) && !this.hasPending(notice.sessionId)) {
        void this.read(notice.sessionId).then((records) => this.set(notice.sessionId, records));
      }
    });
    if (typeof window !== 'undefined') window.addEventListener('pagehide', () => this.flush());
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** The session's messages in order, or undefined until loaded. */
  get(sessionId: string): MessageRecord[] | undefined {
    return this.bySession.get(sessionId);
  }

  /** Loads a session's messages from IndexedDB (once). */
  load(sessionId: string): Promise<MessageRecord[]> {
    const cached = this.bySession.get(sessionId);
    if (cached) return Promise.resolve(cached);
    let loading = this.loads.get(sessionId);
    if (!loading) {
      loading = this.read(sessionId)
        .then((records) => {
          // A message appended while loading wins over the stored copy.
          const local = this.bySession.get(sessionId) ?? [];
          const merged = [...records.filter((r) => !local.some((m) => m.id === r.id)), ...local];
          this.set(sessionId, merged);
          return merged;
        })
        .finally(() => this.loads.delete(sessionId));
      this.loads.set(sessionId, loading);
    }
    return loading;
  }

  /** Adds messages at the end of a session and saves them. Load the session first. */
  append(sessionId: string, drafts: MessageDraft[]): MessageRecord[] {
    const existing = this.bySession.get(sessionId) ?? [];
    let seq = existing.at(-1)?.seq ?? 0;
    const now = Date.now();
    const added = drafts.map((d) => ({ ...d, sessionId, seq: ++seq, createdAt: now, updatedAt: now }));
    this.set(sessionId, [...existing, ...added]);
    markChatsChanged();
    void save(() => dbPutMany('messages', added)).then(() => broadcast({ kind: 'messages', sessionId }));
    return added;
  }

  /**
   * Changes one message. `when: 'now'` saves immediately (finished replies,
   * errors); `'checkpoint'` saves within CHECKPOINT_MS (streaming text).
   */
  patch(
    sessionId: string,
    id: string,
    change: (m: MessageRecord) => Partial<MessageRecord>,
    when: 'now' | 'checkpoint' = 'now',
  ): void {
    const messages = this.bySession.get(sessionId);
    if (!messages) return;
    this.set(
      sessionId,
      messages.map((m) => (m.id === id ? { ...m, ...change(m), updatedAt: Date.now() } : m)),
    );
    markChatsChanged();
    if (when === 'now') {
      this.clearTimer(id);
      this.write(sessionId, id);
    } else if (!this.timers.has(id)) {
      const timer = setTimeout(() => {
        this.timers.delete(id);
        this.write(sessionId, id);
      }, CHECKPOINT_MS);
      this.timers.set(id, { sessionId, timer });
    }
  }

  /** Removes messages from a session and from storage. */
  remove(sessionId: string, ids: string[]): void {
    const messages = this.bySession.get(sessionId);
    if (!messages) return;
    for (const id of ids) this.clearTimer(id);
    this.set(
      sessionId,
      messages.filter((m) => !ids.includes(m.id)),
    );
    void save(() => dbDeleteMany('messages', ids)).then(() => broadcast({ kind: 'messages', sessionId }));
  }

  /** Drops a deleted session from memory (its records are deleted with the session). */
  forget(sessionId: string): void {
    for (const [id, pending] of this.timers) if (pending.sessionId === sessionId) this.clearTimer(id);
    this.bySession.delete(sessionId);
    this.notify();
  }

  /** Saves every pending checkpoint now (e.g. when the page is being hidden or closed). */
  flush(): void {
    for (const [id, { sessionId }] of [...this.timers]) {
      this.clearTimer(id);
      this.write(sessionId, id);
    }
  }

  private write(sessionId: string, id: string): void {
    const message = this.bySession.get(sessionId)?.find((m) => m.id === id);
    if (!message) return;
    void save(() => dbPut('messages', message)).then(() => broadcast({ kind: 'messages', sessionId }));
  }

  private async read(sessionId: string): Promise<MessageRecord[]> {
    let records: MessageRecord[];
    try {
      records = await dbGetAllByIndex<MessageRecord>('messages', 'bySession', sessionRange(sessionId));
    } catch (err) {
      await save(() => Promise.reject(err));
      return [];
    }
    return this.recoverInterrupted(sessionId, records);
  }

  /** Marks replies left "streaming" by a closed or crashed tab as interrupted. */
  private async recoverInterrupted(sessionId: string, records: MessageRecord[]): Promise<MessageRecord[]> {
    if (!records.some((r) => r.status === 'streaming')) return records;
    if (typeof navigator.locks?.request !== 'function') return records;
    const generatingElsewhere = await navigator.locks.request(
      generationLockName(sessionId),
      { ifAvailable: true },
      (lock) => lock === null,
    );
    if (generatingElsewhere) return records;
    const fixed = records.map((r) =>
      r.status === 'streaming'
        ? { ...r, status: 'done' as const, finishReason: 'cancelled' as const, interrupted: true, updatedAt: Date.now() }
        : r,
    );
    void save(() => dbPutMany('messages', fixed.filter((r) => r.interrupted)));
    return fixed;
  }

  private hasPending(sessionId: string): boolean {
    for (const pending of this.timers.values()) if (pending.sessionId === sessionId) return true;
    return false;
  }

  private clearTimer(id: string): void {
    const pending = this.timers.get(id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.timers.delete(id);
  }

  private set(sessionId: string, messages: MessageRecord[]): void {
    this.bySession.set(sessionId, messages);
    this.notify();
  }

  private notify(): void {
    for (const l of this.listeners) l();
  }
}

export const messageStore = new MessageStore();
