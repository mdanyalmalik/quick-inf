// The session list (chats today), kept in memory and saved to IndexedDB.
// Only metadata lives here; messages load separately, per session, when opened.
//
// A new session stays in memory until something happens in it (`touch`), so
// opening "New chat" and leaving doesn't fill the list with empty chats.

import { dbDeleteSession, dbGetAll, dbPut } from '../store/db';
import { markChatsChanged, noteExistingChats } from './backup-status';
import { save } from './persistence';
import { findImageModel } from '../catalogue/image-models';
import type { SessionRecord } from './records';
import { broadcast, onBroadcast } from './sync';

class SessionStore {
  private sessions: SessionRecord[] = [];
  /** Ids written to IndexedDB (the rest are new, unsaved sessions). */
  private saved = new Set<string>();
  private loaded = false;
  private loading: Promise<void> | null = null;
  private readonly listeners = new Set<() => void>();

  constructor() {
    onBroadcast((notice) => {
      if (notice.kind === 'sessions') void this.reload();
    });
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** All sessions, most recently active first. Same array until something changes. */
  getAll = (): SessionRecord[] => this.sessions;

  isLoaded = (): boolean => this.loaded;

  /** Sessions saved to IndexedDB (not counting new, still-empty ones). */
  savedCount = (): number => this.saved.size;

  /** Saved image tabs, of `savedCount`. */
  savedImageCount = (): number => this.sessions.filter((s) => this.saved.has(s.id) && findImageModel(s.modelId)).length;

  /** Reads the saved sessions once; safe to call repeatedly. */
  init(): Promise<void> {
    this.loading ??= this.reload().then(() => noteExistingChats(this.saved.size));
    return this.loading;
  }

  /** Re-reads the saved sessions (after chats were added outside this store, e.g. restored). */
  refresh(): Promise<void> {
    return this.reload();
  }

  get(id: string): SessionRecord | undefined {
    return this.sessions.find((s) => s.id === id);
  }

  create(modelId: string, title: string): SessionRecord {
    const now = Date.now();
    const session: SessionRecord = { id: crypto.randomUUID(), modelId, title, createdAt: now, updatedAt: now };
    this.set([session, ...this.sessions]);
    return session;
  }

  rename(id: string, title: string): void {
    this.update(id, (s) => ({ ...s, title }));
  }

  /** Marks activity in a session (moves it to the top) and saves it. */
  touch(id: string): void {
    if (!this.get(id)) return; // deleted meanwhile
    this.saved.add(id);
    this.update(id, (s) => ({ ...s, updatedAt: Date.now() }));
  }

  /** Removes the session and its messages. */
  async delete(id: string): Promise<void> {
    const wasSaved = this.saved.delete(id);
    this.set(this.sessions.filter((s) => s.id !== id));
    if (wasSaved) {
      await save(() => dbDeleteSession(id));
      broadcast({ kind: 'sessions' });
    }
  }

  private update(id: string, change: (s: SessionRecord) => SessionRecord): void {
    const current = this.get(id);
    if (!current) return;
    const next = change(current);
    this.set(sortByActivity(this.sessions.map((s) => (s.id === id ? next : s))));
    if (this.saved.has(id)) {
      markChatsChanged();
      void save(() => dbPut('sessions', next)).then(() => broadcast({ kind: 'sessions' }));
    }
  }

  private async reload(): Promise<void> {
    let records: SessionRecord[];
    try {
      records = await dbGetAll<SessionRecord>('sessions');
    } catch (err) {
      await save(() => Promise.reject(err)); // records the error for the UI
      records = [];
    }
    // Keep this tab's unsaved new sessions (decided before `saved` is replaced, so a
    // session deleted by another tab isn't mistaken for a new one).
    const unsaved = this.sessions.filter((s) => !this.saved.has(s.id));
    this.saved = new Set(records.map((r) => r.id));
    this.loaded = true;
    this.set(sortByActivity([...unsaved, ...records]));
  }

  private set(sessions: SessionRecord[]): void {
    this.sessions = sessions;
    for (const l of this.listeners) l();
  }
}

function sortByActivity(sessions: SessionRecord[]): SessionRecord[] {
  return [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
}

export const sessionStore = new SessionStore();
