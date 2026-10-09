import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BACKUP_FORMAT, BackupError, backupFileName, parseBackup, planRestore, type ChatBackup } from './backup';
import type { MessageRecord, SessionRecord } from './records';

const session = (id: string): SessionRecord => ({ id, modelId: 'smollm2-360m-instruct', title: `Chat ${id}`, createdAt: 1, updatedAt: 2 });
const message = (id: string, sessionId: string, seq: number, extra: Partial<MessageRecord> = {}): MessageRecord => ({
  id,
  sessionId,
  seq,
  role: seq % 2 ? 'user' : 'assistant',
  content: `message ${id}`,
  createdAt: 1,
  updatedAt: 2,
  ...extra,
});
const backup = (sessions: SessionRecord[], messages: MessageRecord[]): ChatBackup => ({
  format: BACKUP_FORMAT,
  version: 1,
  exportedAt: 100,
  sessions,
  messages,
});

describe('parseBackup', () => {
  it('reads a backup it wrote', () => {
    const b = backup([session('a')], [message('m1', 'a', 1), message('m2', 'a', 2, { status: 'done', finishReason: 'stop' })]);
    expect(parseBackup(JSON.stringify(b))).toEqual(b);
  });

  it.each([
    ['not JSON', 'hello', 'isn’t valid JSON'],
    ['another JSON file', '{"name":"x"}', 'isn’t a Quick Inference chat backup'],
    ['a newer format version', JSON.stringify({ ...backup([], []), version: 2 }), 'newer version'],
    ['a missing chat list', JSON.stringify({ format: BACKUP_FORMAT, version: 1 }), 'damaged'],
    ['an incomplete chat', JSON.stringify(backup([{ id: 'a' } as SessionRecord], [])), 'damaged'],
    ['an incomplete message', JSON.stringify(backup([session('a')], [{ id: 'm', sessionId: 'a' } as MessageRecord])), 'damaged'],
  ])('rejects %s with a clear message', (_name, text, expected) => {
    expect(() => parseBackup(text)).toThrow(BackupError);
    expect(() => parseBackup(text)).toThrow(expected);
  });
});

describe('planRestore', () => {
  const file = backup(
    [session('a'), session('b')],
    [message('a1', 'a', 1), message('a2', 'a', 2), message('b1', 'b', 1), message('x1', 'missing', 1)],
  );

  it('adds every chat into an empty browser', () => {
    const plan = planRestore(file, new Set());
    expect(plan.sessions.map((s) => s.id)).toEqual(['a', 'b']);
    // Messages of chats not in the file are ignored.
    expect(plan.messages.map((m) => m.id)).toEqual(['a1', 'a2', 'b1']);
    expect(plan.skipped).toBe(0);
  });

  it('skips chats already here, with their messages', () => {
    const plan = planRestore(file, new Set(['a']));
    expect(plan.sessions.map((s) => s.id)).toEqual(['b']);
    expect(plan.messages.map((m) => m.id)).toEqual(['b1']);
    expect(plan.skipped).toBe(1);
  });

  it('adds nothing when every chat is already here', () => {
    const plan = planRestore(file, new Set(['a', 'b', 'other']));
    expect(plan).toEqual({ sessions: [], messages: [], skipped: 2 });
  });

  it('marks a reply that was still being written as interrupted', () => {
    const streaming = backup([session('a')], [message('a2', 'a', 2, { status: 'streaming', content: 'Half a rep' })]);
    const [m] = planRestore(streaming, new Set()).messages;
    expect(m).toMatchObject({ status: 'done', finishReason: 'cancelled', interrupted: true, content: 'Half a rep' });
  });
});

it('names backup files by date', () => {
  expect(backupFileName(new Date(2026, 9, 9, 15, 30).getTime())).toBe('quick-inference-chats-2026-10-09.json');
});

describe('backup status', () => {
  let now = 1000;
  beforeEach(() => {
    const data = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
    });
    vi.spyOn(Date, 'now').mockImplementation(() => now);
    vi.resetModules();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('is unsaved after a change, saved after a backup, unsaved again after the next change', async () => {
    const { backupStatus, markBackedUp, markChatsChanged } = await import('./backup-status');
    expect(backupStatus.get()).toEqual({ backedUpAt: null, unsaved: false });
    markChatsChanged();
    expect(backupStatus.get().unsaved).toBe(true);
    now = 2000;
    markBackedUp(2000);
    expect(backupStatus.get()).toEqual({ backedUpAt: 2000, unsaved: false });
    // A change in the same millisecond as the backup still counts as after it.
    markChatsChanged();
    expect(backupStatus.get().unsaved).toBe(true);
  });

  it('counts chats from before change tracking as not backed up, once', async () => {
    const { backupStatus, markBackedUp, noteExistingChats } = await import('./backup-status');
    noteExistingChats(0);
    expect(backupStatus.get().unsaved).toBe(false);
    noteExistingChats(3);
    expect(backupStatus.get().unsaved).toBe(true);
    markBackedUp(5000);
    noteExistingChats(3);
    expect(backupStatus.get().unsaved).toBe(false);
  });

  it('tells subscribers when the status changes', async () => {
    const { backupStatus, markChatsChanged } = await import('./backup-status');
    const listener = vi.fn();
    backupStatus.subscribe(listener);
    markChatsChanged();
    markChatsChanged(); // no change in status
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
