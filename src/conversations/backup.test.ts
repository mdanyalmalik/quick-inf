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
const backup = (sessions: SessionRecord[], messages: MessageRecord[], images: ChatBackup['images'] = []): ChatBackup => ({
  format: BACKUP_FORMAT,
  version: 2,
  exportedAt: 100,
  sessions,
  messages,
  images,
});

describe('parseBackup', () => {
  it('reads a backup it wrote', () => {
    const b = backup([session('a')], [message('m1', 'a', 1), message('m2', 'a', 2, { status: 'done', finishReason: 'stop' })]);
    expect(parseBackup(JSON.stringify(b))).toEqual(b);
  });

  it.each([
    ['not JSON', 'hello', 'isn’t valid JSON'],
    ['another JSON file', '{"name":"x"}', 'isn’t a Quick Inference chat backup'],
    ['a newer format version', JSON.stringify({ ...backup([], []), version: 3 }), 'newer version'],
    ['an incomplete image', JSON.stringify(backup([session('a')], [], [{ sessionId: 'a' } as ChatBackup['images'][number]])), 'damaged'],
    ['a missing chat list', JSON.stringify({ format: BACKUP_FORMAT, version: 1 }), 'damaged'],
    ['an incomplete chat', JSON.stringify(backup([{ id: 'a' } as SessionRecord], [])), 'damaged'],
    ['an incomplete message', JSON.stringify(backup([session('a')], [{ id: 'm', sessionId: 'a' } as MessageRecord])), 'damaged'],
  ])('rejects %s with a clear message', (_name, text, expected) => {
    expect(() => parseBackup(text)).toThrow(BackupError);
    expect(() => parseBackup(text)).toThrow(expected);
  });

  it('reads a version 1 backup (before images) as having none', () => {
    const { images: _images, ...v1 } = { ...backup([session('a')], [message('m1', 'a', 1)]), version: 1 };
    expect(parseBackup(JSON.stringify(v1)).images).toEqual([]);
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
    expect(plan).toEqual({ sessions: [], messages: [], images: [], imageRecords: [], skipped: 2 });
  });

  it('restores an image tab’s record and picture with its session', async () => {
    const png = btoa(String.fromCharCode(137, 80, 78, 71));
    const withImages = backup(
      [session('a'), session('b')],
      [],
      [
        { sessionId: 'a', record: { id: 'g1', prompt: 'a fox' }, generationId: 'g1', png },
        { sessionId: 'b', record: { id: 'g2', prompt: 'placeholder' } },
      ],
    );
    const plan = planRestore(withImages, new Set(['b']));
    expect(plan.imageRecords).toEqual([{ sessionId: 'a', record: { id: 'g1', prompt: 'a fox' } }]);
    expect(plan.images).toHaveLength(1);
    expect(plan.images[0]).toMatchObject({ sessionId: 'a', generationId: 'g1' });
    expect([...new Uint8Array(await plan.images[0]!.png.arrayBuffer())]).toEqual([137, 80, 78, 71]);
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
