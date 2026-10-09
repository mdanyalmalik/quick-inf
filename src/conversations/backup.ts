// Chat backup files: every chat and message in one JSON file the user saves, and
// restoring from one. Browsers may clear site data (or the user may), and chats
// can't be downloaded again the way models can, so this is how they're kept safe.
//
// Restoring adds the chats the file has and this browser doesn't (matched by id);
// chats already here are left as they are, so restoring twice adds nothing.

import { dbImportChats, dbReadChats } from '../store/db';
import { markBackedUp, markChatsChanged } from './backup-status';
import type { MessageRecord, SessionRecord } from './records';
import { sessionStore } from './session-store';
import { broadcast } from './sync';

export const BACKUP_FORMAT = 'quick-inference-chats';
export const BACKUP_VERSION = 1;

export interface ChatBackup {
  format: typeof BACKUP_FORMAT;
  version: number;
  exportedAt: number;
  sessions: SessionRecord[];
  messages: MessageRecord[];
}

export class BackupError extends Error {}

/** Everything saved, as a backup. Read in one transaction, so chats and messages match. */
export async function createBackup(now = Date.now()): Promise<ChatBackup> {
  const { sessions, messages } = await dbReadChats<SessionRecord, MessageRecord>();
  return { format: BACKUP_FORMAT, version: BACKUP_VERSION, exportedAt: now, sessions, messages };
}

export function backupFileName(at: number): string {
  const d = new Date(at);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `quick-inference-chats-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}.json`;
}

/** Saves a backup file through the browser's download, and records the backup. Returns the chat count. */
export async function downloadBackup(): Promise<number> {
  const backup = await createBackup();
  const blob = new Blob([JSON.stringify(backup)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = backupFileName(backup.exportedAt);
  document.body.append(link);
  link.click();
  link.remove();
  // Revoking at once can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  markBackedUp(backup.exportedAt);
  return backup.sessions.length;
}

/** Reads and checks a backup file's text. Throws BackupError with a message for the user. */
export function parseBackup(text: string): ChatBackup {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new BackupError('This file isn’t a Quick Inference chat backup (it isn’t valid JSON).');
  }
  if (!isObject(data) || data.format !== BACKUP_FORMAT) {
    throw new BackupError('This file isn’t a Quick Inference chat backup.');
  }
  if (typeof data.version !== 'number' || data.version > BACKUP_VERSION) {
    throw new BackupError('This backup was made by a newer version of Quick Inference. Update the app and try again.');
  }
  if (!Array.isArray(data.sessions) || !Array.isArray(data.messages)) {
    throw new BackupError('This backup file is damaged (its chat list is missing).');
  }
  const sessions = data.sessions.filter(isSession);
  const messages = data.messages.filter(isMessage);
  if (sessions.length !== data.sessions.length || messages.length !== data.messages.length) {
    throw new BackupError('This backup file is damaged (some chats or messages are incomplete).');
  }
  return { format: BACKUP_FORMAT, version: data.version, exportedAt: Number(data.exportedAt) || 0, sessions, messages };
}

export interface RestorePlan {
  /** Chats to add, with their messages. */
  sessions: SessionRecord[];
  messages: MessageRecord[];
  /** Chats in the file that this browser already has. */
  skipped: number;
}

/**
 * Which chats from `backup` to add, given the chat ids already here. Messages of
 * a reply that was still being written when the backup was made are marked
 * interrupted, as they would be after a closed tab.
 */
export function planRestore(backup: ChatBackup, existingIds: Set<string>): RestorePlan {
  const sessions = backup.sessions.filter((s) => !existingIds.has(s.id));
  const adding = new Set(sessions.map((s) => s.id));
  const messages = backup.messages
    .filter((m) => adding.has(m.sessionId))
    .map((m) =>
      m.status === 'streaming' ? { ...m, status: 'done' as const, finishReason: 'cancelled' as const, interrupted: true } : m,
    );
  return { sessions, messages, skipped: backup.sessions.length - sessions.length };
}

export interface RestoreResult {
  added: number;
  skipped: number;
}

/** Restores chats from a backup file's text (see the top of this file). */
export async function restoreBackup(text: string): Promise<RestoreResult> {
  const backup = parseBackup(text);
  const { sessions, skipped, hadChats } = await dbImportChats((existingIds) => planRestore(backup, existingIds));
  if (sessions.length > 0) {
    // Restoring into an empty browser leaves exactly what the file holds, so it's backed up;
    // adding to existing chats makes a set no single backup holds yet.
    if (hadChats) markChatsChanged();
    else markBackedUp(Math.max(backup.exportedAt, Date.now()));
    await sessionStore.refresh();
    broadcast({ kind: 'sessions' });
  }
  return { added: sessions.length, skipped };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isSession(value: unknown): value is SessionRecord {
  return (
    isObject(value) &&
    typeof value.id === 'string' &&
    typeof value.modelId === 'string' &&
    typeof value.title === 'string' &&
    typeof value.createdAt === 'number' &&
    typeof value.updatedAt === 'number'
  );
}

function isMessage(value: unknown): value is MessageRecord {
  return (
    isObject(value) &&
    typeof value.id === 'string' &&
    typeof value.sessionId === 'string' &&
    typeof value.seq === 'number' &&
    (value.role === 'user' || value.role === 'assistant') &&
    typeof value.content === 'string' &&
    typeof value.createdAt === 'number' &&
    typeof value.updatedAt === 'number'
  );
}
