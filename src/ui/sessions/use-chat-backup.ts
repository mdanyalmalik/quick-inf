import { useState, useSyncExternalStore } from 'react';
import { BackupError, downloadBackup, restoreBackup } from '../../conversations/backup';
import { backupStatus, type BackupStatus } from '../../conversations/backup-status';
import { findImageModel } from '../../catalogue/image-models';
import { sessionStore } from '../../conversations/session-store';

export type BackupMessage = { kind: 'ok' | 'error'; text: string } | null;

export interface ChatBackup {
  status: BackupStatus;
  /** Saved chats and image tabs in this browser. */
  chatCount: number;
  /** How much is saved, in words ("3 chats and 1 image"). */
  savedText: string;
  /** There are chats a backup doesn't have yet. */
  needsBackup: boolean;
  busy: boolean;
  message: BackupMessage;
  save: () => Promise<void>;
  restore: (file: File) => Promise<void>;
}

/** Saving and restoring chat backups, with the last result to show. */
export function useChatBackup(): ChatBackup {
  const status = useSyncExternalStore(backupStatus.subscribe, backupStatus.get);
  const chatCount = useSyncExternalStore(sessionStore.subscribe, sessionStore.savedCount);
  const imageCount = useSyncExternalStore(sessionStore.subscribe, sessionStore.savedImageCount);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<BackupMessage>(null);

  async function run(action: () => Promise<string>) {
    setBusy(true);
    setMessage(null);
    try {
      setMessage({ kind: 'ok', text: await action() });
    } catch (err) {
      const text =
        err instanceof BackupError ? err.message : `That didn’t work: ${err instanceof Error ? err.message : String(err)}`;
      setMessage({ kind: 'error', text });
    } finally {
      setBusy(false);
    }
  }

  return {
    status,
    chatCount,
    savedText: describe(chatCount - imageCount, imageCount),
    needsBackup: status.unsaved && chatCount > 0,
    busy,
    message,
    save: () =>
      run(async () => {
        const sessions = await downloadBackup();
        return `Saved a backup of ${describeSessions(sessions)}. Keep the file somewhere safe.`;
      }),
    restore: (file) =>
      run(async () => {
        const { added, skipped } = await restoreBackup(await file.text());
        if (added.length === 0 && skipped === 0) return 'That backup has nothing in it.';
        if (added.length === 0) return `Nothing to restore: everything in that backup (${skipped}) is already here.`;
        return `Restored ${describeSessions(added)}${skipped > 0 ? ` (${skipped} already here)` : ''}.`;
      }),
  };
}

function describeSessions(sessions: { modelId: string }[]): string {
  const images = sessions.filter((s) => findImageModel(s.modelId)).length;
  return describe(sessions.length - images, images);
}

/** "2 chats", "1 image", "2 chats and 1 image". */
function describe(chats: number, images: number): string {
  const parts = [chats > 0 || images === 0 ? plural(chats, 'chat') : '', images > 0 ? plural(images, 'image') : ''];
  return parts.filter(Boolean).join(' and ');
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}
