import { useState, useSyncExternalStore } from 'react';
import { BackupError, downloadBackup, restoreBackup } from '../../conversations/backup';
import { backupStatus, type BackupStatus } from '../../conversations/backup-status';
import { sessionStore } from '../../conversations/session-store';

export type BackupMessage = { kind: 'ok' | 'error'; text: string } | null;

export interface ChatBackup {
  status: BackupStatus;
  /** Saved chats in this browser. */
  chatCount: number;
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
    needsBackup: status.unsaved && chatCount > 0,
    busy,
    message,
    save: () =>
      run(async () => {
        const count = await downloadBackup();
        return `Saved a backup of ${plural(count, 'chat')}. Keep the file somewhere safe.`;
      }),
    restore: (file) =>
      run(async () => {
        const { added, skipped } = await restoreBackup(await file.text());
        if (added === 0 && skipped === 0) return 'That backup has no chats in it.';
        if (added === 0) return `Nothing to restore: all ${plural(skipped, 'chat')} in that backup are already here.`;
        return `Restored ${plural(added, 'chat')}${skipped > 0 ? ` (${plural(skipped, 'chat')} already here)` : ''}.`;
      }),
  };
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}
