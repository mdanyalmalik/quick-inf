import { useEffect, useState, useSyncExternalStore } from 'react';
import { sessionStore } from '../../conversations/session-store';
import { measureChatBytes } from '../../conversations/usage';
import { useStorage } from './use-runtime';

export interface LocalUsage {
  /** Exact size of the model files on disk. */
  models: number;
  /** Saved chats, measured as data; null until measured. */
  chats: number | null;
  /** Download records and settings, measured as data. */
  appData: number;
}

/** What this app stores on the device, by part. Re-measures chats when the chat list changes. */
export function useLocalUsage(): LocalUsage {
  const storage = useStorage();
  const sessions = useSyncExternalStore(sessionStore.subscribe, sessionStore.getAll);
  const [chats, setChats] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    // Debounced: the list changes several times while a reply is sent.
    const timer = setTimeout(() => {
      measureChatBytes()
        .then((bytes) => alive && setChats(bytes))
        .catch(() => {});
    }, 300);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [sessions]);

  return { models: storage.modelFileBytes, chats, appData: storage.recordBytes + localStorageBytes() };
}

function localStorageBytes(): number {
  try {
    const encoder = new TextEncoder();
    let total = 0;
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i) ?? '';
      total += encoder.encode(key).length + encoder.encode(localStorage.getItem(key) ?? '').length;
    }
    return total;
  } catch {
    return 0;
  }
}
