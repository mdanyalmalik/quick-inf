// React access to a chat's saved messages (see src/conversations/message-store.ts).

import { useEffect, useSyncExternalStore } from 'react';
import { messageStore } from '../../../conversations/message-store';
import type { ChatMessage } from './types';

const EMPTY: ChatMessage[] = [];

/** The session's messages; `loaded` is false until they've been read from storage. */
export function useMessages(sessionId: string): { messages: ChatMessage[]; loaded: boolean } {
  const messages = useSyncExternalStore(messageStore.subscribe, () => messageStore.get(sessionId));
  useEffect(() => {
    void messageStore.load(sessionId);
  }, [sessionId]);
  return { messages: messages ?? EMPTY, loaded: messages !== undefined };
}
