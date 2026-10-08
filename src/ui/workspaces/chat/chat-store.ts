// In-memory message store, keyed by session id, so chats survive switching
// between them. Temporary: milestone 4 replaces it with IndexedDB persistence.

import { useSyncExternalStore } from 'react';
import type { ChatMessage } from './types';

const EMPTY: ChatMessage[] = [];
const messagesBySession = new Map<string, ChatMessage[]>();
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getMessages(sessionId: string): ChatMessage[] {
  return messagesBySession.get(sessionId) ?? EMPTY;
}

export function updateMessages(sessionId: string, update: (prev: ChatMessage[]) => ChatMessage[]): void {
  messagesBySession.set(sessionId, update(getMessages(sessionId)));
  for (const l of listeners) l();
}

export function patchMessage(sessionId: string, messageId: string, patch: (m: ChatMessage) => Partial<ChatMessage>): void {
  updateMessages(sessionId, (prev) => prev.map((m) => (m.id === messageId ? { ...m, ...patch(m) } : m)));
}

export function useMessages(sessionId: string): ChatMessage[] {
  return useSyncExternalStore(subscribe, () => getMessages(sessionId));
}
