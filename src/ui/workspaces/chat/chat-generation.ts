// Sends chat turns to the runtime and streams the reply into the chat store.
// Lives outside React components so a reply keeps streaming (and can still be
// stopped) after the user switches to another chat and back.

import { useSyncExternalStore } from 'react';
import type { ModelSpec } from '../../../catalogue/models';
import { getRuntimeClient, type GenerationHandle } from '../../../runtime/client';
import { RuntimeError, type ChatTurn } from '../../../runtime/contract';
import { getMessages, patchMessage, updateMessages } from './chat-store';

const active = new Map<string, GenerationHandle>();
const listeners = new Set<() => void>();

function notify() {
  for (const l of listeners) l();
}

export function useIsGenerating(sessionId: string): boolean {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => active.has(sessionId),
  );
}

export function stopGeneration(sessionId: string): void {
  active.get(sessionId)?.cancel();
}

export function sendMessage(sessionId: string, model: ModelSpec, content: string): void {
  if (active.has(sessionId)) return;

  const history: ChatTurn[] = getMessages(sessionId)
    .filter((m) => m.status !== 'error' && m.content.trim() !== '')
    .map((m) => ({ role: m.role, content: m.content }));
  const turns: ChatTurn[] = [...history, { role: 'user', content }];

  const stamp = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  const replyId = `a-${stamp}`;
  updateMessages(sessionId, (prev) => [
    ...prev,
    { id: `u-${stamp}`, role: 'user', content },
    { id: replyId, role: 'assistant', content: '', status: 'streaming' },
  ]);

  const { maxNewTokens, temperature, topP, repetitionPenalty, repetitionWindow, contextTokens } = model.generation;
  const handle = getRuntimeClient().generate(
    { turns, options: { maxNewTokens, temperature, topP, repetitionPenalty, repetitionWindow, contextTokens } },
    (delta) => patchMessage(sessionId, replyId, (m) => ({ content: m.content + delta })),
  );
  active.set(sessionId, handle);
  notify();

  handle.result
    .then((result) => {
      patchMessage(sessionId, replyId, () => ({
        content: result.text,
        status: 'done',
        finishReason: result.finishReason,
        stats: result.stats,
      }));
    })
    .catch((err: unknown) => {
      const message = err instanceof RuntimeError || err instanceof Error ? err.message : String(err);
      patchMessage(sessionId, replyId, () => ({ status: 'error', error: message }));
    })
    .finally(() => {
      active.delete(sessionId);
      notify();
    });
}
