// Sends chat turns to the runtime and streams the reply into the message store
// (which saves it). Lives outside React components so a reply keeps streaming
// (and can still be stopped) after the user switches to another chat and back.

import { fittedContext } from '../../../recommend/memory-evidence';
import { useSyncExternalStore } from 'react';
import type { ModelSpec } from '../../../catalogue/models';
import { getRuntimeClient, type GenerationHandle } from '../../../runtime/client';
import { RuntimeError, type ChatTurn } from '../../../runtime/contract';
import { boundedHistory, CHARS_PER_TOKEN_BOUND } from '../../../conversations/context';
import { historyText } from '../../../conversations/reasoning';
import { generationLockName, messageStore } from '../../../conversations/message-store';
import { sessionStore } from '../../../conversations/session-store';

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

/** Removes a failed reply and the message that prompted it, then sends that message again. */
export function retryReply(sessionId: string, model: ModelSpec, replyId: string): void {
  if (active.has(sessionId)) return;
  const messages = messageStore.get(sessionId) ?? [];
  const index = messages.findIndex((m) => m.id === replyId);
  const prompt = messages[index - 1];
  if (index < 1 || prompt?.role !== 'user') return;
  messageStore.remove(sessionId, [replyId, prompt.id]);
  sendMessage(sessionId, model, prompt.content);
}

/** Stops any reply in the session and drops it from memory (its records go with the session). */
export function forgetSession(sessionId: string): void {
  stopGeneration(sessionId);
  messageStore.forget(sessionId);
}

/** Sends a message and streams the reply. The session's messages must be loaded. */
export function sendMessage(sessionId: string, model: ModelSpec, content: string): void {
  const messages = messageStore.get(sessionId);
  if (active.has(sessionId) || !messages) return;

  // A reasoning model's past thinking stays out of what it sees (as its own templates do).
  const reasoning = model.promptFormat.reasoning ?? false;
  const history: ChatTurn[] = messages
    .filter((m) => m.status !== 'error' && m.content.trim() !== '')
    .map((m) => ({ role: m.role, content: m.role === 'assistant' ? historyText(m.content, reasoning) : m.content }))
    .filter((t) => t.content.trim() !== '');
  const { maxNewTokens, temperature, topP, repetitionPenalty, repetitionWindow } = model.generation;
  // Shorter conversations on devices whose memory budget doesn't fit the model's full context (see recommend/budget.ts).
  const contextTokens = Math.min(model.generation.contextTokens, fittedContext(model.id) ?? model.generation.contextTokens);
  // Only recent history goes to the worker; it trims to the exact token budget.
  const turns = boundedHistory([...history, { role: 'user', content }], contextTokens * CHARS_PER_TOKEN_BOUND);

  const replyId = crypto.randomUUID();
  messageStore.append(sessionId, [
    { id: crypto.randomUUID(), role: 'user', content },
    { id: replyId, role: 'assistant', content: '', status: 'streaming' },
  ]);
  sessionStore.touch(sessionId);

  const handle = getRuntimeClient().generate(
    { turns, options: { maxNewTokens, temperature, topP, repetitionPenalty, repetitionWindow, contextTokens } },
    (delta) => messageStore.patch(sessionId, replyId, (m) => ({ content: m.content + delta }), 'checkpoint'),
  );
  active.set(sessionId, handle);
  notify();
  // Tells other tabs (and this one after a reload) that this reply is still being written.
  void navigator.locks?.request(generationLockName(sessionId), () => handle.result.catch(() => {}));

  handle.result
    .then((result) => {
      messageStore.patch(sessionId, replyId, () => ({
        content: result.text,
        status: 'done',
        finishReason: result.finishReason,
        stats: result.stats,
      }));
    })
    .catch((err: unknown) => {
      if (err instanceof RuntimeError && err.code === 'cancelled') {
        // Force stop restarted the runtime: keep whatever text had arrived.
        messageStore.patch(sessionId, replyId, () => ({ status: 'done', finishReason: 'cancelled' }));
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      const code = err instanceof RuntimeError ? err.code : 'generation-failed';
      messageStore.patch(sessionId, replyId, () => ({ status: 'error', error: message, errorCode: code }));
    })
    .finally(() => {
      active.delete(sessionId);
      notify();
      sessionStore.touch(sessionId);
    });
}
