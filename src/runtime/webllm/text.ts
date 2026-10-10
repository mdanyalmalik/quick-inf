// Text handling for the WebLLM adapter, kept apart from the library so it can be unit-tested.

import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm';

/** WebLLM's empty thinking block (Qwen3 with thinking off), which it puts at the start of the reply text. */
const EMPTY_THINKING = /^<think>\s*<\/think>\s*/;
const EMPTY_THINKING_TEXT = '<think>\n\n</think>\n\n';

/**
 * The reply without a leading empty thinking block, or null while the text so
 * far could still be the start of one (`final`: no more text is coming).
 */
export function stripThinking(text: string, final = false): string | null {
  const match = EMPTY_THINKING.exec(text);
  if (match && (final || match[0].length < text.length)) return text.slice(match[0].length);
  if (!final && (match || EMPTY_THINKING_TEXT.startsWith(text))) return null;
  return text;
}

/** Whether `messages` are exactly `previous` plus one new message (WebLLM's condition for reusing its cache). */
export function extendsByOne(messages: ChatCompletionMessageParam[], previous: ChatCompletionMessageParam[]): boolean {
  if (messages.length !== previous.length + 1) return false;
  return previous.every((m, i) => messages[i]!.role === m.role && messages[i]!.content === m.content);
}
