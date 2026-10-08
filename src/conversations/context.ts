// Bounded history: never send a whole stored conversation to the model.
//
// The exact cut needs the model's tokenizer, which lives in the inference
// worker (`fitTurns` in src/runtime/context.ts trims to the token budget). This
// cheap first cut keeps only the most recent turns that could possibly fit, so
// a long conversation costs the same to send as a short one.

import type { ChatTurn } from '../runtime/contract';

/**
 * Generous characters-per-token bound: real text averages ~4 characters per
 * token, so this keeps roughly twice what can fit and leaves the exact trim to the runtime.
 */
export const CHARS_PER_TOKEN_BOUND = 8;

/** The most recent turns whose total length is within `maxChars` (always at least the last turn). */
export function boundedHistory(turns: ChatTurn[], maxChars: number): ChatTurn[] {
  let total = 0;
  let start = turns.length;
  while (start > 0) {
    const length = turns[start - 1]!.content.length;
    if (start < turns.length && total + length > maxChars) break;
    total += length;
    start--;
  }
  return turns.slice(start);
}
