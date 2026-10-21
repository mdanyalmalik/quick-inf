// Orpheus 3B's prompt, as canopyai's engine builds it: start of human, BOS, "{voice}: {text}", end of turn, end of
// human, start of AI, start of speech; Maya1's, which shares the layout with a voice description. Their audio tokens:
// 7 per SNAC frame, token = 128266 + slot × 4096 + code.

import type { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { encodeWithTags } from './chatterbox-text';

const START_OF_HUMAN = 128259;
const BOS = 128000;
const END_OF_TURN = 128009;
const END_OF_HUMAN = 128260;
const START_OF_AI = 128261;
const START_OF_SPEECH = 128257;
/** The first audio token, and codes per SNAC codebook. */
export const ORPHEUS_AUDIO_BASE = 128266;
const SNAC_CODES = 4096;

export function orpheusPrompt(tokenizer: BpeTokenizer, voice: string, text: string): number[] {
  return [START_OF_HUMAN, BOS, ...tokenizer.encode(`${voice}: ${text}`), END_OF_TURN, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH];
}

/** Maya1's tags (<laugh>, <sigh>, …): its added tokens after the audio tokens, each one token in the text. */
export function mayaTags(tokenizerJson: { added_tokens: { id: number; content: string }[] }): string[] {
  return tokenizerJson.added_tokens.filter((t) => t.id >= ORPHEUS_AUDIO_BASE + 7 * SNAC_CODES).map((t) => t.content);
}

/** Maya1's prompt (its model card's): BOS, start of human, BOS, `<description="…"> text`, then as Orpheus. */
export function mayaPrompt(tokenizer: BpeTokenizer, tags: readonly string[], description: string, text: string): number[] {
  return [
    BOS,
    START_OF_HUMAN,
    BOS,
    ...encodeWithTags(tokenizer, `<description="${description}"> ${text}`, tags),
    END_OF_TURN,
    END_OF_HUMAN,
    START_OF_AI,
    START_OF_SPEECH,
  ];
}

/**
 * Generated tokens → SNAC's three code layers (12, 24 and 48 codes a second): the audio tokens counted in order, every
 * 7 a frame [L1, L2a, L3a, L3b, L2b, L3c, L3d]; an unfinished last frame is dropped, out-of-range codes clamped.
 */
export function orpheusSnacCodes(tokens: readonly number[]): [number[], number[], number[]] {
  const codes = tokens.filter((t) => t >= ORPHEUS_AUDIO_BASE).map((t, i) => Math.min(SNAC_CODES - 1, Math.max(0, t - ORPHEUS_AUDIO_BASE - (i % 7) * SNAC_CODES)));
  const l1: number[] = [];
  const l2: number[] = [];
  const l3: number[] = [];
  for (let f = 0; f + 7 <= codes.length; f += 7) {
    const c = codes.slice(f, f + 7);
    l1.push(c[0]!);
    l2.push(c[1]!, c[4]!);
    l3.push(c[2]!, c[3]!, c[5]!, c[6]!);
  }
  return [l1, l2, l3];
}
