// Pocket TTS's text handling (Kyutai's pocket-tts, pocket_tts/models/text_chunking.py), with the English model's
// settings: no number normalisation (the model reads digits itself); curly apostrophes made straight, whitespace
// collapsed, a capital first letter, a sentence end added; split after sentence-ending tokens (not a decimal point),
// long sentences after , ; :, packed into chunks of at most 50 tokens. Tests compare with the Python
// (pocket-text.test.ts).

import type { PhononChunk, PhononTokenizer } from './phonon-text';

const MAX_TOKENS = 50;
const REPLACE: Record<string, string> = { '’': "'", '‘': "'" };
const TERMINAL = '.!?…';
const WEAK = ',;:-–—';
const CLOSERS = '"\'”’)]»';

/** prepare_text_prompt: the text as the model reads it, and the frames to keep after the end (3 for ≤ 4 words, else 1). */
export function preparePocketText(input: string): [string, number] {
  let text = input.trim();
  text = Array.from(text, (c) => REPLACE[c] ?? c).join('');
  text = text.split(/\s+/u).filter(Boolean).join(' ');
  // Deleted quotes leave '"Hi?", she said' as 'Hi?, she said': keep the sentence mark only.
  text = text.replace(/([.!?…])\s*[,;:]/gu, '$1');
  if (text === '') throw new Error('Nothing to synthesize: the text is empty.');
  text = text.replaceAll('\n', ' ').replaceAll('\r', ' ').replaceAll('  ', ' ');
  const framesAfterEos = words(text).length <= 4 ? 3 : 1;
  const first = String.fromCodePoint(text.codePointAt(0)!);
  if (!isUpper(first)) text = first.toUpperCase() + text.slice(first.length);
  return [terminalPunctuation(text), framesAfterEos];
}

/** Python's str.isupper() for one character: cased, and upper case. */
function isUpper(c: string): boolean {
  return c !== c.toLowerCase() && c === c.toUpperCase();
}

function words(text: string): string[] {
  return text.split(/\s+/u).filter(Boolean);
}

function rstrip(text: string, chars: string): string {
  let end = text.length;
  while (end > 0 && chars.includes(text[end - 1]!)) end--;
  return text.slice(0, end);
}

/** A full stop at the end unless it ends a sentence already; a trailing comma, colon or dash becomes one. */
function terminalPunctuation(text: string): string {
  const core = rstrip(text, CLOSERS + ' ');
  const closers = text.slice(core.length).trim();
  if (!core || TERMINAL.includes(core[core.length - 1]!)) return text;
  if (WEAK.includes(core[core.length - 1]!)) return `${rstrip(core, WEAK + ' ')}.${closers}`;
  return `${text}.`;
}

function boundaries(tokens: number[], marks: Set<number>, tokenizer: PhononTokenizer, skipDecimal: boolean): number[] {
  const indices = [0];
  let previous = false;
  tokens.forEach((t, i) => {
    if (marks.has(t)) {
      previous = true;
      return;
    }
    if (previous) {
      const decimal = skipDecimal && isDecimalPoint(tokens, i, tokenizer);
      if (!decimal) indices.push(i);
    }
    previous = false;
  });
  indices.push(tokens.length);
  return indices;
}

/** Whether index `i` starts right after a decimal point ("3.5"). */
function isDecimalPoint(tokens: number[], i: number, tokenizer: PhononTokenizer): boolean {
  const prefix = tokenizer.decode(tokens.slice(0, i));
  const suffix = tokenizer.decode(tokens.slice(i));
  return prefix.length >= 2 && prefix.endsWith('.') && /\p{Nd}/u.test(prefix[prefix.length - 2]!) && /^\p{Nd}/u.test(suffix);
}

function segments(tokens: number[], bounds: number[], tokenizer: PhononTokenizer): [number, string][] {
  return bounds.slice(0, -1).map((start, i) => [bounds[i + 1]! - start, tokenizer.decode(tokens.slice(start, bounds[i + 1]))]);
}

/** split_into_best_sentences: the prepared text's sentences (long ones split at , ; :) packed into chunks of ≤ 50 tokens. */
export function splitPocketText(text: string, tokenizer: PhononTokenizer): string[] {
  const prepared = preparePocketText(text)[0].trim();
  const tokens = tokenizer.encode(prepared);
  const ends = new Set(tokenizer.encode('.!...?').slice(1));
  const sentences = segments(tokens, boundaries(tokens, ends, tokenizer, true), tokenizer);
  const clauseMarks = new Set(tokenizer.encode(',;:').slice(1));
  const refined: [number, string][] = [];
  for (const [n, sentence] of sentences) {
    if (n <= MAX_TOKENS) {
      refined.push([n, sentence]);
      continue;
    }
    const sub = tokenizer.encode(sentence.trim());
    const parts = segments(sub, boundaries(sub, clauseMarks, tokenizer, false), tokenizer);
    if (parts.length > 1) refined.push(...parts);
    else refined.push([n, sentence]);
  }
  const chunks: string[] = [];
  let chunk = '';
  let count = 0;
  for (const [n, sentence] of refined) {
    if (chunk === '') [chunk, count] = [sentence, n];
    else if (count + n > MAX_TOKENS) {
      chunks.push(chunk.trim());
      [chunk, count] = [sentence, n];
    } else {
      chunk += ` ${sentence}`;
      count += n;
    }
  }
  if (chunk !== '') chunks.push(chunk.trim());
  return chunks;
}

/**
 * The chunks the model speaks one at a time: each prepared again, its token ids, the frames to keep after its end
 * (the guess plus 2) and its frame budget, ⌈(tokens / 3 + 2) × 12.5⌉.
 */
export function pocketChunks(text: string, tokenizer: PhononTokenizer): PhononChunk[] {
  return splitPocketText(text, tokenizer).map((c) => {
    const [prepared, guess] = preparePocketText(c);
    const ids = tokenizer.encode(prepared);
    return { ids, framesAfterEos: guess + 2, frameBudget: Math.ceil((ids.length / 3 + 2) * 12.5) };
  });
}
