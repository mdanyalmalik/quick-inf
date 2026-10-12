// CLIP's tokenizer (Stable Diffusion's text encoder), from its vocab.json and merges.txt, matching Hugging Face's
// (fast) CLIPTokenizer, as diffusers uses it (checked by clip-tokenizer.test.ts):
//
// 1. "<|startoftext|>", "<|endoftext|>" and, where it is the pad token (SD 2.x), "!" in the text are those tokens.
// 2. The rest: NFC, runs of whitespace to one space, lowercased; CLIP's split pattern; each piece's UTF-8 bytes
//    as GPT-2 byte characters, then BPE with "</w>" marking the end of a word.
// 3. <|startoftext|> … <|endoftext|>, cut to 77 and padded with the pad token ("!" for SD 2.x, <|endoftext|> for
//    SD 1.x).

import { textToByteChars } from './byte-level';

const START = '<|startoftext|>';
const END = '<|endoftext|>';
const BANG = '!';
const PATTERN = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;

export class ClipTokenizer {
  private readonly ranks = new Map<string, number>();
  private readonly cache = new Map<string, number[]>();
  readonly startId: number;
  readonly endId: number;
  readonly padId: number;
  private readonly special: RegExp;

  constructor(
    private readonly vocab: Record<string, number>,
    merges: string | string[],
    pad: '!' | '<|endoftext|>' = BANG,
    readonly maxLength = 77,
  ) {
    const lines = typeof merges === 'string' ? merges.split('\n').filter((l) => l && !l.startsWith('#version')) : merges;
    lines.forEach((line, rank) => this.ranks.set(line, rank));
    this.startId = vocab[START]!;
    this.endId = vocab[END]!;
    this.padId = vocab[pad]!;
    this.special = pad === BANG ? /(<\|startoftext\|>|<\|endoftext\|>|!)/ : /(<\|startoftext\|>|<\|endoftext\|>)/;
  }

  /** From a tokenizer.json (its BPE model's vocab and merges). */
  static fromJSON(json: unknown, pad: '!' | '<|endoftext|>'): ClipTokenizer {
    const model = (json as { model: { vocab: Record<string, number>; merges: (string | [string, string])[] } }).model;
    return new ClipTokenizer(
      model.vocab,
      model.merges.map((m) => (typeof m === 'string' ? m : m.join(' '))),
      pad,
    );
  }

  /** The text encoder's input: start, the text's tokens, end, padded to `maxLength`. */
  encode(text: string): number[] {
    const ids = [this.startId];
    const normalized = text.normalize('NFC').replace(/\s+/g, ' ').toLowerCase();
    for (const part of normalized.split(this.special)) {
      if (part === START) ids.push(this.startId);
      else if (part === END) ids.push(this.endId);
      else if (part === BANG && this.padId === this.vocab[BANG]) ids.push(this.padId);
      else if (part) ids.push(...this.encodeText(part));
    }
    const body = ids.slice(0, this.maxLength - 1);
    body.push(this.endId);
    while (body.length < this.maxLength) body.push(this.padId);
    return body;
  }

  private encodeText(text: string): number[] {
    const ids: number[] = [];
    for (const match of text.matchAll(PATTERN)) {
      const word = match[0];
      if (word === START || word === END) {
        ids.push(word === START ? this.startId : this.endId);
        continue;
      }
      ids.push(...this.bpe(textToByteChars(word)));
    }
    return ids;
  }

  private bpe(word: string): number[] {
    const cached = this.cache.get(word);
    if (cached) return cached;
    const chars = Array.from(word);
    let symbols = [...chars.slice(0, -1), `${chars[chars.length - 1]}</w>`];
    while (symbols.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      const a = symbols[best]!;
      const b = symbols[best + 1]!;
      const merged: string[] = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else merged.push(symbols[i]!);
      }
      symbols = merged;
    }
    const ids = symbols.map((s) => this.vocab[s] ?? this.endId);
    if (this.cache.size > 10_000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }
}
