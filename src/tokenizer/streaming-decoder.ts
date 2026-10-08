import type { BpeTokenizer } from './bpe-tokenizer';

/**
 * Turns a stream of token ids into text deltas. A multi-byte character can be
 * split across tokens, so text ending in U+FFFD is held back until the rest of
 * the character arrives. Only the ids since the last clean boundary are
 * re-decoded, keeping the cost per token small.
 */
export class StreamingDecoder {
  private pending: number[] = [];
  private emitted = 0;

  constructor(private readonly tokenizer: BpeTokenizer) {}

  push(id: number): string {
    this.pending.push(id);
    const text = this.tokenizer.decode(this.pending, { skipSpecialTokens: true });
    if (text.endsWith('�')) return '';
    const delta = text.slice(this.emitted);
    if (/\s$/.test(text)) {
      this.pending = [];
      this.emitted = 0;
    } else {
      this.emitted = text.length;
    }
    return delta;
  }

  /** Emits anything still held back (e.g. a genuinely invalid byte sequence at the end). */
  flush(): string {
    const text = this.tokenizer.decode(this.pending, { skipSpecialTokens: true });
    const delta = text.slice(this.emitted);
    this.pending = [];
    this.emitted = 0;
    return delta;
  }
}
