import type { BpeTokenizer } from './bpe-tokenizer';

/**
 * Turns a stream of token ids into text deltas. A multi-byte character can be
 * split across tokens, so text ending in U+FFFD is held back until the rest of
 * the character arrives. Only the ids since the last clean boundary are
 * re-decoded, keeping the cost per token small; those after the first boundary
 * decode as a continuation (SentencePiece-style tokenizers drop the leading
 * space of a text's start only).
 */
export class StreamingDecoder {
  private pending: number[] = [];
  private emitted = 0;
  private continuation = false;

  constructor(private readonly tokenizer: BpeTokenizer) {}

  push(id: number): string {
    this.pending.push(id);
    const text = this.tokenizer.decode(this.pending, { skipSpecialTokens: true, continuation: this.continuation });
    if (text.endsWith('�')) return '';
    const delta = text.slice(this.emitted);
    if (/\s$/.test(text)) {
      this.pending = [];
      this.emitted = 0;
      this.continuation = true;
    } else {
      this.emitted = text.length;
    }
    return delta;
  }

  /** Emits anything still held back (e.g. a genuinely invalid byte sequence at the end). */
  flush(): string {
    const text = this.tokenizer.decode(this.pending, { skipSpecialTokens: true, continuation: this.continuation });
    const delta = text.slice(this.emitted);
    this.pending = [];
    this.emitted = 0;
    this.continuation = true;
    return delta;
  }
}
