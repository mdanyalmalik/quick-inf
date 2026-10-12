// gpt-oss replies in Harmony "channels": `<|channel|>analysis<|message|>…<|end|>` (its reasoning), then
// `<|start|>assistant<|channel|>final<|message|>…` (the answer). The control tokens are dropped from the text
// like any special token, so this turns the stream into the `<think>…</think>answer` form the app shows
// reasoning in: analysis and commentary messages become thinking, the final one the answer.

import type { BpeTokenizer } from './bpe-tokenizer';
import { StreamingDecoder } from './streaming-decoder';

export class HarmonyStream {
  private readonly decoder: StreamingDecoder;
  private readonly start: number;
  private readonly channel: number;
  private readonly message: number;
  private readonly end: number;
  /** Where in a message the stream is: its role or channel name (both hidden), or its text. */
  private state: 'role' | 'channel' | 'text' = 'role';
  private channelIds: number[] = [];
  private thinking = false;

  constructor(private readonly tokenizer: BpeTokenizer) {
    this.decoder = new StreamingDecoder(tokenizer);
    this.start = tokenizer.specialTokenId('<|start|>');
    this.channel = tokenizer.specialTokenId('<|channel|>');
    this.message = tokenizer.specialTokenId('<|message|>');
    this.end = tokenizer.specialTokenId('<|end|>');
  }

  push(id: number): string {
    if (id === this.start || id === this.end) {
      this.state = 'role';
      return this.decoder.flush();
    }
    if (id === this.channel) {
      this.state = 'channel';
      this.channelIds = [];
      return '';
    }
    if (id === this.message) {
      this.state = 'text';
      const final = this.tokenizer.decode(this.channelIds, { skipSpecialTokens: true }).trim().startsWith('final');
      if (final && this.thinking) {
        this.thinking = false;
        return '</think>';
      }
      if (!final && !this.thinking) {
        this.thinking = true;
        return '<think>';
      }
      return '';
    }
    if (this.state === 'channel') this.channelIds.push(id);
    return this.state === 'text' ? this.decoder.push(id) : '';
  }

  flush(): string {
    return this.decoder.flush() + (this.thinking ? '</think>' : '');
  }
}
