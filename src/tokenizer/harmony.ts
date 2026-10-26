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

/**
 * Muse Glimmer addresses each message it writes: `<|start|>assistant to=self<|message|>…<|eom|>` is its reasoning, then
 * `<|start|>assistant to=user<|message|>…<|eot|>` the answer (the generation prompt ends at `<|start|>assistant`, so
 * the first header is just ` to=self`). As HarmonyStream does, this turns them into the `<think>…</think>answer` form:
 * messages to the user are the answer, any others thinking.
 */
export class MuseGlimmerStream {
  private readonly decoder: StreamingDecoder;
  private readonly start: number;
  private readonly message: number;
  private readonly ends: Set<number>;
  private header = true;
  private headerIds: number[] = [];
  private thinking = false;

  constructor(private readonly tokenizer: BpeTokenizer) {
    this.decoder = new StreamingDecoder(tokenizer);
    this.start = tokenizer.specialTokenId('<|start|>');
    this.message = tokenizer.specialTokenId('<|message|>');
    this.ends = new Set(['<|eom|>', '<|eot|>'].map((t) => tokenizer.specialTokenId(t)));
  }

  push(id: number): string {
    if (id === this.start || this.ends.has(id)) {
      this.header = true;
      this.headerIds = [];
      return this.decoder.flush();
    }
    if (id === this.message) {
      this.header = false;
      const answer = /\bto=user\b/.test(this.tokenizer.decode(this.headerIds, { skipSpecialTokens: true }));
      if (answer && this.thinking) {
        this.thinking = false;
        return '</think>';
      }
      if (!answer && !this.thinking) {
        this.thinking = true;
        return '<think>';
      }
      return '';
    }
    if (this.header) {
      this.headerIds.push(id);
      return '';
    }
    return this.decoder.push(id);
  }

  flush(): string {
    return this.decoder.flush() + (this.thinking ? '</think>' : '');
  }
}

/**
 * Gemma 4 with thinking on writes its reasoning in a thought channel first: `<|channel>thought\n…<channel|>`, then the
 * answer. As HarmonyStream does, this turns it into the `<think>…</think>answer` form (the channel's name line is
 * dropped).
 */
export class Gemma4Stream {
  private readonly decoder: StreamingDecoder;
  private readonly open: number;
  private readonly close: number;
  /** Where the stream is: the channel's name line (hidden), or text. */
  private state: 'name' | 'text' = 'text';
  private nameIds: number[] = [];
  private thinking = false;

  constructor(private readonly tokenizer: BpeTokenizer) {
    this.decoder = new StreamingDecoder(tokenizer);
    this.open = tokenizer.specialTokenId('<|channel>');
    this.close = tokenizer.specialTokenId('<channel|>');
  }

  push(id: number): string {
    if (id === this.open) {
      this.state = 'name';
      this.nameIds = [];
      return this.decoder.flush();
    }
    if (id === this.close) {
      this.state = 'text';
      const text = this.decoder.flush() + (this.thinking ? '</think>' : '');
      this.thinking = false;
      return text;
    }
    if (this.state === 'text') return this.decoder.push(id);
    this.nameIds.push(id);
    const name = this.tokenizer.decode(this.nameIds, { skipSpecialTokens: true });
    const newline = name.indexOf('\n');
    if (newline < 0) return '';
    this.state = 'text';
    const opened = this.thinking ? '' : '<think>';
    this.thinking = true;
    return opened + name.slice(newline + 1);
  }

  flush(): string {
    return this.decoder.flush() + (this.thinking ? '</think>' : '');
  }
}
