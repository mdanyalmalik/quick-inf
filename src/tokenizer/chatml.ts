// ChatML prompt formatting, matching the Jinja chat template shipped with
// SmolLM2 (and Qwen-style models):
//
//   <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n
//
// Built directly as token ids so message text can never inject control tokens.

import type { BpeTokenizer } from './bpe-tokenizer';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface ChatMlOptions {
  /** Inserted when the first turn is not a system turn, as the model's template does. */
  defaultSystemPrompt?: string;
}

export class ChatMlFormatter {
  private readonly imStart: number;
  private readonly imEnd: number;
  private readonly newline: number[];

  constructor(
    private readonly tokenizer: BpeTokenizer,
    private readonly options: ChatMlOptions = {},
  ) {
    this.imStart = tokenizer.specialTokenId('<|im_start|>');
    this.imEnd = tokenizer.specialTokenId('<|im_end|>');
    this.newline = tokenizer.encode('\n');
  }

  /** Token id that ends an assistant turn. */
  get endOfTurnId(): number {
    return this.imEnd;
  }

  /** Ids for a full conversation followed by the assistant generation prompt. */
  format(turns: readonly ChatTurn[]): number[] {
    const ids: number[] = [];
    const first = turns[0];
    if (this.options.defaultSystemPrompt !== undefined && (!first || first.role !== 'system')) {
      this.pushTurn(ids, { role: 'system', content: this.options.defaultSystemPrompt });
    }
    for (const turn of turns) this.pushTurn(ids, turn);
    ids.push(this.imStart, ...this.tokenizer.encode('assistant\n'));
    return ids;
  }

  /** Token count of one turn, including its control tokens and trailing newline. */
  turnLength(turn: ChatTurn): number {
    return this.tokenizer.encode(`${turn.role}\n${turn.content}`).length + 2 + this.newline.length;
  }

  private pushTurn(ids: number[], turn: ChatTurn): void {
    ids.push(this.imStart, ...this.tokenizer.encode(`${turn.role}\n${turn.content}`), this.imEnd, ...this.newline);
  }
}
