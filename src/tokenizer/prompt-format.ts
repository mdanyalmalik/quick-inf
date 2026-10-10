// Prompt formatting for the ONNX engine, matching each model's own Jinja chat
// template (checked against them by tokenizer.test.ts). Built as token ids, so
// message text can never inject control tokens: each format is a list of parts,
// control tokens by name and text encoded on its own, as Hugging Face encodes the
// text between control tokens.
//
//   chatml  <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n
//           (SmolLM2, Qwen). Qwen3 with thinking off appends an empty
//           `<think>\n\n</think>\n\n` block to the generation prompt.
//   llama3  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
//           Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n{system}<|eot_id|>
//           <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|> ...
//           (always a system block; contents trimmed).
//   phi     <|{role}|>{content}<|end|> ... <|assistant|> (Phi-3.5 and Phi-4). Their
//           templates put "\n" after each control token, but those tokens strip the
//           whitespace that follows them (rstrip), so both come out the same.

import type { BpeTokenizer } from './bpe-tokenizer';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export type PromptFormatKind = 'chatml' | 'llama3' | 'phi';

export interface PromptFormatOptions {
  /** Inserted when the first turn is not a system turn, as the model's template does (chatml). */
  defaultSystemPrompt?: string;
  /** Qwen3-style: pre-fill an empty thinking block so the model answers directly (chatml). */
  disableThinking?: boolean;
  /** Trim each message, as Qwen3.5's template does (chatml; llama3 always trims). */
  trimContent?: boolean;
  /** llama3: the "Today Date" line, e.g. "10 Oct 2026"; today by default. */
  date?: string;
}

/** A control token (by its text) or plain text. */
type Part = { special: string } | string;

export interface PromptFormatter {
  /** Ids for a full conversation followed by the assistant generation prompt. */
  format(turns: readonly ChatTurn[]): number[];
  /** Token count of one turn, including its control tokens. */
  turnLength(turn: ChatTurn): number;
  /** Tokens outside the turns: start of prompt, any implicit system turn, and the generation prompt. */
  fixedLength(hasSystemTurn: boolean): number;
  /** Tokens of the generation prompt that ends every formatted conversation. */
  readonly generationLength: number;
  /** A token id that ends an assistant turn. */
  readonly endOfTurnId: number;
}

export function createPromptFormatter(kind: PromptFormatKind, tokenizer: BpeTokenizer, options: PromptFormatOptions = {}): PromptFormatter {
  return new PartsFormatter(tokenizer, SPECS[kind](options));
}

interface FormatSpec {
  /** Parts before the turns; `implicitSystem` when the conversation has no system turn. */
  start(implicitSystem: boolean): Part[];
  turn(turn: ChatTurn): Part[];
  generation: Part[];
  endOfTurn: string;
}

const SPECS: Record<PromptFormatKind, (o: PromptFormatOptions) => FormatSpec> = {
  chatml: (o) => {
    const turn = (t: ChatTurn): Part[] => [{ special: '<|im_start|>' }, `${t.role}\n${o.trimContent ? t.content.trim() : t.content}`, { special: '<|im_end|>' }, '\n'];
    return {
      start: (implicit) => (implicit && o.defaultSystemPrompt !== undefined ? turn({ role: 'system', content: o.defaultSystemPrompt }) : []),
      turn,
      generation: [
        { special: '<|im_start|>' },
        'assistant\n',
        ...(o.disableThinking ? [{ special: '<think>' }, '\n\n', { special: '</think>' }, '\n\n'] : []),
      ],
      endOfTurn: '<|im_end|>',
    };
  },
  llama3: (o) => {
    const header = (role: string): Part[] => [{ special: '<|start_header_id|>' }, role, { special: '<|end_header_id|>' }];
    const system = (content: string): Part[] => [
      ...header('system'),
      `\n\nCutting Knowledge Date: December 2023\nToday Date: ${o.date ?? today()}\n\n${content.trim()}`,
      { special: '<|eot_id|>' },
    ];
    return {
      start: (implicit) => [{ special: '<|begin_of_text|>' }, ...(implicit ? system('') : [])],
      turn: (t) => (t.role === 'system' ? system(t.content) : [...header(t.role), `\n\n${t.content.trim()}`, { special: '<|eot_id|>' }]),
      generation: [...header('assistant'), '\n\n'],
      endOfTurn: '<|eot_id|>',
    };
  },
  phi: () => ({
    start: () => [],
    // The templates skip an empty system message.
    turn: (t) => (t.role === 'system' && !t.content ? [] : [{ special: `<|${t.role}|>` }, t.content.trimStart(), { special: '<|end|>' }]),
    generation: [{ special: '<|assistant|>' }],
    endOfTurn: '<|end|>',
  }),
};

/** The date as Llama's template writes it ("%d %b %Y"). */
function today(): string {
  const d = new Date();
  const month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getMonth()];
  return `${String(d.getDate()).padStart(2, '0')} ${month} ${d.getFullYear()}`;
}

class PartsFormatter implements PromptFormatter {
  readonly endOfTurnId: number;
  private readonly generation: number[];

  constructor(
    private readonly tokenizer: BpeTokenizer,
    private readonly spec: FormatSpec,
  ) {
    this.endOfTurnId = tokenizer.specialTokenId(spec.endOfTurn);
    this.generation = this.encode(spec.generation);
  }

  format(turns: readonly ChatTurn[]): number[] {
    const ids = this.encode(this.spec.start(turns[0]?.role !== 'system'));
    for (const turn of turns) ids.push(...this.encode(this.spec.turn(turn)));
    ids.push(...this.generation);
    return ids;
  }

  turnLength(turn: ChatTurn): number {
    return this.encode(this.spec.turn(turn)).length;
  }

  get generationLength(): number {
    return this.generation.length;
  }

  fixedLength(hasSystemTurn: boolean): number {
    return this.encode(this.spec.start(!hasSystemTurn)).length + this.generation.length;
  }

  private encode(parts: Part[]): number[] {
    const ids: number[] = [];
    for (const part of parts) {
      if (typeof part !== 'string') ids.push(this.tokenizer.specialTokenId(part.special));
      else if (part) ids.push(...this.tokenizer.encode(part));
    }
    return ids;
  }
}
