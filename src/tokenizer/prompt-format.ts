// Prompt formatting for the ONNX engine, matching each model's own Jinja chat
// template (checked against them by tokenizer.test.ts). Built as token ids, so
// message text can never inject control tokens: each format is a list of parts,
// control tokens by name and text encoded on its own, as Hugging Face encodes the
// text between control tokens.
//
//   chatml  <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n
//           (SmolLM2, Qwen, LFM2, OLMo 3). Qwen3 with thinking off appends an empty
//           `<think>\n\n</think>\n\n` block to the generation prompt. LFM2 starts
//           with a begin token (`beginToken`).
//   llama3  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
//           Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n{system}<|eot_id|>
//           <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|> ...
//           (always a system block; contents trimmed).
//   phi     <|{role}|>{content}<|end|> ... <|assistant|> (Phi-3.5 and Phi-4). Their
//           templates put "\n" after each control token, but those tokens strip the
//           whitespace that follows them (rstrip), so both come out the same.
//   gemma   <bos><start_of_turn>user\n{system}\n\n{content}<end_of_turn>\n
//           <start_of_turn>model\n{content}<end_of_turn>\n ... <start_of_turn>model\n
//           (Gemma 3): no system turn; its text opens the first user turn. Contents trimmed.
//   granite <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n ...
//           <|start_of_role|>assistant<|end_of_role|> (Granite 4); always a system turn
//           (the default one when there's none or it's empty), except Micro's (no default).
//   phi4    <|im_start|>{role}<|im_sep|>{content}<|im_end|> ... <|im_start|>assistant<|im_sep|>
//           (Phi-4 14B). Phi-4-reasoning always opens with its own system prompt and drops
//           the conversation's (`fixedSystemPrompt`).
//   deepseek <｜begin▁of▁sentence｜>{system}<｜User｜>{content}<｜Assistant｜>{content}<｜end▁of▁sentence｜>
//           ... <｜Assistant｜> (DeepSeek-R1 distills).
//   mistral <s>[INST] {system}\n\n{content} [/INST]{content}</s>[INST] ... [/INST]
//           (Mistral 7B v0.2): no system turn; its text opens the first user turn.

import type { BpeTokenizer } from './bpe-tokenizer';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export type PromptFormatKind = 'chatml' | 'llama3' | 'phi' | 'gemma' | 'granite' | 'phi4' | 'deepseek' | 'mistral';

export interface PromptFormatOptions {
  /** Inserted when the first turn is not a system turn, as the model's template does (chatml). */
  defaultSystemPrompt?: string;
  /** Qwen3-style: pre-fill an empty thinking block so the model answers directly (chatml). */
  disableThinking?: boolean;
  /** Trim each message, as Qwen3.5's template does (chatml; llama3 always trims). */
  trimContent?: boolean;
  /** llama3: the "Today Date" line, e.g. "10 Oct 2026"; today by default. */
  date?: string;
  /** chatml, phi: a control token that starts every prompt (LFM2's `<|startoftext|>`, Phi-3's `<s>`). */
  beginToken?: string;
  /** phi: the template drops system messages (Phi-3). */
  noSystem?: boolean;
  /** phi: every prompt's system turn starts with this text (Phi-4-mini-reasoning); the conversation's follows after a space. */
  systemPrefix?: string;
  /** phi4: the system turn every prompt has, in place of the conversation's (Phi-4-reasoning). */
  fixedSystemPrompt?: string;
}

/**
 * A control token (by its text), plain text, or text the template itself writes
 * (a default system prompt), where added tokens such as OLMo's `<functions>` count.
 */
type Part = { special: string } | { template: string } | { continued: string } | string;

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
  /** `prefix`: text a preceding system turn carries into this one (see `systemAsPrefix`). */
  turn(turn: ChatTurn, prefix?: string): Part[];
  /** The system turn has no turn of its own: this text opens the next turn (Gemma). */
  systemAsPrefix?: (content: string) => string;
  generation: Part[];
  endOfTurn: string;
}

const SPECS: Record<PromptFormatKind, (o: PromptFormatOptions) => FormatSpec> = {
  chatml: (o) => {
    const turn = (t: ChatTurn): Part[] => [{ special: '<|im_start|>' }, `${t.role}\n${o.trimContent ? t.content.trim() : t.content}`, { special: '<|im_end|>' }, '\n'];
    return {
      start: (implicit) => [
        ...(o.beginToken ? [{ special: o.beginToken }] : []),
        ...(implicit && o.defaultSystemPrompt !== undefined
          ? [{ special: '<|im_start|>' }, 'system\n', { template: o.defaultSystemPrompt }, { special: '<|im_end|>' }, '\n']
          : []),
      ],
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
  phi: (o) => {
    const system = (content: string): Part[] => [{ special: '<|system|>' }, { template: o.systemPrefix ?? '' }, content ? ` ${content}` : '', { special: '<|end|>' }];
    return {
      start: (implicit) => [...(o.beginToken ? [{ special: o.beginToken }] : []), ...(implicit && o.systemPrefix ? system('') : [])],
      turn: (t) => {
        if (t.role === 'system' && o.systemPrefix) return system(t.content);
        // The templates skip an empty system message; Phi-3's skips every one (`noSystem`).
        if (t.role === 'system' && (!t.content || o.noSystem)) return [];
        return [{ special: `<|${t.role}|>` }, t.content.trimStart(), { special: '<|end|>' }];
      },
      generation: [{ special: '<|assistant|>' }],
      endOfTurn: '<|end|>',
    };
  },
  phi4: (o) => {
    const turn = (role: string, content: Part): Part[] => [{ special: '<|im_start|>' }, role, { special: '<|im_sep|>' }, content, { special: '<|im_end|>' }];
    return {
      start: () => (o.fixedSystemPrompt ? turn('system', { template: o.fixedSystemPrompt }) : []),
      turn: (t) => (t.role === 'system' && o.fixedSystemPrompt ? [] : turn(t.role, t.content)),
      generation: [{ special: '<|im_start|>' }, 'assistant', { special: '<|im_sep|>' }],
      endOfTurn: '<|im_end|>',
    };
  },
  deepseek: () => ({
    start: () => [{ special: '<｜begin▁of▁sentence｜>' }],
    turn: (t) =>
      t.role === 'system'
        ? [t.content]
        : t.role === 'user'
          ? [{ special: '<｜User｜>' }, t.content]
          : [{ special: '<｜Assistant｜>' }, t.content, { special: '<｜end▁of▁sentence｜>' }],
    generation: [{ special: '<｜Assistant｜>' }],
    endOfTurn: '<｜end▁of▁sentence｜>',
  }),
  mistral: () => ({
    start: () => [{ special: '<s>' }],
    // A reply follows "[/INST]" directly, in the same run of text (no leading "▁").
    turn: (t, prefix = '') => (t.role === 'user' ? [`[INST] ${prefix}${t.content} [/INST]`] : [{ continued: t.content }, { special: '</s>' }]),
    systemAsPrefix: (content) => `${content}\n\n`,
    // The user turn already ends with the prompt for a reply.
    generation: [],
    endOfTurn: '</s>',
  }),
  gemma: () => ({
    start: () => [{ special: '<bos>' }],
    turn: (t, prefix = '') => [{ special: '<start_of_turn>' }, `${t.role === 'assistant' ? 'model' : t.role}\n${prefix}${t.content.trim()}`, { special: '<end_of_turn>' }, '\n'],
    systemAsPrefix: (content) => `${content}\n\n`,
    generation: [{ special: '<start_of_turn>' }, 'model\n'],
    endOfTurn: '<end_of_turn>',
  }),
  granite: (o) => {
    const turn = (t: ChatTurn): Part[] => {
      const content = t.role === 'system' && !t.content ? (o.defaultSystemPrompt ?? '') : t.content;
      return [{ special: '<|start_of_role|>' }, t.role, { special: '<|end_of_role|>' }, content, { special: '<|end_of_text|>' }, '\n'];
    };
    return {
      start: (implicit) => (implicit && o.defaultSystemPrompt !== undefined ? turn({ role: 'system', content: '' }) : []),
      turn,
      generation: [{ special: '<|start_of_role|>' }, 'assistant', { special: '<|end_of_role|>' }],
      endOfTurn: '<|end_of_text|>',
    };
  },
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
    let prefix: string | undefined;
    for (const turn of turns) {
      if (turn.role === 'system' && this.spec.systemAsPrefix) {
        prefix = this.spec.systemAsPrefix(turn.content);
        continue;
      }
      ids.push(...this.encode(this.spec.turn(turn, prefix)));
      prefix = undefined;
    }
    ids.push(...this.generation);
    return ids;
  }

  /** For a system turn that opens the next turn (Gemma): the length of its text there, near enough. */
  turnLength(turn: ChatTurn): number {
    if (turn.role === 'system' && this.spec.systemAsPrefix) return this.encode([this.spec.systemAsPrefix(turn.content)]).length;
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
      if (typeof part === 'string') {
        if (part) ids.push(...this.tokenizer.encode(part));
      } else if ('special' in part) {
        ids.push(this.tokenizer.specialTokenId(part.special));
      } else if ('continued' in part) {
        if (part.continued) ids.push(...this.tokenizer.encode(part.continued, { continuation: true }));
      } else {
        ids.push(...this.encodeTemplateText(part.template));
      }
    }
    return ids;
  }

  /** Text the template writes: added tokens in it count, and swallow whitespace next to them if they strip. */
  private encodeTemplateText(text: string): number[] {
    const pieces = text.split(/(<[^<>\s]+>)/);
    const ids = pieces.map((piece) => this.tokenizer.addedTokenId(piece));
    const out: number[] = [];
    pieces.forEach((piece, i) => {
      const id = ids[i];
      if (id !== undefined) {
        out.push(id);
        return;
      }
      const before = i > 0 ? ids[i - 1] : undefined;
      const after = i < pieces.length - 1 ? ids[i + 1] : undefined;
      let textPiece = piece;
      if (before !== undefined && this.tokenizer.addedTokenStrips(before).rstrip) textPiece = textPiece.trimStart();
      if (after !== undefined && this.tokenizer.addedTokenStrips(after).lstrip) textPiece = textPiece.trimEnd();
      if (textPiece) out.push(...this.tokenizer.encode(textPiece));
    });
    return out;
  }
}
