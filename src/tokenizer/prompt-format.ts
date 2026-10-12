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
//           ... <｜Assistant｜> (DeepSeek-R1 distills; newer templates add "<think>\n": `openThinking`).
//   mistral <s>[INST] {system}\n\n{content} [/INST]{content}</s>[INST] ... [/INST]
//           (Mistral 7B v0.2): no system turn; its text opens the first user turn.
//   mistral-tekken <s>[INST]{content}[/INST]{content}</s>[INST]{system}\n\n{content}[/INST]
//           (Mistral Nemo): [INST] and [/INST] are control tokens; the system text opens the last user turn.
//   mistral3 <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{content}[/INST]{content}</s> ...
//           (Ministral 3): without a system turn, Mistral's default system prompt (with today's and
//           yesterday's dates filled in, as mistral-common does).
//   gemma4  <bos><|turn>{role}\n{content}<turn|>\n ... <|turn>model\n (Gemma 4): a system turn of its own;
//           contents trimmed; thinking off.
//   smollm3 <|im_start|>system\n## Metadata ... ## Custom Instructions\n\n{system}\n\n (no <|im_end|>),
//           then chatml turns; thinking off: "Reasoning Mode: /no_think" and an empty think block
//           opening every reply (past ones too).
//   zephyr  <|system|>\n{system}</s>\n<|user|>\n{content}</s>\n<|assistant|>\n (TinyLlama): the role tags are text.
//   harmony <|start|>system<|message|>{model identity, dates, reasoning level}<|end|>
//           <|start|>developer<|message|># Instructions\n\n{system}<|end|> <|start|>user<|message|>{content}<|end|>
//           <|start|>assistant<|channel|>final<|message|>{content}<|end|> ... <|start|>assistant (gpt-oss).
//           Replies come in channels (analysis, then final): see HarmonyStream.

import type { BpeTokenizer } from './bpe-tokenizer';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export type PromptFormatKind =
  | 'chatml'
  | 'llama3'
  | 'phi'
  | 'gemma'
  | 'granite'
  | 'phi4'
  | 'deepseek'
  | 'mistral'
  | 'mistral-tekken'
  | 'mistral3'
  | 'gemma4'
  | 'smollm3'
  | 'zephyr'
  | 'harmony';

export interface PromptFormatOptions {
  /** Inserted when the first turn is not a system turn, as the model's template does (chatml). */
  defaultSystemPrompt?: string;
  /** Qwen3-style: pre-fill an empty thinking block so the model answers directly (chatml). */
  disableThinking?: boolean;
  /** Trim each message, as Qwen3.5's template does (chatml; llama3 always trims). */
  trimContent?: boolean;
  /**
   * Today's date for templates that state it (llama3 "10 Oct 2026", smollm3 "12 October 2026", mistral3 and
   * harmony "2026-10-12"), in the template's own form; today by default.
   */
  date?: string;
  /** chatml, phi: a control token that starts every prompt (LFM2's `<|startoftext|>`, Phi-3's `<s>`). */
  beginToken?: string;
  /** phi: the template drops system messages (Phi-3). */
  noSystem?: boolean;
  /** phi: every prompt's system turn starts with this text (Phi-4-mini-reasoning); the conversation's follows after a space. */
  systemPrefix?: string;
  /** phi4: the system turn every prompt has, in place of the conversation's (Phi-4-reasoning). */
  fixedSystemPrompt?: string;
  /** deepseek: the reply starts inside an opened think block, as newer R1 templates do. */
  openThinking?: boolean;
  /** harmony: how much the model reasons ("low", "medium", "high"). */
  reasoningEffort?: 'low' | 'medium' | 'high';
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
  /** The system turn has no turn of its own: this text opens the next turn (Gemma), or the last user turn (`systemToLastUser`). */
  systemAsPrefix?: (content: string) => string;
  /** The system text goes to the last user turn instead of the next one (Mistral Nemo). */
  systemToLastUser?: boolean;
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
  deepseek: (o) => ({
    start: () => [{ special: '<｜begin▁of▁sentence｜>' }],
    turn: (t) =>
      t.role === 'system'
        ? [t.content]
        : t.role === 'user'
          ? [{ special: '<｜User｜>' }, t.content]
          : [{ special: '<｜Assistant｜>' }, t.content, { special: '<｜end▁of▁sentence｜>' }],
    generation: [{ special: '<｜Assistant｜>' }, ...(o.openThinking ? [{ special: '<think>' }, '\n'] : [])],
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
  'mistral-tekken': () => ({
    start: () => [{ special: '<s>' }],
    turn: (t, prefix = '') =>
      t.role === 'user' ? [{ special: '[INST]' }, `${prefix}${t.content}`, { special: '[/INST]' }] : [t.content, { special: '</s>' }],
    systemAsPrefix: (content) => `${content}\n\n`,
    systemToLastUser: true,
    generation: [],
    endOfTurn: '</s>',
  }),
  mistral3: (o) => {
    const system = (content: Part): Part[] => [{ special: '[SYSTEM_PROMPT]' }, content, { special: '[/SYSTEM_PROMPT]' }];
    return {
      start: (implicit) => [{ special: '<s>' }, ...(implicit && o.defaultSystemPrompt ? system({ template: fillMistralDates(o.defaultSystemPrompt, o.date) }) : [])],
      turn: (t) =>
        t.role === 'system'
          ? system(t.content)
          : t.role === 'user'
            ? [{ special: '[INST]' }, t.content, { special: '[/INST]' }]
            : [t.content, { special: '</s>' }],
      generation: [],
      endOfTurn: '</s>',
    };
  },
  gemma4: () => ({
    start: () => [{ special: '<bos>' }],
    turn: (t) => [{ special: '<|turn>' }, `${t.role === 'assistant' ? 'model' : t.role}\n${t.content.trim()}`, { special: '<turn|>' }, '\n'],
    generation: [{ special: '<|turn>' }, 'model\n'],
    endOfTurn: '<turn|>',
  }),
  smollm3: (o) => {
    const system = (custom: string): Part[] => [
      { special: '<|im_start|>' },
      `system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: ${o.date ?? longDate()}\nReasoning Mode: /no_think\n\n## Custom Instructions\n\n${custom}\n\n`,
    ];
    const emptyThink: Part[] = [{ special: '<think>' }, '\n\n', { special: '</think>' }, '\n'];
    return {
      start: (implicit) => (implicit ? system(o.defaultSystemPrompt ?? '') : []),
      turn: (t) =>
        t.role === 'system'
          ? system(t.content)
          : [{ special: '<|im_start|>' }, `${t.role}\n`, ...(t.role === 'assistant' ? emptyThink : []), t.content, { special: '<|im_end|>' }, '\n'],
      generation: [{ special: '<|im_start|>' }, 'assistant\n', ...emptyThink],
      endOfTurn: '<|im_end|>',
    };
  },
  zephyr: () => ({
    start: () => [],
    turn: (t) => [`<|${t.role}|>\n${t.content}`, { special: '</s>' }, '\n'],
    generation: ['<|assistant|>\n'],
    endOfTurn: '</s>',
  }),
  harmony: (o) => {
    const message = (role: string, content: Part, channel?: string): Part[] => [
      { special: '<|start|>' },
      role,
      ...(channel ? [{ special: '<|channel|>' }, channel] : []),
      { special: '<|message|>' },
      content,
      { special: '<|end|>' },
    ];
    const identity = `You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: ${o.date ?? isoDate()}\n\nReasoning: ${o.reasoningEffort ?? 'medium'}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.`;
    return {
      start: () => message('system', identity),
      turn: (t) =>
        t.role === 'system'
          ? t.content
            ? message('developer', `# Instructions\n\n${t.content}`)
            : []
          : t.role === 'user'
            ? message('user', t.content)
            : message('assistant', t.content, 'final'),
      generation: [{ special: '<|start|>' }, 'assistant'],
      endOfTurn: '<|end|>',
    };
  },
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

/** Today as "2026-10-12" (mistral-common, Harmony). */
function isoDate(d = new Date()): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** Today as SmolLM3's template writes it ("%d %B %Y": "12 October 2026"). */
function longDate(d = new Date()): string {
  return `${String(d.getDate()).padStart(2, '0')} ${d.toLocaleString('en-GB', { month: 'long' })} ${d.getFullYear()}`;
}

/** Mistral's default system prompt names today and yesterday ({today}, {yesterday}), as "2026-10-12". */
export function fillMistralDates(text: string, date?: string): string {
  const today = date ? new Date(`${date}T12:00:00`) : new Date();
  const yesterday = new Date(today.getTime() - 86_400_000);
  return text.replaceAll('{today}', isoDate(today)).replaceAll('{yesterday}', isoDate(yesterday));
}

/** The date as Llama's template writes it ("%d %b %Y"). */
function today(): string {
  const d = new Date();
  const month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getMonth()];
  return `${String(d.getDate()).padStart(2, '0')} ${month} ${d.getFullYear()}`;
}

/** Adjacent plain-text parts as one (a template's text between two control tokens is encoded in one go). */
function mergeText(parts: Part[]): Part[] {
  const out: Part[] = [];
  for (const part of parts) {
    const last = out[out.length - 1];
    if (typeof part === 'string' && typeof last === 'string') out[out.length - 1] = last + part;
    else out.push(part);
  }
  return out;
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
    // One run of parts, so text that meets across turns is encoded together, as the template's output is.
    const parts = this.spec.start(turns[0]?.role !== 'system');
    const lastUser = turns.map((t) => t.role).lastIndexOf('user');
    const system = turns.find((t) => t.role === 'system');
    let prefix: string | undefined;
    turns.forEach((turn, i) => {
      if (turn.role === 'system' && this.spec.systemAsPrefix) {
        if (!this.spec.systemToLastUser) prefix = this.spec.systemAsPrefix(turn.content);
        return;
      }
      if (this.spec.systemToLastUser && i === lastUser && system) prefix = this.spec.systemAsPrefix!(system.content);
      parts.push(...this.spec.turn(turn, prefix));
      prefix = undefined;
    });
    parts.push(...this.spec.generation);
    return this.encode(parts);
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
    for (const part of mergeText(parts)) {
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
