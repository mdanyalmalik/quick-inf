// BPE tokenizer for Hugging Face `tokenizer.json` files, in two flavours:
//
// - Byte-level (GPT-2 style; SmolLM2, Qwen, Llama 3, Phi-4): an optional NFC
//   normalizer; a pre-tokenizer made of Digits (individual), Split (regex,
//   isolated, or inverted with the gaps removed) and ByteLevel steps; BPE
//   without dropout/unk/byte-fallback, optionally with `ignore_merges` (a
//   pre-token found whole in the vocabulary is one token); a ByteLevel decoder.
// - SentencePiece-style (Llama 2 style; Phi-3.5): spaces become "▁" and one is
//   prepended; no pre-tokenizer; characters missing from the vocabulary fall
//   back to byte tokens "<0xNN>"; decoding reverses that and drops the first
//   leading space. Gemma 3's variant prepends nothing and keeps that space.
//
// Post-processors only add BOS/EOS when asked to or adjust character offsets,
// never the ids `encode` returns, so ByteLevel and TemplateProcessing ones (and
// sequences of them) are accepted. Anything else throws instead of silently
// tokenizing differently from the reference.
//
// `encode` treats its input as plain text: strings like "<|im_end|>" inside a
// message are tokenized as ordinary characters, never as control tokens.
// Control tokens are added structurally by the prompt formatter.

import { byteCharsToBytes, textToByteChars } from './byte-level';

/** GPT-2 pre-tokenization pattern used by the ByteLevel pre-tokenizer. */
const BYTE_LEVEL_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const DIGIT = /\p{N}/u;

/** One pre-tokenizer step: splits each text segment into smaller segments. */
type SplitStep = (segment: string) => Iterable<string>;

const CACHE_LIMIT = 20_000;

type PreTokenizerJson = Record<string, unknown> & { type: string; pretokenizers?: PreTokenizerJson[] };

interface TokenizerJson {
  normalizer: unknown;
  post_processor: unknown;
  pre_tokenizer: PreTokenizerJson | null;
  decoder: { type: string } | null;
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: Array<string | [string, string]>;
    dropout?: number | null;
    unk_token?: string | null;
    byte_fallback?: boolean;
    continuing_subword_prefix?: string | null;
    end_of_word_suffix?: string | null;
    ignore_merges?: boolean;
  };
  added_tokens: Array<{ id: number; content: string; special: boolean; lstrip?: boolean; rstrip?: boolean }>;
}

export class BpeTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly idToToken: string[];
  private readonly mergeRanks: Map<string, number>;
  private readonly addedById: Map<number, { content: string; special: boolean; lstrip: boolean; rstrip: boolean }>;
  private readonly addedByContent: Map<string, number>;
  private readonly normalizeNfc: boolean;
  private readonly steps: SplitStep[];
  private readonly sentencePiece: SentencePieceStyle | null;
  private readonly ignoreMerges: boolean;
  private readonly cache = new Map<string, number[]>();
  private readonly utf8Decoder = new TextDecoder('utf-8', { fatal: false });

  private constructor(json: TokenizerJson, normalizeNfc: boolean, steps: SplitStep[], sentencePiece: SentencePieceStyle | null) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.idToToken = [];
    for (const [token, id] of this.vocab) this.idToToken[id] = token;

    this.mergeRanks = new Map();
    json.model.merges.forEach((merge, rank) => {
      const [a, b] = typeof merge === 'string' ? splitMerge(merge) : merge;
      this.mergeRanks.set(pairKey(a, b), rank);
    });

    this.addedById = new Map();
    this.addedByContent = new Map();
    for (const t of json.added_tokens) {
      this.addedById.set(t.id, { content: t.content, special: t.special, lstrip: t.lstrip === true, rstrip: t.rstrip === true });
      this.addedByContent.set(t.content, t.id);
    }
    this.normalizeNfc = normalizeNfc;
    this.steps = steps;
    this.sentencePiece = sentencePiece;
    this.ignoreMerges = json.model.ignore_merges === true;
  }

  static fromJSON(json: unknown): BpeTokenizer {
    const t = json as TokenizerJson;
    const m = t.model;
    const unsupported: string[] = [];
    if (m?.type !== 'BPE') unsupported.push(`model type ${m?.type}`);
    if (m?.dropout) unsupported.push('BPE dropout');
    if (m?.continuing_subword_prefix || m?.end_of_word_suffix) unsupported.push('subword affixes');
    if (!acceptedPostProcessor(t.post_processor)) unsupported.push(`post-processor ${(t.post_processor as { type?: string }).type}`);

    const style = sentencePieceStyle(t);
    if (style) {
      // The unk token is only reached for bytes missing from the vocabulary, which these have.
      if (unsupported.length > 0) throw new Error(`Unsupported tokenizer configuration: ${unsupported.join(', ')}`);
      return new BpeTokenizer(t, false, [], style);
    }

    if (m?.unk_token) unsupported.push('unk token');
    if (m?.byte_fallback) unsupported.push('byte fallback');
    const normalizer = t.normalizer as { type?: string } | null;
    if (normalizer != null && normalizer.type !== 'NFC') unsupported.push(`normalizer ${normalizer.type}`);
    // A sequence of just a ByteLevel decoder (LFM2) is the same decoder.
    const decoders = t.decoder?.type === 'Sequence' ? ((t.decoder as { decoders?: { type: string }[] }).decoders ?? []) : [t.decoder];
    if (decoders.length !== 1 || decoders[0]?.type !== 'ByteLevel') unsupported.push(`decoder ${t.decoder?.type}`);

    const pre = t.pre_tokenizer;
    const configured = pre?.type === 'Sequence' ? (pre.pretokenizers ?? []) : pre ? [pre] : [];
    const last = configured[configured.length - 1];
    if (!last || last.type !== 'ByteLevel' || last.add_prefix_space !== false) {
      unsupported.push('pre-tokenizer must end with ByteLevel (no prefix space)');
    }
    const steps: SplitStep[] = [];
    for (const step of configured) {
      const source = (step.pattern as { Regex?: string } | undefined)?.Regex;
      if (step.type === 'Digits' && step.individual_digits === true) {
        steps.push(splitIndividualDigits);
      } else if (step.type === 'Split' && typeof source === 'string' && step.behavior === 'Isolated' && step.invert === false) {
        steps.push(isolatedSplit(toJsRegex(source)));
      } else if (step.type === 'Split' && typeof source === 'string' && step.behavior === 'Removed' && step.invert === true) {
        // Inverted: the matches are the pieces, and the text between them is dropped.
        steps.push(matchesOnly(toJsRegex(source)));
      } else if (step.type === 'ByteLevel' && step === last) {
        if (step.use_regex === true) steps.push(isolatedSplit(BYTE_LEVEL_PATTERN));
      } else {
        unsupported.push(`pre-tokenizer step ${step.type}`);
      }
    }
    if (unsupported.length > 0) throw new Error(`Unsupported tokenizer configuration: ${unsupported.join(', ')}`);
    return new BpeTokenizer(t, normalizer?.type === 'NFC', steps, null);
  }

  get vocabSize(): number {
    return this.idToToken.length;
  }

  /** Id of an added token, if `content` is one. */
  addedTokenId(content: string): number | undefined {
    return this.addedByContent.get(content);
  }

  /** Whether an added token swallows the whitespace before (lstrip) and after (rstrip) it. */
  addedTokenStrips(id: number): { lstrip: boolean; rstrip: boolean } {
    const t = this.addedById.get(id);
    return { lstrip: t?.lstrip ?? false, rstrip: t?.rstrip ?? false };
  }

  /** Id of an added/special token such as "<|im_end|>". */
  specialTokenId(content: string): number {
    const id = this.addedByContent.get(content);
    if (id === undefined) throw new Error(`Unknown special token ${content}`);
    return id;
  }

  /**
   * Encodes plain text (no special-token parsing, no BOS/EOS added). A
   * SentencePiece-style tokenizer prepends "▁" to each text it encodes, as
   * Hugging Face does to each piece of text between control tokens.
   */
  encode(text: string, { continuation = false } = {}): number[] {
    if (this.sentencePiece) return this.encodeSentencePiece(text, continuation);
    const ids: number[] = [];
    for (const piece of this.preTokenize(text)) {
      for (const id of this.encodePiece(piece)) ids.push(id);
    }
    return ids;
  }

  /**
   * `continuation`: the ids follow earlier text (streamed replies), so a
   * SentencePiece-style tokenizer keeps their leading space.
   */
  decode(ids: readonly number[], { skipSpecialTokens = false, continuation = false } = {}): string {
    if (this.sentencePiece) return this.decodeSentencePiece(ids, skipSpecialTokens, continuation);
    let out = '';
    let bytes: number[] = [];
    const flush = () => {
      if (bytes.length === 0) return;
      out += this.utf8Decoder.decode(new Uint8Array(bytes));
      bytes = [];
    };
    for (const id of ids) {
      const added = this.addedById.get(id);
      if (added) {
        if (added.special && skipSpecialTokens) continue;
        flush();
        out += added.content;
        continue;
      }
      const token = this.idToToken[id];
      if (token === undefined) throw new Error(`Token id ${id} is out of range`);
      byteCharsToBytes(token, bytes);
    }
    flush();
    return out;
  }

  /** Normalizes, runs the pre-tokenizer steps in order, then maps each piece to byte chars. */
  private *preTokenize(text: string): Generator<string> {
    let segments = [this.normalizeNfc ? text.normalize('NFC') : text];
    for (const step of this.steps) segments = segments.flatMap((segment) => [...step(segment)]);
    for (const segment of segments) yield textToByteChars(segment);
  }

  private encodePiece(piece: string): number[] {
    const cached = this.cache.get(piece);
    if (cached) return cached;
    const whole = this.ignoreMerges ? this.vocab.get(piece) : undefined;
    const ids = whole !== undefined ? [whole] : this.merge(Array.from(piece));
    if (this.cache.size >= CACHE_LIMIT) this.cache.clear();
    this.cache.set(piece, ids);
    return ids;
  }

  /** `continuation`: the text continues earlier text without a control token between (no "▁" prepended). */
  private encodeSentencePiece(text: string, continuation: boolean): number[] {
    if (text === '') return [];
    const style = this.sentencePiece!;
    const ids: number[] = [];
    // Text splits before every "▁" that follows another character, except where some token
    // joins that character to a "▁" (Gemma's ">▁</"), so the result doesn't change.
    const spaced = text.replaceAll(' ', '▁');
    for (const word of (style.prepend && !continuation ? `▁${spaced}` : spaced).split(style.wordBoundary)) {
      const cached = this.cache.get(word);
      if (cached) {
        ids.push(...cached);
        continue;
      }
      // Characters missing from the vocabulary become their UTF-8 bytes, "<0xNN>".
      const symbols = Array.from(word).flatMap((ch) =>
        this.vocab.has(ch) ? [ch] : Array.from(utf8.encode(ch), (b) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`),
      );
      const wordIds = this.merge(symbols);
      if (this.cache.size >= CACHE_LIMIT) this.cache.clear();
      this.cache.set(word, wordIds);
      ids.push(...wordIds);
    }
    return ids;
  }

  private decodeSentencePiece(ids: readonly number[], skipSpecialTokens: boolean, continuation: boolean): string {
    let out = '';
    let bytes: number[] = [];
    const flush = () => {
      if (bytes.length === 0) return;
      out += this.utf8Decoder.decode(new Uint8Array(bytes));
      bytes = [];
    };
    for (const id of ids) {
      const added = this.addedById.get(id);
      if (added) {
        if (added.special && skipSpecialTokens) continue;
        flush();
        // Gemma's whitespace runs ("▁▁") are added tokens, decoded like the rest.
        out += added.special ? added.content : added.content.replaceAll('▁', ' ');
        continue;
      }
      const token = this.idToToken[id];
      if (token === undefined) throw new Error(`Token id ${id} is out of range`);
      const byte = /^<0x([0-9A-F]{2})>$/.exec(token);
      if (byte) bytes.push(parseInt(byte[1]!, 16));
      else for (const b of utf8.encode(token.replaceAll('▁', ' '))) bytes.push(b);
    }
    flush();
    return this.sentencePiece!.prepend && !continuation && out.startsWith(' ') ? out.slice(1) : out;
  }

  /** Applies the merges to `symbols` (lowest rank first) and maps the result to ids. */
  private merge(initial: string[]): number[] {
    let symbols = initial;
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.mergeRanks.get(pairKey(symbols[i]!, symbols[i + 1]!));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;

      // Merge every non-overlapping occurrence of the best pair, left to right.
      const a = symbols[bestIndex]!;
      const b = symbols[bestIndex + 1]!;
      const merged: string[] = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else {
          merged.push(symbols[i]!);
        }
      }
      symbols = merged;
    }

    return symbols.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`Symbol ${JSON.stringify(s)} missing from vocabulary`);
      return id;
    });
  }
}

const utf8 = new TextEncoder();

/** Post-processors that never change the ids of `encode` (see the header). */
function acceptedPostProcessor(post: unknown): boolean {
  if (post == null) return true;
  const p = post as { type?: string; processors?: unknown[] };
  if (p.type === 'ByteLevel' || p.type === 'TemplateProcessing') return true;
  return p.type === 'Sequence' && (p.processors ?? []).every(acceptedPostProcessor);
}

interface SentencePieceStyle {
  /** Llama 2 style: a "▁" is prepended to each text, and decoding drops the first leading space. */
  prepend: boolean;
  /** Where text can be split into words without changing the result. */
  wordBoundary: RegExp;
}

const REPLACE_SPACE = { type: 'Replace', pattern: { String: ' ' }, content: '▁' };
const SP_DECODERS = [{ type: 'Replace', pattern: { String: '▁' }, content: ' ' }, { type: 'ByteFallback' }, { type: 'Fuse' }];

/** The SentencePiece BPE layouts, exactly (see the header): Llama 2 style, or Gemma's. */
function sentencePieceStyle(t: TokenizerJson): SentencePieceStyle | null {
  const json = (value: unknown) => JSON.stringify(value);
  const n = t.normalizer as { type?: string; normalizers?: Record<string, unknown>[] } | null;
  const d = t.decoder as { type?: string; decoders?: Record<string, unknown>[] } | null;
  // Gemma 2's Split on " " is a no-op: the normalizer has already turned spaces into "▁".
  const noOpSplit = json(t.pre_tokenizer) === json({ type: 'Split', pattern: { String: ' ' }, behavior: 'MergedWithPrevious', invert: false });
  if (t.model?.byte_fallback !== true || (t.pre_tokenizer != null && !noOpSplit) || d?.type !== 'Sequence') return null;
  const llama =
    n?.type === 'Sequence' &&
    json(n.normalizers) === json([{ type: 'Prepend', prepend: '▁' }, REPLACE_SPACE]) &&
    json(d.decoders) === json([...SP_DECODERS, { type: 'Strip', content: ' ', start: 1, stop: 0 }]);
  const gemma = json(n) === json(REPLACE_SPACE) && json(d.decoders) === json(SP_DECODERS);
  if (!llama && !gemma) return null;
  // Characters some token joins to a following "▁": no split there.
  const joined = new Set<string>();
  for (const token of Object.keys(t.model.vocab)) {
    for (const m of token.matchAll(/([^▁])▁/gu)) joined.add(m[1]!);
  }
  const except = [...joined].map((c) => c.replace(/[\\\]^-]/g, '\\$&')).join('');
  return { prepend: llama, wordBoundary: new RegExp(`(?<=[^▁${except}])(?=▁)`, 'u') };
}

/** Split pre-tokenizer, inverted with "Removed" behaviour: only the matches are kept. */
function matchesOnly(pattern: RegExp): SplitStep {
  return function* (segment: string) {
    for (const match of segment.matchAll(pattern)) if (match[0] !== '') yield match[0];
  };
}

function pairKey(a: string, b: string): string {
  return `${a}\u0000${b}`;
}

function splitMerge(merge: string): [string, string] {
  const space = merge.indexOf(' ');
  if (space <= 0) throw new Error(`Malformed merge ${JSON.stringify(merge)}`);
  return [merge.slice(0, space), merge.slice(space + 1)];
}

/**
 * Split pre-tokenizer with "Isolated" behaviour: every match becomes its own
 * segment, and so does any text between matches.
 */
function isolatedSplit(pattern: RegExp): SplitStep {
  return function* (segment: string) {
    let last = 0;
    for (const match of segment.matchAll(pattern)) {
      if (match[0] === '') continue;
      if (match.index > last) yield segment.slice(last, match.index);
      yield match[0];
      last = match.index + match[0].length;
    }
    if (last < segment.length) yield segment.slice(last);
  };
}

/**
 * Converts a tokenizer.json (Rust regex) pattern to a JS RegExp. The only
 * non-JS syntax our models use is the case-insensitive group `(?i:...)`, which
 * is expanded into explicit [xX] character classes (older browsers lack `(?i:`).
 */
export function toJsRegex(source: string): RegExp {
  let out = '';
  let i = 0;
  while (i < source.length) {
    if (source.startsWith('(?i:', i)) {
      let depth = 1;
      let j = i + 4;
      let body = '';
      while (j < source.length && depth > 0) {
        const ch = source[j]!;
        if (ch === '\\') {
          body += ch + (source[j + 1] ?? '');
          j += 2;
          continue;
        }
        if (ch === '(') depth++;
        if (ch === ')') depth--;
        if (depth > 0) body += ch;
        j++;
      }
      if (/[[\]\\]/.test(body)) throw new Error(`Unsupported case-insensitive group in ${source}`);
      out += '(?:' + body.replace(/[a-zA-Z]/g, (c) => `[${c.toLowerCase()}${c.toUpperCase()}]`) + ')';
      i = j;
      continue;
    }
    out += source[i];
    i++;
  }
  return new RegExp(out, 'gu');
}

/** Digits pre-tokenizer with individual_digits: every numeric character becomes its own segment. */
function splitIndividualDigits(text: string): string[] {
  const out: string[] = [];
  let current = '';
  for (const ch of text) {
    if (DIGIT.test(ch)) {
      if (current) out.push(current);
      out.push(ch);
      current = '';
    } else {
      current += ch;
    }
  }
  if (current) out.push(current);
  return out;
}
