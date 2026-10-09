// Byte-level BPE tokenizer for Hugging Face `tokenizer.json` files.
//
// Supports exactly the configurations our catalogue models use (checked in
// `fromJSON`): an optional NFC normalizer; a pre-tokenizer made of Digits
// (individual), Split (regex, isolated) and ByteLevel steps; BPE without
// dropout/unk/byte-fallback; a ByteLevel decoder; no post-processor, or a
// ByteLevel one that leaves offsets untrimmed (it never changes ids). Anything
// else throws instead of silently tokenizing differently from the reference.
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
  added_tokens: Array<{ id: number; content: string; special: boolean }>;
}

export class BpeTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly idToToken: string[];
  private readonly mergeRanks: Map<string, number>;
  private readonly addedById: Map<number, { content: string; special: boolean }>;
  private readonly addedByContent: Map<string, number>;
  private readonly normalizeNfc: boolean;
  private readonly steps: SplitStep[];
  private readonly cache = new Map<string, number[]>();
  private readonly utf8Decoder = new TextDecoder('utf-8', { fatal: false });

  private constructor(json: TokenizerJson, normalizeNfc: boolean, steps: SplitStep[]) {
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
      this.addedById.set(t.id, { content: t.content, special: t.special });
      this.addedByContent.set(t.content, t.id);
    }
    this.normalizeNfc = normalizeNfc;
    this.steps = steps;
  }

  static fromJSON(json: unknown): BpeTokenizer {
    const t = json as TokenizerJson;
    const m = t.model;
    const unsupported: string[] = [];
    if (m?.type !== 'BPE') unsupported.push(`model type ${m?.type}`);
    if (m?.dropout) unsupported.push('BPE dropout');
    if (m?.unk_token) unsupported.push('unk token');
    if (m?.byte_fallback) unsupported.push('byte fallback');
    if (m?.continuing_subword_prefix || m?.end_of_word_suffix) unsupported.push('subword affixes');
    if (m?.ignore_merges) unsupported.push('ignore_merges');
    const normalizer = t.normalizer as { type?: string } | null;
    if (normalizer != null && normalizer.type !== 'NFC') unsupported.push(`normalizer ${normalizer.type}`);
    // A ByteLevel post-processor without offset trimming only adjusts character offsets, never ids.
    const post = t.post_processor as { type?: string; trim_offsets?: boolean } | null;
    if (post != null && !(post.type === 'ByteLevel' && post.trim_offsets === false)) {
      unsupported.push(`post-processor ${post.type}`);
    }
    if (t.decoder?.type !== 'ByteLevel') unsupported.push(`decoder ${t.decoder?.type}`);

    const pre = t.pre_tokenizer;
    const configured = pre?.type === 'Sequence' ? (pre.pretokenizers ?? []) : pre ? [pre] : [];
    const last = configured[configured.length - 1];
    if (!last || last.type !== 'ByteLevel' || last.add_prefix_space !== false) {
      unsupported.push('pre-tokenizer must end with ByteLevel (no prefix space)');
    }
    const steps: SplitStep[] = [];
    for (const step of configured) {
      if (step.type === 'Digits' && step.individual_digits === true) {
        steps.push(splitIndividualDigits);
      } else if (step.type === 'Split' && step.behavior === 'Isolated' && step.invert === false) {
        const source = (step.pattern as { Regex?: string } | undefined)?.Regex;
        if (typeof source === 'string') steps.push(isolatedSplit(toJsRegex(source)));
        else unsupported.push('Split pre-tokenizer without a regex pattern');
      } else if (step.type === 'ByteLevel' && step === last) {
        if (step.use_regex === true) steps.push(isolatedSplit(BYTE_LEVEL_PATTERN));
      } else {
        unsupported.push(`pre-tokenizer step ${step.type}`);
      }
    }
    if (unsupported.length > 0) throw new Error(`Unsupported tokenizer configuration: ${unsupported.join(', ')}`);
    return new BpeTokenizer(t, normalizer?.type === 'NFC', steps);
  }

  get vocabSize(): number {
    return this.idToToken.length;
  }

  /** Id of an added/special token such as "<|im_end|>". */
  specialTokenId(content: string): number {
    const id = this.addedByContent.get(content);
    if (id === undefined) throw new Error(`Unknown special token ${content}`);
    return id;
  }

  /** Encodes plain text (no special-token parsing, no BOS/EOS added). */
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const piece of this.preTokenize(text)) {
      for (const id of this.encodePiece(piece)) ids.push(id);
    }
    return ids;
  }

  decode(ids: readonly number[], { skipSpecialTokens = false } = {}): string {
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

    let symbols = Array.from(piece);
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

    const ids = symbols.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`Symbol ${JSON.stringify(s)} missing from vocabulary`);
      return id;
    });
    if (this.cache.size >= CACHE_LIMIT) this.cache.clear();
    this.cache.set(piece, ids);
    return ids;
  }
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
