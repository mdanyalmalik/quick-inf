// T5's tokenizer (SD 3.5's and FLUX.1's third text encoder), as Hugging Face's `tokenizers` runs its
// tokenizer.json: a Unigram model (SentencePiece) over Metaspace-split words.
//
// Normalization is SentencePiece's "precompiled" character map (nmt_nfkc): a double-array trie (Darts) over UTF-8
// bytes whose leaves point at replacement strings, applied per grapheme cluster, falling back to each character, as
// the spm_precompiled crate does. Then trailing whitespace is stripped and runs of two or more spaces become a single
// "▁" (T5's legacy quirk). Metaspace turns spaces into "▁", splits before each "▁" and puts one at the start; each
// piece is then segmented into the vocabulary's pieces by Viterbi over their log-probabilities, unknown characters
// costing the lowest score minus 10 (consecutive unknowns become one <unk>).

const REPLACEMENT = '▁';
/** Hugging Face's penalty for an unknown character, below the vocabulary's lowest score. */
const UNK_PENALTY = 10;

interface UnigramJson {
  normalizer?: { type: string; normalizers?: NormalizerJson[] } & NormalizerJson;
  model: { type: string; unk_id: number | null; vocab: [string, number][] };
  added_tokens?: { id: number; content: string; special: boolean }[];
}

interface NormalizerJson {
  type: string;
  precompiled_charsmap?: string;
  strip_left?: boolean;
  strip_right?: boolean;
  pattern?: { Regex?: string; String?: string };
  content?: string;
}

/** SentencePiece's precompiled normalization map. */
class PrecompiledCharsmap {
  private readonly trie: Uint32Array;
  private readonly normalized: Uint8Array;
  private readonly cache = new Map<string, string | null>();

  constructor(base64: string) {
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    const trieBytes = view.getUint32(0, true);
    this.trie = new Uint32Array(trieBytes / 4);
    for (let i = 0; i < this.trie.length; i++) this.trie[i] = view.getUint32(4 + i * 4, true);
    this.normalized = bytes.subarray(4 + trieBytes);
  }

  normalize(text: string): string {
    let out = '';
    for (const { segment } of graphemes(text)) {
      if (utf8.encode(segment).length < 6) {
        const whole = this.transform(segment);
        if (whole !== null) {
          out += whole;
          continue;
        }
      }
      for (const c of segment) out += this.transform(c) ?? c;
    }
    return out;
  }

  /** The replacement for `chunk` (the first prefix the trie matches), or null. */
  private transform(chunk: string): string | null {
    const cached = this.cache.get(chunk);
    if (cached !== undefined) return cached;
    const key = utf8.encode(chunk);
    const { trie } = this;
    let result: number | null = null;
    let pos = 0;
    let unit = trie[pos]!;
    pos ^= offset(unit);
    for (const c of key) {
      if (c === 0) break;
      pos ^= c;
      unit = trie[pos]!;
      if ((unit & 0x800000ff) >>> 0 !== c) break;
      pos ^= offset(unit);
      if ((unit >>> 8) & 1) {
        result = trie[pos]! & 0x7fffffff;
        break;
      }
    }
    let value: string | null = null;
    if (result !== null) {
      let end = result;
      while (end < this.normalized.length && this.normalized[end] !== 0) end++;
      value = utf8d.decode(this.normalized.subarray(result, end));
    }
    this.cache.set(chunk, value);
    return value;
  }
}

/** A Darts unit's offset to its children. */
function offset(unit: number): number {
  return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
}

const utf8 = new TextEncoder();
const utf8d = new TextDecoder();
let segmenter: Intl.Segmenter | null = null;
function graphemes(text: string): Iterable<{ segment: string }> {
  segmenter ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  return segmenter.segment(text);
}

/** A trie over the vocabulary's pieces, by character. */
interface TrieNode {
  id: number;
  next: Map<string, TrieNode>;
}

export class UnigramTokenizer {
  private readonly root: TrieNode = { id: -1, next: new Map() };
  private readonly scores: Float64Array;
  private readonly unkId: number;
  private readonly unkScore: number;
  private readonly normalizers: ((text: string) => string)[];
  private readonly special = new Map<string, number>();
  private specialPattern: RegExp | null = null;

  private constructor(json: UnigramJson) {
    const { vocab, unk_id } = json.model;
    this.scores = Float64Array.from(vocab, ([, score]) => score);
    this.unkId = unk_id ?? 0;
    let min = Infinity;
    vocab.forEach(([piece, score], id) => {
      min = Math.min(min, score);
      let node = this.root;
      for (const c of piece) {
        let child = node.next.get(c);
        if (!child) node.next.set(c, (child = { id: -1, next: new Map() }));
        node = child;
      }
      node.id = id;
    });
    this.unkScore = min - UNK_PENALTY;
    for (const t of json.added_tokens ?? []) if (t.special) this.special.set(t.content, t.id);
    const list = json.normalizer?.type === 'Sequence' ? (json.normalizer.normalizers ?? []) : json.normalizer ? [json.normalizer] : [];
    this.normalizers = list.map(normalizer);
  }

  static fromJSON(json: unknown): UnigramTokenizer {
    const parsed = json as UnigramJson;
    if (parsed.model?.type !== 'Unigram') throw new Error('Not a Unigram tokenizer.');
    return new UnigramTokenizer(parsed);
  }

  specialTokenId(token: string): number {
    const id = this.special.get(token);
    if (id === undefined) throw new Error(`The tokenizer has no ${token} token.`);
    return id;
  }

  /** Token ids for `text`, without adding special tokens (those written in it count, as in Hugging Face's). */
  encode(text: string): number[] {
    if (this.special.size === 0) return this.encodeText(text);
    this.specialPattern ??= new RegExp(`(${[...this.special.keys()].map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`);
    const ids: number[] = [];
    for (const part of text.split(this.specialPattern)) {
      const special = this.special.get(part);
      if (special !== undefined) ids.push(special);
      else if (part) ids.push(...this.encodeText(part));
    }
    return ids;
  }

  private encodeText(text: string): number[] {
    let normalized = text;
    for (const n of this.normalizers) normalized = n(normalized);
    if (!normalized) return [];
    // Metaspace: spaces become "▁", a "▁" goes in front, and each word starts at a "▁".
    let replaced = normalized.replaceAll(' ', REPLACEMENT);
    if (!replaced.startsWith(REPLACEMENT)) replaced = REPLACEMENT + replaced;
    const ids: number[] = [];
    for (const word of replaced.split(/(?=▁)/u)) if (word) ids.push(...this.segment(word));
    return ids;
  }

  /** T5's input: the text's tokens truncated to leave room for </s>, then </s>, then <pad> up to `length`. */
  encodePadded(text: string, length: number): number[] {
    const ids = this.encode(text).slice(0, length - 1);
    ids.push(this.specialTokenId('</s>'));
    const pad = this.specialTokenId('<pad>');
    while (ids.length < length) ids.push(pad);
    return ids;
  }

  /** The most likely segmentation of one word (Viterbi). */
  private segment(word: string): number[] {
    const chars = [...word];
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1).fill(-1);
    const token = new Int32Array(n + 1).fill(-1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let node: TrieNode | undefined = this.root;
      let singleChar = false;
      for (let j = i; j < n && node; j++) {
        node = node.next.get(chars[j]!);
        if (!node) break;
        if (node.id >= 0) {
          if (j === i) singleChar = true;
          const score = best[i]! + this.scores[node.id]!;
          if (score > best[j + 1]!) {
            best[j + 1] = score;
            from[j + 1] = i;
            token[j + 1] = node.id;
          }
        }
      }
      if (!singleChar) {
        const score = best[i]! + this.unkScore;
        if (score > best[i + 1]!) {
          best[i + 1] = score;
          from[i + 1] = i;
          token[i + 1] = this.unkId;
        }
      }
    }
    const ids: number[] = [];
    for (let end = n; end > 0; end = from[end]!) ids.push(token[end]!);
    ids.reverse();
    // Consecutive unknowns become one.
    return ids.filter((id, k) => !(id === this.unkId && ids[k - 1] === this.unkId));
  }
}

function normalizer(json: NormalizerJson): (text: string) => string {
  switch (json.type) {
    case 'Precompiled': {
      const map = new PrecompiledCharsmap(json.precompiled_charsmap!);
      return (text) => map.normalize(text);
    }
    case 'Strip':
      return (text) => {
        let out = text;
        if (json.strip_left) out = out.replace(/^\s+/u, '');
        if (json.strip_right) out = out.replace(/\s+$/u, '');
        return out;
      };
    case 'Replace': {
      const pattern = json.pattern?.Regex !== undefined ? new RegExp(json.pattern.Regex, 'gu') : null;
      const literal = json.pattern?.String ?? '';
      const content = json.content ?? '';
      return (text) => (pattern ? text.replace(pattern, content) : text.replaceAll(literal, content));
    }
    case 'NFKC':
      return (text) => text.normalize('NFKC');
    default:
      throw new Error(`Unsupported normalizer: ${json.type}`);
  }
}
