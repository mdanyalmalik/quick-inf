// A character-level BPE tokenizer (Hugging Face tokenizers' "BPE" model without byte-level encoding), as Soprano's
// tokenizer.json is: added tokens like [TEXT] matched first, digits split one by one, then runs of whitespace, of
// word characters and of other characters each merged on their own; anything outside the vocabulary is [UNK].
// Without a pre-tokenizer (Chatterbox's), the text between added tokens is merged as one piece; with the "Whitespace"
// one (Chatterbox Multilingual's), each run of word characters or of other characters (Unicode-aware, as the Rust
// regex's \w is), after a string Replace normalizer (its spaces become the [SPACE] token).

interface CharBpeJson {
  added_tokens: { id: number; content: string }[];
  model: { type: string; vocab: Record<string, number>; merges: ([string, string] | string)[]; unk_token?: string | null };
  normalizer?: Normalizer | null;
  pre_tokenizer?: { type: string } | null;
}

interface Normalizer {
  type: string;
  normalizers?: Normalizer[];
  pattern?: { String?: string };
  content?: string;
}

/** Rust regex's \w: letters, marks, decimal digits, connector punctuation and joiners. */
const WORD = '\\p{Alphabetic}\\p{M}\\p{Nd}\\p{Pc}\\p{Join_Control}';
const WHITESPACE_SPLIT = new RegExp(`[${WORD}]+|[^${WORD}\\s]+`, 'gu');

export class CharBpeTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly ranks = new Map<string, number>();
  private readonly added: Map<string, number>;
  private readonly addedPattern: RegExp | null;
  private readonly unk: number | undefined;
  private readonly split: 'none' | 'whitespace' | 'soprano';
  /** String replacements the normalizer makes, in order. */
  private readonly replacements: [string, string][];

  constructor(json: unknown) {
    const t = json as CharBpeJson;
    if (t.model.type !== 'BPE') throw new Error(`Unsupported tokenizer model ${t.model.type}`);
    this.vocab = new Map(Object.entries(t.model.vocab));
    t.model.merges.forEach((m, rank) => {
      const [a, b] = typeof m === 'string' ? (m.split(' ') as [string, string]) : m;
      this.ranks.set(`${a} ${b}`, rank);
    });
    this.added = new Map(t.added_tokens.map((a) => [a.content, a.id]));
    const escaped = [...this.added.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    this.addedPattern = escaped.length ? new RegExp(`(${escaped.join('|')})`) : null;
    this.unk = t.model.unk_token ? this.vocab.get(t.model.unk_token) : undefined;
    this.split = t.pre_tokenizer === null ? 'none' : t.pre_tokenizer?.type === 'Whitespace' ? 'whitespace' : 'soprano';
    const normalizers = t.normalizer?.type === 'Sequence' ? (t.normalizer.normalizers ?? []) : t.normalizer ? [t.normalizer] : [];
    this.replacements = normalizers.flatMap((n): [string, string][] =>
      n.type === 'Replace' && n.pattern?.String !== undefined ? [[n.pattern.String, n.content ?? '']] : [],
    );
  }

  encode(text: string): number[] {
    for (const [from, to] of this.replacements) text = text.replaceAll(from, to);
    const ids: number[] = [];
    for (const part of this.addedPattern ? text.split(this.addedPattern) : [text]) {
      if (!part) continue;
      const special = this.added.get(part);
      if (special !== undefined) {
        ids.push(special);
        continue;
      }
      const pieces =
        this.split === 'none'
          ? [part]
          : this.split === 'whitespace'
            ? (part.match(WHITESPACE_SPLIT) ?? [])
            : part.split(/(\d)/).flatMap((p) => p.match(/\s+|\w+|[^\w\s]+/gu) ?? []);
      for (const piece of pieces) ids.push(...this.merge(piece));
    }
    return ids;
  }

  /** One piece's characters merged by rank, lowest first. */
  private merge(piece: string): number[] {
    let symbols = Array.from(piece);
    for (;;) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      symbols = [...symbols.slice(0, best), symbols[best]! + symbols[best + 1]!, ...symbols.slice(best + 2)];
    }
    return symbols.flatMap((s) => {
      const id = this.vocab.get(s);
      return id !== undefined ? [id] : this.unk !== undefined ? [this.unk] : [];
    });
  }
}
