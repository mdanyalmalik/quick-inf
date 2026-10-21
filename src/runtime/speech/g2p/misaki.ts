// Misaki's English G2P pipeline (misaki/en.py G2P, Apache-2.0, hexgrad), ported: tokens are split into sub-tokens,
// glued words are looked up as the longest known spans, and each word is phonemized right to left so "the", "to"
// and "a" see what follows. A small rule tagger stands in for spaCy's part-of-speech tags; Misaki only reads them
// for punctuation, numbers, a few function words and heteronyms. Unknown words go to the fallback.

import { CONSONANTS, CURRENCIES, Lexicon, PRIMARY, VOWELS, applyStress, stressWeight, type TokenContext } from './lexicon';
import { tagTokens } from './tagger';

export interface MToken {
  text: string;
  tag: string;
  whitespace: string;
  /** Misaki phonemes ('' for silent tokens), null while unknown. */
  phonemes: string | null;
  isHead: boolean;
  alias: string | null;
  stress: number | null;
  currency: string | null;
  numFlags: string;
  prespace: boolean;
}

/** Phonemes for a word the lexicon does not know, or null. */
export type Fallback = (word: string) => string | null;

const PUNCTS = new Set(';:,.!?—…"“”');
const NON_QUOTE_PUNCTS = new Set(';:,.!?—…');
const PUNCT_TAGS = new Set(['.', ',', '-LRB-', '-RRB-', '``', '""', "''", ':', '$', '#', 'NFP']);
const PUNCT_TAG_PHONEMES: Record<string, string> = { '-LRB-': '(', '-RRB-': ')', '``': '“', '""': '”', "''": '”' };
const SUBTOKEN_JUNKS = new Set("',-._‘’/");
const SUBTOKEN =
  /^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$/gu;

export function token(text: string, tag: string, whitespace: string): MToken {
  return { text, tag, whitespace, phonemes: null, isHead: true, alias: null, stress: null, currency: null, numFlags: '', prespace: false };
}

/** Joins tokens into one, as Misaki's merge_tokens (phonemes only when `unk` is given). */
function merge(tokens: MToken[], unk?: string): MToken {
  const stresses = new Set(tokens.flatMap((t) => (t.stress === null ? [] : [t.stress])));
  const currencies = tokens.flatMap((t) => (t.currency === null ? [] : [t.currency])).sort();
  let phonemes: string | null = null;
  if (unk !== undefined) {
    phonemes = '';
    for (const t of tokens) {
      if (t.prespace && phonemes && !/\s$/.test(phonemes) && t.phonemes) phonemes += ' ';
      phonemes += t.phonemes ?? unk;
    }
  }
  const weight = (t: MToken) => [...t.text].reduce((n, c) => n + (c === c.toLowerCase() ? 1 : 2), 0);
  const heaviest = tokens.reduce((best, t) => (weight(t) > weight(best) ? t : best));
  const last = tokens[tokens.length - 1]!;
  return {
    text: tokens.slice(0, -1).map((t) => t.text + t.whitespace).join('') + last.text,
    tag: heaviest.tag,
    whitespace: last.whitespace,
    phonemes,
    isHead: tokens[0]!.isHead,
    alias: null,
    stress: stresses.size === 1 ? [...stresses][0]! : null,
    currency: currencies.at(-1) ?? null,
    numFlags: [...new Set(tokens.flatMap((t) => [...t.numFlags]))].sort().join(''),
    prespace: tokens[0]!.prespace,
  };
}

/** Splits text at whitespace and around punctuation; tags come from the rule tagger. */
export function tokenize(text: string): MToken[] {
  const out: MToken[] = [];
  for (const m of text.trim().matchAll(/(\S+)(\s*)/g)) {
    const pieces = splitPunctuation(m[1]!);
    pieces.forEach((p, i) => out.push(token(p, '', i === pieces.length - 1 && m[2] ? ' ' : '')));
  }
  return tagTokens(out);
}

const ABBREVIATIONS = new Set(['mr.', 'mrs.', 'ms.', 'dr.', 'st.', 'jr.', 'sr.', 'prof.', 'vs.', 'etc.', 'mt.', 'no.', 'inc.', 'ltd.', 'co.']);
const LEADING = /^(["“”‘«(\[{¿¡$£€#]|\.\.\.|…)/;
const TRAILING = /(["“”»)\]}!?,;:%…]|\.\.\.|--)$/;

/** spaCy-like prefix / suffix splitting: quotes, brackets, currency signs and end punctuation become tokens. */
function splitPunctuation(chunk: string): string[] {
  const head: string[] = [];
  const tail: string[] = [];
  let s = chunk;
  for (let m = LEADING.exec(s); m && s.length > m[0].length; m = LEADING.exec(s)) {
    head.push(m[0]);
    s = s.slice(m[0].length);
  }
  for (;;) {
    const m = TRAILING.exec(s);
    if (m && s.length > m[0].length) {
      tail.unshift(m[0]);
      s = s.slice(0, -m[0].length);
    } else if (s.length > 1 && s.endsWith('.') && !isAbbreviation(s)) {
      tail.unshift('.');
      s = s.slice(0, -1);
    } else {
      break;
    }
  }
  return [...head, s, ...tail];
}

const isAbbreviation = (s: string) => ABBREVIATIONS.has(s.toLowerCase()) || /^(\p{L}\.)+$/u.test(s);

/** Sub-tokens, currency and punctuation phonemes, and runs of glued sub-tokens grouped as one word (Misaki's retokenize). */
function retokenize(tokens: MToken[]): (MToken | MToken[])[] {
  const words: (MToken | MToken[])[] = [];
  let currency: string | null = null;
  tokens.forEach((tok, i) => {
    let tks: MToken[] = [tok];
    if (tok.alias === null && tok.phonemes === null) {
      const subs = [...tok.text.matchAll(SUBTOKEN)].map((m) => m[0]);
      tks = (subs.length ? subs : [tok.text]).map((t) => ({ ...token(t, tok.tag, ''), numFlags: tok.numFlags, stress: tok.stress }));
    }
    tks[tks.length - 1]!.whitespace = tok.whitespace;
    tks.forEach((tk, j) => {
      if (tk.alias !== null || tk.phonemes !== null) {
        // already resolved
      } else if (tk.tag === '$' && CURRENCIES.has(tk.text)) {
        currency = tk.text;
        tk.phonemes = '';
      } else if (tk.tag === ':' && (tk.text === '-' || tk.text === '–')) {
        tk.phonemes = '—';
      } else if (PUNCT_TAGS.has(tk.tag) && ![...tk.text].every((c) => /[a-z]/.test(c.toLowerCase()))) {
        tk.phonemes = PUNCT_TAG_PHONEMES[tk.tag] ?? [...tk.text].filter((c) => PUNCTS.has(c)).join('');
      } else if (currency !== null) {
        if (tk.tag !== 'CD') currency = null;
        else if (j + 1 === tks.length && (i + 1 === tokens.length || tokens[i + 1]!.tag !== 'CD')) tk.currency = currency;
      } else if (j > 0 && j < tks.length - 1 && tk.text === '2' && /^\p{L}+$/u.test(tks[j - 1]!.text.slice(-1) + tks[j + 1]!.text[0])) {
        tk.alias = 'to';
      }
      const lastWord = words[words.length - 1];
      if (tk.alias !== null || tk.phonemes !== null) {
        words.push(tk);
      } else if (Array.isArray(lastWord) && !lastWord[lastWord.length - 1]!.whitespace) {
        tk.isHead = false;
        lastWord.push(tk);
      } else {
        words.push(tk.whitespace ? tk : [tk]);
      }
    });
  });
  return words.map((w) => (Array.isArray(w) && w.length === 1 ? w[0]! : w));
}

function tokenContext(ctx: TokenContext, ps: string | null, tk: MToken): TokenContext {
  let vowel = ctx.futureVowel;
  for (const c of ps ?? '') {
    if (VOWELS.has(c) || CONSONANTS.has(c) || NON_QUOTE_PUNCTS.has(c)) {
      vowel = NON_QUOTE_PUNCTS.has(c) ? null : VOWELS.has(c);
      break;
    }
  }
  const futureTo = tk.text === 'to' || tk.text === 'To' || (tk.text === 'TO' && (tk.tag === 'TO' || tk.tag === 'IN'));
  return { futureVowel: vowel, futureTo };
}

const isJunk = (text: string) => [...text].every((c) => SUBTOKEN_JUNKS.has(c));

/** Spaces between the parts of a glued word that mixes letters, digits and symbols; otherwise one primary stress. */
function resolveTokens(tokens: MToken[]): void {
  const text = tokens.slice(0, -1).map((t) => t.text + t.whitespace).join('') + tokens[tokens.length - 1]!.text;
  const kinds = new Set([...text].filter((c) => !SUBTOKEN_JUNKS.has(c)).map((c) => (/\p{L}/u.test(c) ? 0 : /[0-9]/.test(c) ? 1 : 2)));
  const prespace = text.includes(' ') || text.includes('/') || kinds.size > 1;
  tokens.forEach((tk, i) => {
    if (tk.phonemes === null) {
      if (i === tokens.length - 1 && NON_QUOTE_PUNCTS.has(tk.text)) tk.phonemes = tk.text;
      else if (isJunk(tk.text)) tk.phonemes = '';
    } else if (i > 0) {
      tk.prespace = prespace;
    }
  });
  if (prespace) return;
  let indices = tokens.flatMap((tk, i) => (tk.phonemes ? [[tk.phonemes.includes(PRIMARY) ? 1 : 0, stressWeight(tk.phonemes), i] as const] : []));
  if (indices.length === 2 && tokens[indices[0]![2]]!.text.length === 1) {
    const tk = tokens[indices[1]![2]]!;
    tk.phonemes = applyStress(tk.phonemes, -0.5);
    return;
  }
  if (indices.length < 2 || indices.reduce((n, [b]) => n + b, 0) <= Math.floor((indices.length + 1) / 2)) return;
  indices = [...indices].sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]).slice(0, Math.floor(indices.length / 2));
  for (const [, , i] of indices) tokens[i]!.phonemes = applyStress(tokens[i]!.phonemes, -0.5);
}

/** Phonemizes a glued word: the longest known spans first, else the fallback for the whole word. */
function phonemizeGroup(lex: Lexicon, w: MToken[], ctx: TokenContext, fallback?: Fallback): TokenContext {
  let left = 0;
  let right = w.length;
  let useFallback = false;
  while (left < right) {
    const span = w.slice(left, right);
    const tk = span.some((t) => t.alias !== null || t.phonemes !== null) ? null : merge(span);
    const ps = tk ? lex.phonemize(tk, ctx) : null;
    if (ps !== null) {
      w[left]!.phonemes = ps;
      for (const x of w.slice(left + 1, right)) x.phonemes = '';
      ctx = tokenContext(ctx, ps, tk!);
      right = left;
      left = 0;
    } else if (left + 1 < right) {
      left++;
    } else {
      right--;
      const t = w[right]!;
      if (t.phonemes === null) {
        if (isJunk(t.text)) t.phonemes = '';
        else if (fallback) {
          useFallback = true;
          break;
        }
      }
      left = 0;
    }
  }
  if (useFallback) {
    const whole = merge(w);
    w[0]!.phonemes = fallback!(whole.text);
    for (const x of w.slice(1)) x.phonemes = '';
  } else {
    resolveTokens(w);
  }
  return ctx;
}

/** Words with Misaki phonemes ('ɾ' and 'ʔ' as written in the lexicon); `unk` stands for words nothing could read. */
export function phonemizeTokens(lex: Lexicon, text: string, fallback?: Fallback, unk = ''): MToken[] {
  const words = retokenize(tokenize(text));
  let ctx: TokenContext = { futureVowel: null, futureTo: false };
  for (let i = words.length - 1; i >= 0; i--) {
    const w = words[i]!;
    if (Array.isArray(w)) {
      ctx = phonemizeGroup(lex, w, ctx, fallback);
      continue;
    }
    if (w.phonemes === null) w.phonemes = lex.phonemize(w, ctx);
    if (w.phonemes === null && fallback) w.phonemes = fallback(w.text);
    ctx = tokenContext(ctx, w.phonemes, w);
  }
  return words.map((w) => (Array.isArray(w) ? merge(w, unk) : w));
}

/** Misaki's (Kokoro's) phoneme string for a sentence, as misaki.en.G2P gives it: flaps as 'T', glottal stops as 't'. */
export function misakiPhonemes(lex: Lexicon, text: string, fallback?: Fallback, unk = ''): string {
  return phonemizeTokens(lex, text, fallback, unk)
    .map((t) => (t.phonemes ?? unk).replaceAll('ɾ', 'T').replaceAll('ʔ', 't') + t.whitespace)
    .join('');
}
