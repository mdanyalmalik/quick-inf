// Misaki's English lexicon lookup (misaki/en.py Lexicon, Apache-2.0, hexgrad), ported: gold and silver tables, case
// handling, possessives, -s / -ed / -ing morphology, numbers and the stress rules, giving Misaki (Kokoro) phonemes.

import type { Entry, LexiconData, TaggedPhonemes } from './lexicon-file';
import { cardinal, decimal, ordinal, year } from './num2words';

export const PRIMARY = 'ˈ';
export const SECONDARY = 'ˌ';
const STRESSES = PRIMARY + SECONDARY;
export const VOWELS = new Set('AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ');
export const CONSONANTS = new Set('bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ');
const DIPHTHONGS = new Set('AIOQWYʤʧ');
const US_TAUS = new Set('AIOWYiuæɑəɛɪɹʊʌ');
const ORDINAL_SUFFIXES = new Set(['st', 'nd', 'rd', 'th']);
export const CURRENCIES = new Map<string, [string, string]>([['$', ['dollar', 'cent']], ['£', ['pound', 'pence']], ['€', ['euro', 'cent']]]);
const ADD_SYMBOLS = new Map([['.', 'dot'], ['/', 'slash']]);
const SYMBOLS = new Map([['%', 'percent'], ['&', 'and'], ['+', 'plus'], ['@', 'at']]);

/** What the words after this one start with: a vowel (true), a consonant (false) or nothing / punctuation (null). */
export interface TokenContext {
  futureVowel: boolean | null;
  futureTo: boolean;
}

/** The parts of a Misaki token the lexicon reads. */
export interface LexToken {
  text: string;
  tag: string;
  alias: string | null;
  stress: number | null;
  currency: string | null;
  isHead: boolean;
  numFlags: string;
}

type Result = string | null;

const isAlpha = (s: string) => /^\p{L}+$/u.test(s);
const isDigits = (s: string) => /^[0-9]+$/.test(s);
const isLexiconWord = (s: string) => /^[A-Za-z'-]*$/.test(s);
const lower = (s: string) => s.toLowerCase();
const upper = (s: string) => s.toUpperCase();

export function stressWeight(ps: string | null): number {
  let n = 0;
  for (const c of ps ?? '') n += DIPHTHONGS.has(c) ? 2 : 1;
  return n;
}

/** Moves each stress mark to just before the next vowel. */
function restress(ps: string): string {
  const chars = [...ps];
  const placed = chars.map((c, i) => {
    if (!STRESSES.includes(c)) return { c, at: i };
    const j = chars.findIndex((v, k) => k >= i && VOWELS.has(v));
    return { c, at: j < 0 ? i : j - 0.5 };
  });
  return placed.sort((a, b) => a.at - b.at).map((p) => p.c).join('');
}

const hasStress = (ps: string) => [...STRESSES].some((s) => ps.includes(s));
const hasVowel = (ps: string) => [...ps].some((c) => VOWELS.has(c));

/** Misaki's apply_stress: below -1 drops stress, -1 / -0.5 / 0 demote it, 0.5–1 add a secondary, above 1 a primary. */
export function applyStress(ps: string | null, stress: number | null): string | null {
  if (ps == null || stress == null) return ps;
  if (stress < -1) return ps.replaceAll(PRIMARY, '').replaceAll(SECONDARY, '');
  if (stress === -1 || ((stress === 0 || stress === -0.5) && ps.includes(PRIMARY))) {
    return ps.replaceAll(SECONDARY, '').replaceAll(PRIMARY, SECONDARY);
  }
  if ((stress === 0 || stress === 0.5 || stress === 1) && !hasStress(ps)) {
    return hasVowel(ps) ? restress(SECONDARY + ps) : ps;
  }
  if (stress >= 1 && !ps.includes(PRIMARY) && ps.includes(SECONDARY)) return ps.replaceAll(SECONDARY, PRIMARY);
  if (stress > 1 && !hasStress(ps)) return hasVowel(ps) ? restress(PRIMARY + ps) : ps;
  return ps;
}

function parentTag(tag: string): string {
  if (tag.startsWith('VB')) return 'VERB';
  if (tag.startsWith('NN')) return 'NOUN';
  if (tag.startsWith('ADV') || tag.startsWith('RB')) return 'ADV';
  if (tag.startsWith('ADJ') || tag.startsWith('JJ')) return 'ADJ';
  return tag;
}

export class Lexicon {
  readonly golds: Map<string, Entry>;
  readonly silvers: Map<string, Entry>;

  constructor(data: LexiconData) {
    this.golds = data.golds;
    this.silvers = data.silvers;
  }

  /** A gold entry as a plain string (its default for a heteronym). */
  gold(word: string, tag = 'DEFAULT'): string {
    const v = this.golds.get(word);
    return (typeof v === 'string' ? v : (v?.[tag] ?? v?.DEFAULT)) ?? '';
  }

  /** Spelled out, letter by letter, stress on the last letter. */
  getNNP(word: string): Result {
    const ps: string[] = [];
    for (const c of word) {
      if (!isAlpha(c)) continue;
      const p = this.golds.get(upper(c));
      if (typeof p !== 'string') return null;
      ps.push(p);
    }
    const s = applyStress(ps.join(''), 0)!;
    const i = s.lastIndexOf(SECONDARY);
    return i < 0 ? s : s.slice(0, i) + PRIMARY + s.slice(i + 1);
  }

  private specialCase(word: string, tag: string, stress: number | null, ctx: TokenContext): Result {
    const add = tag === 'ADD' ? ADD_SYMBOLS.get(word) : undefined;
    if (add) return this.lookup(add, '', -0.5, ctx);
    const symbol = SYMBOLS.get(word);
    if (symbol) return this.lookup(symbol, '', null, ctx);
    const parts = word.split('.');
    if (word.replace(/^\.+|\.+$/g, '').includes('.') && isAlpha(word.replaceAll('.', '')) && Math.max(...parts.map((p) => p.length)) < 3) {
      return this.getNNP(word);
    }
    if (word === 'a' || word === 'A') return tag === 'DT' ? 'ɐ' : 'ˈA';
    if (word === 'am' || word === 'Am' || word === 'AM') {
      if (tag.startsWith('NN')) return this.getNNP(word);
      if (ctx.futureVowel === null || word !== 'am' || (stress && stress > 0)) return this.gold('am');
      return 'ɐm';
    }
    if (word === 'an' || word === 'An' || word === 'AN') return word === 'AN' && tag.startsWith('NN') ? this.getNNP(word) : 'ɐn';
    if (word === 'I' && tag === 'PRP') return `${SECONDARY}I`;
    if ((word === 'by' || word === 'By' || word === 'BY') && parentTag(tag) === 'ADV') return 'bˈI';
    if (word === 'to' || word === 'To' || (word === 'TO' && (tag === 'TO' || tag === 'IN'))) {
      return ctx.futureVowel === null ? this.gold('to') : ctx.futureVowel ? 'tʊ' : 'tə';
    }
    if (word === 'in' || word === 'In' || (word === 'IN' && tag !== 'NNP')) return (ctx.futureVowel === null || tag !== 'IN' ? PRIMARY : '') + 'ɪn';
    if (word === 'the' || word === 'The' || (word === 'THE' && tag === 'DT')) return ctx.futureVowel === true ? 'ði' : 'ðə';
    if (tag === 'IN' && /^vs\.?$/i.test(word)) return this.lookup('versus', '', null, ctx);
    if (word === 'used' || word === 'Used' || word === 'USED') {
      return (tag === 'VBD' || tag === 'JJ') && ctx.futureTo ? this.gold('used', 'VBD') : this.gold('used');
    }
    return null;
  }

  isKnown(word: string): boolean {
    if (this.golds.has(word) || SYMBOLS.has(word) || this.silvers.has(word)) return true;
    if (!isAlpha(word) || !isLexiconWord(word)) return false;
    if (word.length === 1) return true;
    if (word === upper(word) && this.golds.has(lower(word))) return true;
    return word.slice(1) === upper(word.slice(1)) || this.derive(word, '', null, null) !== null;
  }

  /**
   * A regular -s / -ed / -ing form from its stem, as Misaki's get_word does for unknown words. The lexicon file
   * leaves out the forms this rebuilds exactly, so lookups and isKnown use it where Misaki's tables had them.
   */
  private derive(word: string, tag: string, stress: number | null, ctx: TokenContext | null): Result {
    return this.stemS(word, tag, stress, ctx) ?? this.stemEd(word, tag, stress, ctx) ?? this.stemIng(word, tag, stress ?? 0.5, ctx);
  }

  lookup(word: string, tag: string, stress: number | null, ctx: TokenContext | null): Result {
    let isNNP = false;
    if (word === upper(word) && !this.golds.has(word)) {
      word = lower(word);
      isNNP = tag === 'NNP';
    }
    let entry: Entry | undefined = this.golds.get(word);
    if (entry === undefined && !isNNP) entry = this.silvers.get(word);
    if (entry === undefined) {
      const derived = this.derive(word, tag, stress, ctx);
      if (derived !== null) return derived;
    }
    let ps: Result = null;
    if (typeof entry === 'string') ps = entry;
    else if (entry) ps = pickTag(entry, tag, ctx);
    if (ps === null || (isNNP && !ps.includes(PRIMARY))) return this.getNNP(word);
    return applyStress(ps, stress);
  }

  private s(stem: Result): Result {
    if (!stem) return null;
    const last = stem.at(-1)!;
    if ('ptkfθ'.includes(last)) return `${stem}s`;
    if ('szʃʒʧʤ'.includes(last)) return `${stem}ᵻz`;
    return `${stem}z`;
  }

  private stemS(word: string, tag: string, stress: number | null, ctx: TokenContext | null): Result {
    if (word.length < 3 || !word.endsWith('s')) return null;
    let stem: string;
    if (!word.endsWith('ss') && this.isKnown(word.slice(0, -1))) stem = word.slice(0, -1);
    else if ((word.endsWith("'s") || (word.length > 4 && word.endsWith('es') && !word.endsWith('ies'))) && this.isKnown(word.slice(0, -2))) stem = word.slice(0, -2);
    else if (word.length > 4 && word.endsWith('ies') && this.isKnown(`${word.slice(0, -3)}y`)) stem = `${word.slice(0, -3)}y`;
    else return null;
    return this.s(this.lookup(stem, tag, stress, ctx));
  }

  private ed(stem: Result): Result {
    if (!stem) return null;
    const last = stem.at(-1)!;
    if ('pkfθʃsʧ'.includes(last)) return `${stem}t`;
    if (last === 'd') return `${stem}ᵻd`;
    if (last !== 't') return `${stem}d`;
    if (stem.length < 2) return `${stem}ɪd`;
    if (US_TAUS.has(stem.at(-2)!)) return `${stem.slice(0, -1)}ɾᵻd`;
    return `${stem}ᵻd`;
  }

  private stemEd(word: string, tag: string, stress: number | null, ctx: TokenContext | null): Result {
    if (word.length < 4 || !word.endsWith('d')) return null;
    let stem: string;
    if (!word.endsWith('dd') && this.isKnown(word.slice(0, -1))) stem = word.slice(0, -1);
    else if (word.length > 4 && word.endsWith('ed') && !word.endsWith('eed') && this.isKnown(word.slice(0, -2))) stem = word.slice(0, -2);
    else return null;
    return this.ed(this.lookup(stem, tag, stress, ctx));
  }

  private ing(stem: Result): Result {
    if (!stem) return null;
    if (stem.length > 1 && stem.at(-1) === 't' && US_TAUS.has(stem.at(-2)!)) return `${stem.slice(0, -1)}ɾɪŋ`;
    return `${stem}ɪŋ`;
  }

  private stemIng(word: string, tag: string, stress: number | null, ctx: TokenContext | null): Result {
    if (word.length < 5 || !word.endsWith('ing')) return null;
    let stem: string;
    if (word.length > 5 && this.isKnown(word.slice(0, -3))) stem = word.slice(0, -3);
    else if (this.isKnown(`${word.slice(0, -3)}e`)) stem = `${word.slice(0, -3)}e`;
    else if (word.length > 5 && /([bcdgklmnprstvxz])\1ing$|cking$/.test(word) && this.isKnown(word.slice(0, -4))) stem = word.slice(0, -4);
    else return null;
    return this.ing(this.lookup(stem, tag, stress, ctx));
  }

  getWord(word: string, tag: string, stress: number | null, ctx: TokenContext): Result {
    const special = this.specialCase(word, tag, stress, ctx);
    if (special !== null) return special;
    const wl = lower(word);
    if (
      word.length > 1 && isAlpha(word.replaceAll("'", '')) && word !== wl && (tag !== 'NNP' || word.length > 7) &&
      !this.golds.has(word) && !this.silvers.has(word) && (word === upper(word) || word.slice(1) === lower(word.slice(1))) &&
      (this.golds.has(wl) || this.silvers.has(wl) || this.stemS(wl, tag, stress, ctx) || this.stemEd(wl, tag, stress, ctx) || this.stemIng(wl, tag, stress, ctx))
    ) {
      word = wl;
    }
    if (this.isKnown(word)) return this.lookup(word, tag, stress, ctx);
    if (word.endsWith("s'") && this.isKnown(`${word.slice(0, -2)}'s`)) return this.lookup(`${word.slice(0, -2)}'s`, tag, stress, ctx);
    if (word.endsWith("'") && this.isKnown(word.slice(0, -1))) return this.lookup(word.slice(0, -1), tag, stress, ctx);
    return this.derive(word, tag, stress, ctx);
  }

  private static isCurrency(word: string): boolean {
    if (!word.includes('.')) return true;
    if (word.split('.').length > 2) return false;
    return word.split('.')[1]!.length < 3;
  }

  /** A number token: years, ordinals, digits read one by one, decimals and currency amounts, as Misaki reads them. */
  getNumber(word: string, currency: string | null, isHead: boolean, numFlags: string): Result {
    const suffixMatch = /[a-z']+$/.exec(word);
    const suffix = suffixMatch?.[0] ?? null;
    if (suffix) word = word.slice(0, -suffix.length);
    const result: string[] = [];
    if (word.startsWith('-')) {
      result.push(this.lookup('minus', '', null, null) ?? '');
      word = word.slice(1);
    }
    const extend = (num: string, first = true, escape = false) => {
      const splits = (escape ? num : cardinal(num)).split(/[^a-z]+/);
      splits.forEach((w, i) => {
        if (w !== 'and' || numFlags.includes('&')) {
          if (first && i === 0 && splits.length > 1 && w === 'one' && numFlags.includes('a')) result.push('ə');
          else result.push(this.lookup(w, '', w === 'point' ? -2 : null, null) ?? '');
        } else if (w === 'and' && numFlags.includes('n') && result.length) {
          result[result.length - 1] += 'ən';
        }
      });
    };
    if (isDigits(word) && suffix && ORDINAL_SUFFIXES.has(suffix)) {
      extend(ordinal(word), true, true);
    } else if (!result.length && word.length === 4 && !(currency && CURRENCIES.has(currency)) && isDigits(word)) {
      extend(year(word), true, true);
    } else if (!isHead && !word.includes('.')) {
      const num = word.replaceAll(',', '');
      if (num[0] === '0' || num.length > 3) {
        for (const n of num) extend(n, false);
      } else if (num.length === 3 && !num.endsWith('00')) {
        extend(num[0]!);
        if (num[1] === '0') {
          result.push(this.lookup('O', '', -2, null) ?? '');
          extend(num[2]!, false);
        } else {
          extend(num.slice(1), false);
        }
      } else {
        extend(num);
      }
    } else if (word.split('.').length > 2 || !isHead) {
      let first = true;
      for (const num of word.replaceAll(',', '').split('.')) {
        if (!num) continue;
        if (num[0] === '0' || (num.length !== 2 && [...num.slice(1)].some((n) => n !== '0'))) {
          for (const n of num) extend(n, false);
        } else {
          extend(num, first);
        }
        first = false;
      }
    } else if (currency && CURRENCIES.has(currency) && Lexicon.isCurrency(word)) {
      const units = CURRENCIES.get(currency)!;
      let pairs = word.replaceAll(',', '').split('.').slice(0, 2).map((n, i) => [n ? Number(n) : 0, units[i]!] as const);
      if (pairs.length > 1) {
        if (pairs[1]![0] === 0) pairs = pairs.slice(0, 1);
        else if (pairs[0]![0] === 0) pairs = pairs.slice(1);
      }
      pairs.forEach(([num, unit], i) => {
        if (i > 0) result.push(this.lookup('and', '', null, null) ?? '');
        extend(String(num), i === 0);
        result.push((Math.abs(num) !== 1 && unit !== 'pence' ? this.stemS(`${unit}s`, '', null, null) : this.lookup(unit, '', null, null)) ?? '');
      });
    } else {
      let words: string;
      if (isDigits(word)) words = cardinal(word);
      else if (!word.includes('.')) words = suffix && ORDINAL_SUFFIXES.has(suffix) ? ordinal(word.replaceAll(',', '')) : cardinal(word.replaceAll(',', ''));
      else {
        const w = word.replaceAll(',', '');
        words = w[0] === '.' ? `point ${[...w.slice(1)].map((n) => cardinal(n)).join(' ')}` : decimal(w);
      }
      extend(words, true, true);
    }
    if (!result.length) return null;
    const ps = result.join(' ');
    if (suffix === 's' || suffix === "'s") return this.s(ps);
    if (suffix === 'ed' || suffix === "'d") return this.ed(ps);
    if (suffix === 'ing') return this.ing(ps);
    return ps;
  }

  private appendCurrency(ps: string, currency: string | null): string {
    const units = currency ? CURRENCIES.get(currency) : undefined;
    const name = units ? this.stemS(`${units[0]}s`, '', null, null) : null;
    return name ? `${ps} ${name}` : ps;
  }

  static isNumber(word: string, isHead: boolean): boolean {
    if (![...word].some((c) => isDigits(c))) return false;
    for (const s of ['ing', "'d", 'ed', "'s", ...ORDINAL_SUFFIXES, 's']) {
      if (word.endsWith(s)) {
        word = word.slice(0, -s.length);
        break;
      }
    }
    return [...word].every((c, i) => isDigits(c) || c === ',' || c === '.' || (isHead && i === 0 && c === '-'));
  }

  /** Misaki phonemes for one token (null when the word is unknown). */
  phonemize(tk: LexToken, ctx: TokenContext): Result {
    const word = (tk.alias ?? tk.text).replaceAll('‘', "'").replaceAll('’', "'").normalize('NFKC');
    const stress = word === lower(word) ? null : word === upper(word) ? 2 : 0.5;
    const ps = this.getWord(word, tk.tag, stress, ctx);
    if (ps !== null) return applyStress(this.appendCurrency(ps, tk.currency), tk.stress);
    if (Lexicon.isNumber(word, tk.isHead)) return applyStress(this.getNumber(word, tk.currency, tk.isHead, tk.numFlags), tk.stress);
    return null;
  }
}

function pickTag(entry: TaggedPhonemes, tag: string, ctx: TokenContext | null): Result {
  if (ctx && ctx.futureVowel === null && 'None' in entry) return entry.None ?? null;
  const key = tag in entry ? tag : parentTag(tag);
  return key in entry ? (entry[key] ?? null) : (entry.DEFAULT ?? null);
}
