// Soprano's text preparation, ported from its Python package (soprano/utils/text_normalizer.py clean_text, after
// tortoise-tts, and text_splitter.py with SopranoTTS._preprocess_text): the text spelled out in lower-case words
// (numbers, money, times, abbreviations, symbols), then split into sentences, short ones merged with a neighbour.

import { numberToWords, ordinalToWords, pairsToWords } from './english-numbers';

const ABBREVIATIONS: [RegExp, string][] = (
  [
    ['mrs', 'misess'],
    ['ms', 'miss'],
    ['mr', 'mister'],
    ['dr', 'doctor'],
    ['st', 'saint'],
    ['co', 'company'],
    ['jr', 'junior'],
    ['maj', 'major'],
    ['gen', 'general'],
    ['drs', 'doctors'],
    ['rev', 'reverend'],
    ['lt', 'lieutenant'],
    ['hon', 'honorable'],
    ['sgt', 'sergeant'],
    ['capt', 'captain'],
    ['esq', 'esquire'],
    ['ltd', 'limited'],
    ['col', 'colonel'],
    ['ft', 'fort'],
  ] as const
).map(([a, b]) => [new RegExp(`\\b${a}\\.`, 'gi'), b]);

const CASED_ABBREVIATIONS: [RegExp, string][] = (
  [
    ['Hz', 'hertz'],
    ['kHz', 'kilohertz'],
    ['KBs', 'kilobytes'],
    ['KB', 'kilobyte'],
    ['MBs', 'megabytes'],
    ['MB', 'megabyte'],
    ['GBs', 'gigabytes'],
    ['GB', 'gigabyte'],
    ['TBs', 'terabytes'],
    ['TB', 'terabyte'],
    ['APIs', "a p i's"],
    ['API', 'a p i'],
    ['CLIs', "c l i's"],
    ['CLI', 'c l i'],
    ['CPUs', "c p u's"],
    ['CPU', 'c p u'],
    ['GPUs', "g p u's"],
    ['GPU', 'g p u'],
    ['Ave', 'avenue'],
    ['etc', 'et cetera'],
    ['Mon', 'monday'],
    ['Tues', 'tuesday'],
    ['Wed', 'wednesday'],
    ['Thurs', 'thursday'],
    ['Fri', 'friday'],
    ['Sat', 'saturday'],
    ['Jan', 'january'],
    ['Feb', 'february'],
    ['Mar', 'march'],
    ['Apr', 'april'],
    ['Aug', 'august'],
    ['Sept', 'september'],
    ['Oct', 'october'],
    ['Nov', 'november'],
    ['Dec', 'december'],
    ['and/or', 'and or'],
  ] as const
).map(([a, b]) => [new RegExp(`\\b${a.replace('/', '\\/')}\\b`, 'g'), b]);

const SCALE_SUFFIX: Record<string, string> = { K: 'thousand', M: 'million', B: 'billion', T: 'trillion' };

function expandTime(match: string): string {
  const parts = match.split(':');
  const oh = (p: string) => (p.startsWith('0') ? `oh ${p.slice(1)}` : p);
  if (parts.length === 2) {
    const [hours, minutes] = parts as [string, string];
    if (minutes === '00') {
      if (Number(hours) === 0) return '0';
      return Number(hours) > 12 ? `${hours} minutes` : `${hours} o'clock`;
    }
    return `${hours} ${oh(minutes)}`;
  }
  const [hours, minutes, seconds] = parts as [string, string, string];
  const tail = (p: string) => (p === '00' ? 'oh oh' : oh(p));
  if (Number(hours) !== 0) return `${hours} ${tail(minutes)} ${seconds === '00' ? '' : oh(seconds)}`;
  if (minutes !== '00') return `${minutes} ${tail(seconds)}`;
  return seconds;
}

function expandDollars(amount: string): string {
  const parts = amount.split('.');
  if (parts.length > 2) return `${amount} dollars`;
  const dollars = parts[0] ? parseInt(parts[0].replace(/,/g, ''), 10) || 0 : 0;
  const cents = parts[1] ? parseInt(parts[1], 10) || 0 : 0;
  const unit = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  if (dollars && cents) return `${unit(dollars, 'dollar', 'dollars')}, ${unit(cents, 'cent', 'cents')}`;
  if (dollars) return unit(dollars, 'dollar', 'dollars');
  if (cents) return unit(cents, 'cent', 'cents');
  return 'zero dollars';
}

/** A whole number as Soprano reads it: 1001–2999 (years) in pairs, except round thousands and hundreds. */
function expandNumber(digits: string): string {
  const n = parseInt(digits, 10);
  if (n > 1000 && n < 3000) {
    if (n === 2000) return 'two thousand';
    if (n > 2000 && n < 2010) return `two thousand ${numberToWords(String(n % 100))}`;
    if (n % 100 === 0) return `${numberToWords(String(n / 100))} hundred`;
    return pairsToWords(String(n)).replace(/, /g, ' ');
  }
  return numberToWords(digits);
}

function normalizeNumbers(text: string): string {
  return text
    .replace(/#\d/g, (m) => `number ${m[1]}`)
    .replace(/\b(\d+)([KMBT])\b/gi, (_m, n: string, s: string) => `${n} ${SCALE_SUFFIX[s.toUpperCase()]}`)
    .replace(/(\d[\d,]+\d)/g, (m) => m.replace(/,/g, ''))
    .replace(/(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])/g, (_m, a: string, date: string, b: string) => a + date.split(/[./-]/).join(' dash ') + b)
    .replace(/(\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4})/g, (m) => {
      const d = m.replace(/\D/g, '');
      return `${d.slice(0, 3).split('').join(' ')}, ${d.slice(3, 6).split('').join(' ')}, ${d.slice(6).split('').join(' ')}`;
    })
    .replace(/(\d\d?:\d\d(?::\d\d)?)/g, (m) => expandTime(m))
    .replace(/£([\d,]*\d+)/g, '$1 pounds')
    .replace(/\$([\d.,]*\d+)/g, (_m, amount: string) => expandDollars(amount))
    .replace(/(\d+(?:\.\d+)+)/g, (m) => {
      const [whole, ...rest] = m.split('.');
      return `${whole} point ${rest.map((r) => r.split('').join(' ')).join(' point ')}`;
    })
    .replace(/(\d\s?\*\s?\d)/g, (m) => m.split('*').join(' times '))
    .replace(/(\d\s?\/\s?\d)/g, (m) => m.split('/').join(' over '))
    .replace(/(\d\s?\+\s?\d)/g, (m) => m.split('+').join(' plus '))
    .replace(/(\d?\s?-\s?\d)/g, (m) => m.split('-').join(' minus '))
    .replace(/(\d+(?:\/\d+)+)/g, (m) => {
      const parts = m.split('/');
      return parts.join(parts.length === 2 ? ' over ' : ' slash ');
    })
    .replace(/\d+(st|nd|rd|th)/g, (m) => ordinalToWords(m))
    .replace(/(\d[a-z]|[a-z]\d)/gi, (m) => `${m[0]} ${m[1]}`)
    .replace(/(\d[a-z]|[a-z]\d)/gi, (m) => `${m[0]} ${m[1]}`)
    .replace(/\d+/g, (m) => expandNumber(m));
}

const SPECIAL_CHARACTERS: [RegExp, string][] = [
  [/@/g, ' at '],
  [/&/g, ' and '],
  [/%/g, ' percent '],
  [/:/g, '.'],
  [/;/g, ','],
  [/\+/g, ' plus '],
  [/\\/g, ' backslash '],
  [/~/g, ' about '],
  [/(^| )<3/g, ' heart '],
  [/<=/g, ' less than or equal to '],
  [/>=/g, ' greater than or equal to '],
  [/</g, ' less than '],
  [/>/g, ' greater than '],
  [/=/g, ' equals '],
  [/\//g, ' slash '],
  [/_/g, ' '],
  [/\*/g, ' '],
];

function normalizeSpecial(text: string): string {
  return text
    .replace(/(https?:\/\/)/g, 'h t t p s colon slash slash ')
    .replace(/(. - .)/g, (m) => `${m[0]}, ${m[4]}`)
    .replace(/([A-Z]\.[A-Z])/gi, (m) => `${m[0]} dot ${m[2]}`)
    .replace(/[([{].*[)\]}](.|$)/g, (m) =>
      m
        .replace(/[([{]/g, ', ')
        .replace(/[)\]}][^$.!?,]/g, ', ')
        .replace(/[)\]}]/g, ''),
    );
}

function splitMixedCase(word: string): string {
  const caps = word.match(/[A-Z][a-z]*/g) ?? [];
  if (caps.length === 1) return word;
  if (caps.length === word.length) return word;
  if (caps.length === word.length - 1 && word.endsWith('s')) return `${word.slice(0, -1)}'s`;
  return caps.join(' ');
}

/** Roughly Python's `unidecode` for Latin text: accents dropped, a few symbols spelled as it does, the rest removed. */
function toAscii(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/£/g, 'PS')
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[^\x00-\x7f]/g, '');
}

/** The text as Soprano reads it (soprano/utils/text_normalizer.py clean_text). */
export function cleanText(input: string): string {
  let text = input.replace(/—/g, ' - ');
  text = toAscii(text);
  text = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => (/[.!?]$/.test(line) ? line : `${line}.`))
    .join(' ');
  text = normalizeNumbers(text);
  text = normalizeSpecial(text);
  for (const [re, to] of [...ABBREVIATIONS, ...CASED_ABBREVIATIONS]) text = text.replace(re, to);
  text = text.replace(/\b([A-Z][a-z]*)+\b/g, (m) => splitMixedCase(m));
  for (const [re, to] of SPECIAL_CHARACTERS) text = text.replace(re, to);
  text = text.toLowerCase();
  text = text.replace(/[^A-Za-z !$%&'*+,\-./0-9<>?_]/g, '').replace(/[<>/_+]/g, '');
  text = text
    .replace(/\s+/g, ' ')
    .replace(/ [.?!,]/g, (m) => m[1]!)
    .trim();
  text = text
    .replace(/\.\.\.+/g, '[ELLIPSIS]')
    .replace(/,+/g, ',')
    .replace(/[.,]*\.[.,]*/g, '.')
    .replace(/[.,!]*![.,!]*/g, '!')
    .replace(/[.,!?]*\?[.,!?]*/g, '?')
    .replace(/\[ELLIPSIS\]/g, '...');
  return text.replace(/(\w)\1{2,}/g, (m) => m.slice(0, 2));
}

/**
 * Sentences of a cleaned text (tortoise's split_and_recombine_text with a desired length of 1, so each sentence
 * stands alone, at most `maxLength` characters), with those under `minLength` merged into the one before (or
 * after, for the first), as Soprano does before generating.
 */
export function splitSentences(text: string, minLength = 30, maxLength = 300): string[] {
  const t = text.replace(/\n\n+/g, '\n').replace(/\s+/g, ' ');
  const out: string[] = [];
  let current = '';
  let inQuote = false;
  for (let pos = 0; pos < t.length; pos++) {
    const c = t[pos]!;
    current += c;
    if (c === '"') inQuote = !inQuote;
    if (current.length >= maxLength) {
      // No sentence end in reach: split at the last space.
      const cut = current.lastIndexOf(' ');
      if (cut > 0) {
        out.push(current.slice(0, cut));
        current = current.slice(cut + 1);
      } else {
        out.push(current);
        current = '';
      }
    } else if (!inQuote && ('!?\n'.includes(c) || (c === '.' && ' \n'.includes(t[pos + 1] ?? '')))) {
      while (pos < t.length - 1 && current.length < maxLength && '!?.'.includes(t[pos + 1]!)) current += t[++pos];
      out.push(current);
      current = '';
    }
  }
  out.push(current);
  const sentences = out.map((s) => s.trim()).filter((s) => s.length > 0 && !/^[\s.,;:!?]*$/.test(s));
  if (sentences.length < 2) return sentences;
  const merged: string[] = [];
  for (let i = 0; i < sentences.length; i++) {
    const s = sentences[i]!;
    if (s.length >= minLength) merged.push(s);
    else if (merged.length) merged[merged.length - 1] = `${merged[merged.length - 1]} ${s}`.trim();
    else if (i + 1 < sentences.length) sentences[i + 1] = `${s} ${sentences[i + 1]}`.trim();
    else merged.push(s);
  }
  return merged;
}
