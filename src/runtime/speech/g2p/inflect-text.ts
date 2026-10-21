// Inflect v2's text preparation, as its ONNX package does it (onnx/inference_onnx.py split_text and
// inflect_nano_v2_frontend.py normalize_text): chunks of whole sentences up to 280 characters, then abbreviations,
// identifiers, money, dates, times, phone numbers, decimals, ordinals, numbers (num2words, hyphens and commas
// dropped) and acronyms in words, with tidy punctuation spacing.

import { cardinal, ordinal } from './num2words';

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const WORD_OVERRIDES: [string, string][] = [
  ['Qwen3', 'Qwen three'], ['Qwen', 'Qwen'], ['PyTorch', 'pie torch'], ['SQLite', 'ess cue lite'], ['USB-C', 'you ess bee see'],
  ['RTX 3060', 'ar tee ex thirty sixty'], ['RTX 3090', 'ar tee ex thirty ninety'], ['RTX 4090', 'ar tee ex forty ninety'],
  ['RTX 5080', 'ar tee ex fifty eighty'], ['RTX 5090', 'ar tee ex fifty ninety'],
];
const LETTER_NAMES: Record<string, string> = {
  A: 'ay', B: 'bee', C: 'see', D: 'dee', E: 'ee', F: 'eff', G: 'gee', H: 'aitch', I: 'eye', J: 'jay', K: 'kay', L: 'ell', M: 'em',
  N: 'en', O: 'oh', P: 'pee', Q: 'cue', R: 'ar', S: 'ess', T: 'tee', U: 'you', V: 'vee', W: 'double you', X: 'ex', Y: 'why', Z: 'zee',
};
const ABBREVIATIONS: [string, string][] = [
  ['Dr.', 'doctor'], ['Mr.', 'mister'], ['Mrs.', 'missus'], ['Ms.', 'miss'], ['Prof.', 'professor'], ['St.', 'saint'],
  ['vs.', 'versus'], ['etc.', 'et cetera'], ['e.g.', 'for example'], ['i.e.', 'that is'],
];
const PUNCT: Record<string, string> = { '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': ', ', '…': '...', '(': ', ', ')': ', ', '[': ', ', ']': ', ', '{': ', ', '}': ', ' };

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');
const words = (n: string | number, ord = false) => (ord ? ordinal(n) : cardinal(n)).replaceAll('-', ' ').replaceAll(',', '');
const digitWords = (s: string) => [...s].filter((c) => /\d/.test(c)).map((c) => words(c)).join(' ');
const identifierDigits = (s: string) => [...s].flatMap((c, i) => (/\d/.test(c) ? [c === '0' && i > 0 ? 'oh' : words(c)] : [])).join(' ');

function identifierToken(token: string): string {
  const m = /^([A-Za-z]?)(\d+)([A-Za-z]?)$/.exec(token);
  if (!m) return token;
  const [, prefix = '', digits = '', suffix = ''] = m;
  const pieces: string[] = [];
  if (prefix) pieces.push(LETTER_NAMES[prefix.toUpperCase()]!);
  pieces.push(digits.length === 3 || digits.startsWith('0') ? identifierDigits(digits) : words(digits));
  if (suffix) pieces.push(LETTER_NAMES[suffix.toUpperCase()]!);
  return pieces.join(' ');
}

function money(_: string, raw: string): string {
  const [dollars = '0', cents] = raw.replaceAll(',', '').split('.');
  const parts = [words(dollars), Number(dollars) === 1 ? 'dollar' : 'dollars'];
  if (cents) {
    const c = Number(cents.slice(0, 2).padEnd(2, '0'));
    if (c) parts.push('and', words(c), c === 1 ? 'cent' : 'cents');
  }
  return parts.join(' ');
}

function slashDate(all: string, m: string, d: string, y: string): string {
  const month = Number(m);
  const day = Number(d);
  const year = Number(y);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return all;
  return `${MONTHS[month - 1]} ${words(day, true)} ${words(year)}`;
}

function clockTime(_: string, h: string, m: string, suffix: string | undefined): string {
  const pieces = [words(h)];
  const minute = Number(m);
  if (minute === 0) pieces.push('o clock');
  else if (minute < 10) pieces.push('oh', words(minute));
  else pieces.push(words(minute));
  if (suffix) pieces.push(...suffix.toLowerCase().replaceAll('.', ''));
  return pieces.join(' ');
}

/** Inflect's normalize_text. */
export function inflectNormalize(input: string): string {
  let t = [...input].map((c) => PUNCT[c] ?? c).join('');
  t = t.replace(/\s+/g, ' ').trim();
  for (const [src, dst] of WORD_OVERRIDES) t = t.replace(new RegExp(`\\b${escape(src)}\\b`, 'g'), dst);
  for (const [src, dst] of ABBREVIATIONS) t = t.replace(new RegExp(`\\b${escape(src)}`, 'gi'), dst);
  t = t.replace(/\b([A-Z])(?:\.([A-Z]))+\./g, (m) => (m.match(/[A-Z]/g) ?? []).join(' '));
  t = t.replace(/\b(apartment|apt\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\s+([A-Za-z]?\d{1,4}[A-Za-z]?)\b/gi, (_, label: string, id: string) => `${label} ${identifierToken(id)}`);
  t = t.replace(/\b(\d{3})(?=\s+(?:North|South|East|West)\b)/gi, (_, n: string) => identifierDigits(n));
  t = t.replace(/\$(\d[\d,]*(?:\.\d{1,2})?)/g, money);
  t = t.replace(/\b(0?[1-9]|1[0-2])\/(0?[1-9]|[12]\d|3[01])\/(20\d{2}|19\d{2})\b/g, slashDate);
  t = t.replace(/\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s*[Mm]\.?)?\b/g, clockTime);
  t = t.replace(/\b(\d{1,2})\s*([AaPp]\.?\s*[Mm]\.?)\b/g, (_, h: string, s: string) => `${words(h)} ${[...s.replace(/[^A-Za-z]/g, '').toLowerCase()].join(' ')}`);
  t = t.replace(/\b(\d{3})-(\d{4})\b/g, (_, a: string, b: string) => `${digitWords(a)}, ${digitWords(b)}`);
  t = t.replace(/\b\d+(?:\.\d+){2,}\b/g, (m) => m.split('.').map((p) => words(p)).join(' point '));
  t = t.replace(/\b(\d+)\.(\d+)\b/g, (_, w: string, f: string) => `${words(w)} point ${digitWords(f)}`);
  t = t.replace(/\b(\d+)(st|nd|rd|th)\b/gi, (_, n: string) => words(n, true));
  t = t.replace(/\b\d[\d,]*\b/g, (m) => {
    const value = m.replaceAll(',', '');
    return value.length >= 5 && !value.startsWith('20') ? digitWords(value) : words(value);
  });
  t = t.replace(/\b[A-Z]{2,}\b/g, (m) => [...m].map((c) => LETTER_NAMES[c] ?? c).join(' '));
  t = t.replace(/,(?:\s*,)+/g, ',');
  t = t.replace(/,\s*([.!?])/g, '$1');
  t = t.replace(/\s+([,;:.!?])/g, '$1');
  t = t.replace(/([,;:.!?])(?=\S)/g, '$1 ');
  return t.replace(/\s+/g, ' ').trim();
}

/** Inflect's split_text: whole sentences (split after . ! ? ; :), long ones cut at a comma or space near 280 characters. */
export function inflectChunks(text: string, limit = 280): string[] {
  const normalized = text.split(/\s+/).filter(Boolean).join(' ');
  const sentences = normalized.split(/(?<=[.!?;:])\s+/).map((s) => s.trim()).filter(Boolean);
  const chunks: string[] = [];
  for (let sentence of sentences.length ? sentences : [normalized]) {
    while (sentence.length > limit) {
      const search = sentence.slice(0, limit + 1);
      const punctuation = Math.max(search.lastIndexOf(','), search.lastIndexOf(';'), search.lastIndexOf(':'));
      let at = punctuation >= limit / 2 ? punctuation + 1 : sentence.lastIndexOf(' ', limit);
      if (at < limit / 2) at = limit;
      chunks.push(sentence.slice(0, at).trim());
      sentence = sentence.slice(at).trim();
    }
    if (sentence) chunks.push(sentence);
  }
  return chunks;
}
