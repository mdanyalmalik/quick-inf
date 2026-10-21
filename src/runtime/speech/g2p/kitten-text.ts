// Kitten TTS's text preparation, as its Python package does it (kittentts/preprocess.py TextPreprocessor with its
// defaults, and onnx_model.py chunk_text): numbers, currency, times, units and the like in words, URLs and emails
// dropped, contractions expanded, punctuation removed, lower case; then chunks of up to 400 characters, each
// ending in a comma. Two deliberate fixes: ordinals of tens read "twentieth", not the package's "twentyth", and a
// time keeps the space after it ("9:30 and" → "nine thirty and"; the package's pattern swallows it).

const ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALE = ['', 'thousand', 'million', 'billion', 'trillion'];
const DIGITS = ['zero', ...ONES.slice(1, 10)];
const ORDINAL_EXCEPTIONS: Record<string, string> = { one: 'first', two: 'second', three: 'third', four: 'fourth', five: 'fifth', six: 'sixth', seven: 'seventh', eight: 'eighth', nine: 'ninth', twelve: 'twelfth' };
const CURRENCY: Record<string, string> = { $: 'dollar', '€': 'euro', '£': 'pound', '¥': 'yen', '₹': 'rupee', '₩': 'won', '₿': 'bitcoin' };
const SCALE_SUFFIX: Record<string, string> = { K: 'thousand', M: 'million', B: 'billion', T: 'trillion' };
const UNITS: Record<string, string> = {
  km: 'kilometers', kg: 'kilograms', mg: 'milligrams', ml: 'milliliters', gb: 'gigabytes', mb: 'megabytes', kb: 'kilobytes', tb: 'terabytes',
  hz: 'hertz', khz: 'kilohertz', mhz: 'megahertz', ghz: 'gigahertz', mph: 'miles per hour', kph: 'kilometers per hour',
  ms: 'milliseconds', ns: 'nanoseconds', µs: 'microseconds', '°c': 'degrees Celsius', 'c°': 'degrees Celsius', '°f': 'degrees Fahrenheit', 'f°': 'degrees Fahrenheit',
};
const DECADES = ['hundreds', 'tens', 'twenties', 'thirties', 'forties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties'];

function threeDigits(n: number): string {
  const parts: string[] = [];
  const rest = n % 100;
  if (n >= 100) parts.push(`${ONES[Math.floor(n / 100)]} hundred`);
  if (rest >= 20) parts.push(rest % 10 ? `${TENS[Math.floor(rest / 10)]}-${ONES[rest % 10]}` : TENS[Math.floor(rest / 10)]!);
  else if (rest) parts.push(ONES[rest]!);
  return parts.join(' ');
}

/** Kitten's number_to_words: no "and", 1200 → "twelve hundred", -42 → "negative forty-two". */
export function kittenNumber(value: string | number): string {
  let s = String(value).replace(/^\+/, '');
  if (s.startsWith('-')) return /^-0*$/.test(s) ? 'zero' : `negative ${kittenNumber(s.slice(1))}`;
  s = s.replace(/^0+(?=\d)/, '');
  const n = Number(s);
  if (n === 0) return 'zero';
  if (n >= 100 && n <= 9999 && n % 100 === 0 && n % 1000 !== 0 && n / 100 < 20) return `${ONES[n / 100]} hundred`;
  const parts: string[] = [];
  for (let end = s.length, i = 0; end > 0 && i < SCALE.length; end -= 3, i++) {
    const chunk = Number(s.slice(Math.max(0, end - 3), end));
    if (chunk) parts.unshift(SCALE[i] ? `${threeDigits(chunk)} ${SCALE[i]}` : threeDigits(chunk));
  }
  return parts.join(' ');
}

/** "3.10" → "three point one zero" (digits read one by one, trailing zeros kept). */
function floatWords(raw: string): string {
  const negative = raw.startsWith('-');
  const text = negative ? raw.slice(1) : raw;
  let result: string;
  if (text.includes('.')) {
    const [whole = '', frac = ''] = text.split('.');
    result = `${whole ? kittenNumber(whole) : 'zero'} point ${[...frac].map((d) => DIGITS[Number(d)]).join(' ')}`;
  } else {
    result = kittenNumber(text);
  }
  return negative ? `negative ${result}` : result;
}

const numOrFloat = (raw: string) => (raw.includes('.') ? floatWords(raw) : kittenNumber(raw));

function ordinalWords(n: string): string {
  const word = kittenNumber(n);
  const cut = word.includes('-') ? word.lastIndexOf('-') : word.lastIndexOf(' ');
  const [prefix, last] = cut < 0 ? ['', word] : [word.slice(0, cut + 1), word.slice(cut + 1)];
  const ord = ORDINAL_EXCEPTIONS[last] ?? (last.endsWith('t') ? `${last}h` : last.endsWith('e') ? `${last.slice(0, -1)}th` : last.endsWith('y') ? `${last.slice(0, -1)}ieth` : `${last}th`);
  return prefix + ord;
}

const digitWords = (s: string) => [...s].map((c) => DIGITS[Number(c)]).join(' ');

function currency(_: string, symbol: string, rawNum: string, suffix: string | undefined): string {
  const raw = rawNum.replaceAll(',', '');
  const unit = CURRENCY[symbol] ?? '';
  if (suffix) return `${numOrFloat(raw)} ${SCALE_SUFFIX[suffix]} ${unit}${unit ? 's' : ''}`.trim();
  if (raw.includes('.')) {
    const [whole = '', frac = ''] = raw.split('.');
    const cents = Number(frac.slice(0, 2).padEnd(2, '0'));
    let result = unit ? `${kittenNumber(whole)} ${unit}s` : kittenNumber(whole);
    if (cents) result += ` and ${kittenNumber(cents)} cent${cents !== 1 ? 's' : ''}`;
    return result;
  }
  const words = kittenNumber(raw);
  return unit ? `${words} ${unit}${Number(raw) !== 1 ? 's' : ''}` : words;
}

function time(_: string, h: string, m: string, _s: string | undefined, ampm: string | undefined): string {
  const suffix = ampm ? ` ${ampm.toLowerCase()}` : '';
  const hw = kittenNumber(h);
  const mins = Number(m);
  if (mins === 0) return ampm ? `${hw}${suffix}` : `${hw} hundred${suffix}`;
  return mins < 10 ? `${hw} oh ${kittenNumber(mins)}${suffix}` : `${hw} ${kittenNumber(mins)}${suffix}`;
}

function fraction(all: string, num: string, den: string): string {
  const d = Number(den);
  if (d === 0) return all;
  const n = Number(num);
  const denom = d === 2 ? (n === 1 ? 'half' : 'halves') : d === 4 ? (n === 1 ? 'quarter' : 'quarters') : ordinalWords(den) + (n !== 1 ? 's' : '');
  return `${kittenNumber(num)} ${denom}`;
}

function decade(_: string, base: string): string {
  const b = Number(base);
  return b < 10 ? DECADES[b % 10]! : `${kittenNumber(Math.floor(b / 10))} ${DECADES[b % 10]}`;
}

const CONTRACTIONS: [RegExp, string][] = [
  [/\bcan't\b/gi, 'cannot'],
  [/\bwon't\b/gi, 'will not'],
  [/\bshan't\b/gi, 'shall not'],
  [/\bain't\b/gi, 'is not'],
  [/\blet's\b/gi, 'let us'],
  [/\b(\w+)n't\b/gi, '$1 not'],
  [/\b(\w+)'re\b/gi, '$1 are'],
  [/\b(\w+)'ve\b/gi, '$1 have'],
  [/\b(\w+)'ll\b/gi, '$1 will'],
  [/\b(\w+)'d\b/gi, '$1 would'],
  [/\b(\w+)'m\b/gi, '$1 am'],
  [/\bit's\b/gi, 'it is'],
];

const PHONE_JOIN = (...groups: string[]) => groups.map(digitWords).join(' ');

/** Kitten's TextPreprocessor with its default settings. */
export function kittenNormalize(input: string): string {
  let t = input.normalize('NFC');
  t = t.replace(/<[^>]+>/g, ' ');
  t = t.replace(/https?:\/\/\S+|www\.\S+/g, '').trim();
  t = t.replace(/\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b/gi, '').trim();
  for (const [re, to] of CONTRACTIONS) t = t.replace(re, to);
  t = t.replace(/\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b/g, (_, ...g: string[]) => g.slice(0, 4).map(digitWords).join(' dot '));
  t = t.replace(/(?<!\d)(-)\.(\d)/g, (_, minus: string, d: string) => `${minus}0.${d}`);
  t = t.replace(/(?<!\d)\.(\d)/g, (_, d: string) => `0.${d}`);
  t = t.replace(/([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)\s*([KMBT])?(?![a-zA-Z\d])/g, currency);
  t = t.replace(/(-?[\d,]+(?:\.\d+)?)\s*%/g, (_, raw: string) => {
    const r = raw.replaceAll(',', '');
    return `${r.includes('.') ? floatWords(String(Number(r))) : kittenNumber(r)} percent`;
  });
  t = t.replace(/(?<![a-zA-Z\d])(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![a-zA-Z\d])/g, (_, c: string, e: string) => {
    const exp = Number(e);
    return `${numOrFloat(c)} times ten to the ${exp < 0 ? 'negative ' : ''}${kittenNumber(Math.abs(exp))}`;
  });
  t = t.replace(/\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(am|pm))?\b/gi, time);
  t = t.replace(/\b(\d+)(st|nd|rd|th)\b/gi, (_, n: string) => ordinalWords(n));
  t = t.replace(/(\d+(?:\.\d+)?)\s*(km|kg|mg|ml|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|°[cCfF]|[cCfF]°|ms|ns|µs)\b/gi, (_, raw: string, unit: string) => `${raw.includes('.') ? floatWords(String(Number(raw))) : kittenNumber(raw)} ${UNITS[unit.toLowerCase()] ?? unit}`);
  t = t.replace(/(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d])/g, (_, raw: string, s: string) => `${numOrFloat(raw)} ${SCALE_SUFFIX[s]}`);
  t = t.replace(/\b(\d+)\s*\/\s*(\d+)\b/g, fraction);
  t = t.replace(/\b(\d{1,3})0s\b/g, decade);
  t = t.replace(/(?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!-\d)/g, (_, ...g: string[]) => PHONE_JOIN(...g.slice(0, 4)));
  t = t.replace(/(?<!\d-)(?<!\d)\b(\d{3})-(\d{3})-(\d{4})\b(?!-\d)/g, (_, ...g: string[]) => PHONE_JOIN(...g.slice(0, 3)));
  t = t.replace(/(?<!\d-)\b(\d{3})-(\d{4})\b(?!-\d)/g, (_, ...g: string[]) => PHONE_JOIN(...g.slice(0, 2)));
  t = t.replace(/(?<!\w)(\d+)-(\d+)(?!\w)/g, (_, lo: string, hi: string) => `${kittenNumber(lo)} to ${kittenNumber(hi)}`);
  t = t.replace(/\b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)(?=[^\d.]|$)/g, '$1 $2');
  t = t.replace(/(?<![a-zA-Z])-?[\d,]+(?:\.\d+)?/g, (m) => {
    const raw = m.replaceAll(',', '');
    if (!/\d/.test(raw)) return m;
    return raw.includes('.') ? floatWords(raw) : kittenNumber(raw.split('.')[0]!);
  });
  t = t.replace(/[^\p{L}\p{N}_\s]/gu, ' ');
  return t.toLowerCase().replace(/\s+/g, ' ').trim();
}

const ensureComma = (s: string) => (s && !/[.!?,;:]$/.test(s) ? `${s},` : s);

/** Kitten's chunk_text: sentences (split at . ! ?) of up to 400 characters, each ending in punctuation. */
export function kittenChunks(text: string, maxLen = 400): string[] {
  const chunks: string[] = [];
  for (const raw of text.split(/[.!?]+/)) {
    const sentence = raw.trim();
    if (!sentence) continue;
    if (sentence.length <= maxLen) {
      chunks.push(ensureComma(sentence));
      continue;
    }
    let temp = '';
    for (const word of sentence.split(/\s+/)) {
      if (temp.length + word.length + 1 <= maxLen) temp = temp ? `${temp} ${word}` : word;
      else {
        if (temp) chunks.push(ensureComma(temp.trim()));
        temp = word;
      }
    }
    if (temp) chunks.push(ensureComma(temp.trim()));
  }
  return chunks;
}
