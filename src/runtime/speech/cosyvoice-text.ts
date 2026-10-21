// Fun-CosyVoice 3's English text handling: the common cases of its text normaliser (wetext's English rules, then any
// digits left spelled out with inflect, upstream's cosyvoice/cli/frontend.py) and its split_paragraph (sentences at
// . ? ! ; : packed into chunks of 60-80 tokens, a short tail merged into the chunk before). Numbers read as wetext reads
// them: "five hundred and fifty five", years in pairs, "oh" for a leading zero. Tests compare with upstream
// (cosyvoice-text.test.ts). Money differs on purpose: upstream reads "$4.50" as "four point five dollars".

const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALES = ['', 'thousand', 'million', 'billion', 'trillion'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

function tens(n: number): string {
  if (n < 20) return ONES[n]!;
  return n % 10 ? `${TENS[Math.floor(n / 10)]} ${ONES[n % 10]}` : TENS[Math.floor(n / 10)]!;
}

function hundreds(n: number, and: boolean): string {
  const h = Math.floor(n / 100);
  const rest = n % 100;
  if (!h) return rest ? (and ? `and ${tens(rest)}` : tens(rest)) : '';
  return rest ? `${ONES[h]} hundred and ${tens(rest)}` : `${ONES[h]} hundred`;
}

/** A whole number as wetext reads it ("one thousand two hundred and thirty four"); a leading zero is "oh". */
export function cardinal(digits: string): string {
  if (/^0\d/.test(digits)) return `oh ${cardinal(digits.slice(1))}`;
  const n = digits.replace(/^0+(?=\d)/, '');
  if (/^0+$/.test(n)) return 'zero';
  if (n.length > 15) return [...n].map((d) => ONES[Number(d)]).join(' ');
  const groups: number[] = [];
  for (let end = n.length; end > 0; end -= 3) groups.unshift(Number(n.slice(Math.max(0, end - 3), end)));
  const parts: string[] = [];
  groups.forEach((g, i) => {
    if (!g) return;
    const scale = SCALES[groups.length - 1 - i]!;
    // "one thousand and five": the last group under a hundred after larger ones takes "and".
    const words = hundreds(g, i === groups.length - 1 && groups.length > 1 && g < 100);
    parts.push(scale ? `${words} ${scale}` : words);
  });
  return parts.join(' ');
}

/** A four-digit year read in pairs ("nineteen ninety nine", "twenty twenty seven", "nineteen oh five"). */
function year(digits: string): string {
  const [a, b] = [Number(digits.slice(0, 2)), Number(digits.slice(2))];
  if (b === 0) return a % 10 === 0 ? cardinal(digits) : `${tens(a)} hundred`;
  // 2001-2009: "two thousand one".
  if (a === 20 && b < 10) return `two thousand ${ONES[b]}`;
  return `${tens(a)} ${b < 10 ? `oh ${ONES[b]}` : tens(b)}`;
}

const ORDINAL: Record<string, string> = { one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth', nine: 'ninth', twelve: 'twelfth' };

function ordinal(digits: string): string {
  return cardinal(digits).replace(/(\w+)$/, (w) => ORDINAL[w] ?? (w.endsWith('y') ? `${w.slice(0, -1)}ieth` : `${w}th`));
}

/** Reads numbers 1000-2999 (but not round thousands) as years, the rest as cardinals. */
function number(digits: string): string {
  return /^(1[1-9]|20)\d\d$/.test(digits) && !digits.endsWith('000') ? year(digits) : cardinal(digits);
}

const TITLES: Record<string, string> = { 'Dr.': 'doctor', 'Mr.': 'mister', 'Mrs.': 'missus', 'Ms.': 'miss' };

/** The text with numbers, dates, times, symbols and e-mail addresses in words, as CosyVoice reads English. */
export function normalizeEnglish(input: string): string {
  let t = input.trim();
  t = t.replace(/\.{2,}/g, '.');
  t = t.replace(/[\w.+-]+@[\w-]+(?:\.[\w-]+)+/g, (m) => m.replace(/\./g, ' dot ').replace('@', ' at '));
  t = t.replace(/\b(Dr|Mr|Mrs|Ms)\./g, (m) => TITLES[m]!);
  t = t.replace(/(?<=[A-Z]\w* )St\.(?=\s|$)/g, 'Street.');
  t = t.replace(new RegExp(`\\b(${MONTHS.join('|')}) (\\d{1,2})(?:st|nd|rd|th)?(?:, (\\d{4}))?,?`, 'g'), (_, month: string, day: string, y?: string) =>
    `the ${ordinal(day)} of ${month.toLowerCase()}${y ? `, ${year(y)}` : ''}`,
  );
  t = t.replace(/\b(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?\b)?/gi, (_, h: string, m: string, ap?: string) =>
    `${cardinal(h)} ${m === '00' ? "o'clock" : Number(m) < 10 ? `oh ${ONES[Number(m)]}` : tens(Number(m))}${ap ? ` ${ap.toUpperCase()}M` : ''}`,
  );
  t = t.replace(/\b(\d+)\s*([ap])\.?m\b\.?/gi, (_, h: string, ap: string) => `${cardinal(h)} ${ap.toUpperCase()}M`);
  t = t.replace(/\$(\d+)(?:\.(\d{2}))?\b/g, (_, d: string, c?: string) =>
    `${cardinal(d)} dollar${d === '1' ? '' : 's'}${c && Number(c) ? ` and ${cardinal(c)} cent${c === '01' ? '' : 's'}` : ''}`,
  );
  t = t.replace(/(\d+(?:\.\d+)?)%/g, '$1 percent');
  t = t.replace(/°\s*([FC])\b/g, (_, u: string) => ` degrees ${u === 'F' ? 'Fahrenheit' : 'Celsius'}`);
  t = t.replace(/\b(\d+)(st|nd|rd|th)\b/g, (_, d: string) => ordinal(d));
  t = t.replace(/\b(\d+)\.(\d+)\b/g, (_, a: string, b: string) => `${cardinal(a)} point ${[...b].map((d) => ONES[Number(d)]).join(' ')}`);
  t = t.replace(/\d+/g, (d) => number(d));
  return t.replace(/ {2,}/g, ' ');
}

const SPLIT = new Set(['.', '?', '!', ';', ':']);

/** split_paragraph (English): sentences packed into chunks of 60-80 tokens, a tail under 20 merged into the last. */
export function splitParagraph(text: string, count: (text: string) => number, max = 80, min = 60, merge = 20): string[] {
  if (!SPLIT.has(text[text.length - 1]!)) text += '.';
  const utts: string[] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    if (!SPLIT.has(text[i]!)) continue;
    if (i > start) utts.push(text.slice(start, i + 1));
    if (i + 1 < text.length && (text[i + 1] === '"' || text[i + 1] === '”')) {
      utts.push(`${utts.pop() ?? ''}${text[i + 1]}`);
      start = i + 2;
    } else start = i + 1;
  }
  const chunks: string[] = [];
  let current = '';
  for (const u of utts) {
    if (count(current + u) > max && count(current) > min) {
      chunks.push(current);
      current = '';
    }
    current += u;
  }
  if (current) {
    if (count(current) < merge && chunks.length) chunks[chunks.length - 1] += current;
    else chunks.push(current);
  }
  // Upstream drops chunks that are only punctuation.
  return chunks.filter((c) => /[\p{L}\p{N}]/u.test(c));
}
