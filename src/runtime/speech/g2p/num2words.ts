// English numbers in words as Python's num2words writes them (lang_EN), which Misaki and Inflect's text frontend
// use: "one thousand, two hundred and thirty-four", ordinals ("twenty-first"), years ("nineteen oh-five") and
// decimals read digit by digit ("three point one four").

const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALES = ['', 'thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion'];
const ORDINALS: Record<string, string> = { one: 'first', two: 'second', three: 'third', four: 'fourth', five: 'fifth', six: 'sixth', seven: 'seventh', eight: 'eighth', nine: 'ninth', ten: 'tenth', eleven: 'eleventh', twelve: 'twelfth' };

/** 0–99. */
function tens(n: number): string {
  if (n < 20) return ONES[n]!;
  return n % 10 ? `${TENS[Math.floor(n / 10)]}-${ONES[n % 10]}` : TENS[Math.floor(n / 10)]!;
}

/** 1–999. */
function hundreds(n: number): string {
  const h = Math.floor(n / 100);
  const rest = n % 100;
  if (!h) return tens(rest);
  return rest ? `${ONES[h]} hundred and ${tens(rest)}` : `${ONES[h]} hundred`;
}

/** A whole number in words, from its digits (any length up to the decillions). */
export function cardinal(digits: string | number): string {
  let s = String(digits).replace(/^0+(?=\d)/, '');
  const negative = s.startsWith('-');
  if (negative) s = s.slice(1);
  if (/^0+$/.test(s)) return 'zero';
  const groups: number[] = [];
  for (let end = s.length; end > 0; end -= 3) groups.unshift(Number(s.slice(Math.max(0, end - 3), end)));
  if (groups.length > SCALES.length) return s.split('').map((d) => ONES[Number(d)]).join(' ');
  let out = '';
  groups.forEach((g, i) => {
    if (!g) return;
    const scale = SCALES[groups.length - 1 - i]!;
    const words = scale ? `${hundreds(g)} ${scale}` : hundreds(g);
    // A last group under a hundred joins with "and"; the other groups with commas.
    out = !out ? words : !scale && g < 100 ? `${out} and ${words}` : `${out}, ${words}`;
  });
  return negative ? `minus ${out}` : out;
}

/** 21 → "twenty-first". */
export function ordinal(digits: string | number): string {
  return cardinal(digits).replace(/[a-z]+$/, (last) => ORDINALS[last] ?? `${last.endsWith('y') ? `${last.slice(0, -1)}ie` : last}th`);
}

/** A year: 1905 → "nineteen oh-five", 2027 → "twenty twenty-seven", 2005 → "two thousand and five". */
export function year(value: string | number): string {
  const n = Math.abs(Number(value));
  const high = Math.floor(n / 100);
  const low = n % 100;
  const words = high === 0 || (high % 10 === 0 && low < 10) || high >= 100
    ? cardinal(n)
    : `${cardinal(high)} ${low === 0 ? 'hundred' : low < 10 ? `oh-${cardinal(low)}` : cardinal(low)}`;
  return Number(value) < 0 ? `${words} BC` : words;
}

/** A decimal as num2words reads a float: "3.14" → "three point one four", "1.0" → "one". */
export function decimal(text: string): string {
  const [whole = '0', frac = ''] = text.split('.');
  const digits = frac.replace(/0+$/, '');
  const head = cardinal(whole || '0');
  return digits ? `${head} point ${digits.split('').map((d) => ONES[Number(d)]).join(' ')}` : head;
}
