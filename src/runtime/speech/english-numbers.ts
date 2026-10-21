// English numbers in words, as Python's `inflect` writes them with `andword=''` (what the speech models' text
// cleaners use): "one thousand, two hundred thirty-four", ordinals ("twenty-first"), and years read in pairs
// ("nineteen, oh five").

const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALES = ['', 'thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion'];

/** 1–99. */
function tens(n: number): string {
  if (n < 20) return ONES[n]!;
  return n % 10 ? `${TENS[Math.floor(n / 10)]}-${ONES[n % 10]}` : TENS[Math.floor(n / 10)]!;
}

/** 1–999. */
function hundreds(n: number): string {
  const h = Math.floor(n / 100);
  const rest = n % 100;
  if (!h) return tens(rest);
  return rest ? `${ONES[h]} hundred ${tens(rest)}` : `${ONES[h]} hundred`;
}

/** A whole number in words, from its digits (so numbers of any length work). */
export function numberToWords(digits: string): string {
  const clean = digits.replace(/^0+(?=\d)/, '');
  if (/^0+$/.test(clean)) return 'zero';
  const groups: number[] = [];
  for (let end = clean.length; end > 0; end -= 3) groups.unshift(Number(clean.slice(Math.max(0, end - 3), end)));
  if (groups.length > SCALES.length) return clean.split('').map((d) => ONES[Number(d)]).join(' ');
  const parts: string[] = [];
  groups.forEach((g, i) => {
    if (!g) return;
    const scale = SCALES[groups.length - 1 - i]!;
    parts.push(scale ? `${hundreds(g)} ${scale}` : hundreds(g));
  });
  // Groups are separated by commas, except a last group under a hundred ("one thousand one").
  const last = groups[groups.length - 1]!;
  if (parts.length > 1 && last > 0 && last < 100) return `${parts.slice(0, -1).join(', ')} ${parts[parts.length - 1]}`;
  return parts.join(', ');
}

const ORDINAL_ENDINGS: Record<string, string> = {
  one: 'first',
  two: 'second',
  three: 'third',
  five: 'fifth',
  eight: 'eighth',
  nine: 'ninth',
  twelve: 'twelfth',
};

/** "21st" (or 21) → "twenty-first". */
export function ordinalToWords(digits: string): string {
  const words = numberToWords(digits.replace(/\D+$/, ''));
  return words.replace(/(\w+)$/, (last) => {
    if (ORDINAL_ENDINGS[last]) return ORDINAL_ENDINGS[last]!;
    if (last.endsWith('y')) return `${last.slice(0, -1)}ieth`;
    return `${last}th`;
  });
}

/** A four-digit number read in pairs, as `inflect`'s group=2 with zero='oh' ("nineteen, oh five", "eleven, oh oh"). */
export function pairsToWords(digits: string): string {
  const pairs: string[] = [];
  for (let i = 0; i < digits.length; i += 2) pairs.push(digits.slice(i, i + 2));
  return pairs
    .map((p) => {
      if (p.length === 1) return Number(p) ? ONES[Number(p)]! : 'oh';
      const n = Number(p);
      if (p[0] === '0') return n ? `oh ${ONES[n]}` : 'oh oh';
      return tens(n);
    })
    .join(', ');
}
