// Phonon's English text front end, ported from Gradium's Rust runtime (xn-ptts, ptts/src): preprocess.rs
// normalize_text with Lang::En and Rules::DEFAULT (the character pass, then the rewrite rules of preprocess/rewrite/:
// numbers, currency, dashed digits, emails, URLs, abbreviations, elongations), tts_model.rs
// split_into_best_sentences and prepare_text_prompt, and plan.rs chunks, fit_or_error and frame_budget. The readings
// are Rust's, quirks included ("$1" reads "1 dollars"); change one only on purpose.
//
// Strings are walked by code point where Rust walks chars; the predicates are Rust's char ones (White_Space,
// Alphabetic, Alphabetic or Numeric, Cc).

/** What the chunker needs from the tokenizer: Hugging Face's encode (no special tokens) and decode. */
export interface PhononTokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
}

export interface PhononChunk {
  ids: number[];
  /** Frames still generated after the model signals the end. */
  framesAfterEos: number;
  /** The most frames the chunk may generate. */
  frameBudget: number;
}

/** Sentences are grouped into chunks of up to this many tokens (tts_model.rs MAX_TOKENS_PER_CHUNK). */
const MAX_TOKENS_PER_CHUNK = 50;
/** A chunk over this many tokens is cut (plan.rs MAX_FIT_TOKENS). */
const MAX_FIT_TOKENS = 200;
/** Mimi's frame rate in the checkpoint's config. */
const FRAME_RATE = 12.5;

const isWhitespace = (c: string | undefined) => c !== undefined && /^\p{White_Space}$/u.test(c);
const isAlphabetic = (c: string | undefined) => c !== undefined && /^\p{Alphabetic}$/u.test(c);
const isAlphanumeric = (c: string | undefined) => c !== undefined && /^[\p{Alphabetic}\p{N}]$/u.test(c);
const isDigit = (c: string | undefined) => c !== undefined && c >= '0' && c <= '9';
const isAsciiPunctuation = (c: string | undefined) => c !== undefined && c.length === 1 && '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'.includes(c);
const trim = (s: string) => s.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, '');
const splitWhitespace = (s: string) => s.split(/\p{White_Space}+/u).filter(Boolean);

// ---- The character pass (preprocess.rs) ----

/**
 * A character sink that keeps out the punctuation pile-ups the substitutions would make: a "." or "," swallows the
 * whitespace and punctuation right before it, and whitespace never doubles.
 */
class Appender {
  buffer: string[] = [];

  push(c: string): void {
    if (c === '.' || c === ',') {
      for (let l = this.last(); isWhitespace(l) || (isAsciiPunctuation(l) && !this.keeps(l!)); l = this.last()) this.buffer.pop();
    }
    this.buffer.push(c);
  }

  pushStr(s: string): void {
    for (const c of s) this.push(c);
  }

  /** Quotes, the symbols the rules read or the model says, and the "$" after a number stay before a "." or ",". */
  private keeps(last: string): boolean {
    if ('"\'@+%#&'.includes(last)) return true;
    return last === '$' && isDigit(this.buffer[this.buffer.length - 2]);
  }

  last(): string | undefined {
    return this.buffer[this.buffer.length - 1];
  }

  endsWithOne(): boolean {
    let n = this.buffer.length;
    while (n > 0 && isWhitespace(this.buffer[n - 1])) n--;
    const before = this.buffer[n - 2];
    return this.buffer[n - 1] === '1' && !(isDigit(before) || before === '.' || before === ',');
  }

  /** A word "n" or "N", with or without a period: before "°" it is the number sign. */
  endsWithN(): boolean {
    const n = this.buffer.length - (this.last() === '.' ? 1 : 0);
    const c = this.buffer[n - 1];
    return (c === 'n' || c === 'N') && !isAlphanumeric(this.buffer[n - 2]);
  }

  endsWithSpacedDigit(): boolean {
    let n = this.buffer.length;
    while (n > 0 && isWhitespace(this.buffer[n - 1])) n--;
    return n < this.buffer.length && isDigit(this.buffer[n - 1]);
  }

  pushSpoken(word: string): void {
    this.pushWhitespace();
    this.pushStr(word);
    this.pushWhitespace();
  }

  pushWhitespace(): void {
    if (this.buffer.length && !isWhitespace(this.last())) this.push(' ');
  }

  popWhitespace(): void {
    while (isWhitespace(this.last())) this.buffer.pop();
  }

  toString(): string {
    this.popWhitespace();
    return this.buffer.join('');
  }
}

/** Letters with a diacritic the tokenizer has no piece for, each followed by the letter it folds to. */
const FOLDS = new Map(
  (
    Array.from(
      'ÌIÒOÝYìiòoýyÿyĀAāaĂAăaĄAąaĆCćcĈCĉcĊCċcČCčcĎDďdĒEēeĔEĕeĖEėeĘEęeĚEěeĜGĝgĞGğgĠGġgĢGģgĤHĥhĨIĩiĪIīiĬIĭiĮIįiİIĴJĵjĶKķkĹLĺlĻLļlĽLľlŃNńnŅNņnŇNňnŌOōoŎOŏoŐOőoŔRŕrŖRŗrŘRřrŚSśsŜSŝsŞSşsŠSšsŢTţtŤTťtŨUũuŪUūuŬUŭuŮUůuŰUűuŲUųuŴWŵwŶYŷyŸYŹZźzŻZżzŽZžzƠOơoƯUưuǍAǎaǏIǐiǑOǒoǓUǔuǕUǖuǗUǘuǙUǚuǛUǜuǞAǟaǠAǡaǦGǧgǨKǩkǪOǫoǬOǭoǰjǴGǵgǸNǹnǺAǻaȀAȁaȂAȃaȄEȅeȆEȇeȈIȉiȊIȋiȌOȍoȎOȏoȐRȑrȒRȓrȔUȕuȖUȗuȘSșsȚTțtȞHȟhȦAȧaȨEȩeȪOȫoȬOȭoȮOȯoȰOȱoȲYȳyḀAḁaḂBḃbḄBḅbḆBḇbḈCḉcḊDḋdḌDḍdḎDḏdḐDḑdḒDḓdḔEḕeḖEḗeḘEḙeḚEḛeḜEḝeḞFḟfḠGḡgḢHḣhḤHḥhḦHḧhḨHḩhḪHḫhḬIḭiḮIḯiḰKḱkḲKḳkḴKḵkḶLḷlḸLḹlḺLḻlḼLḽlḾMḿmṀMṁmṂMṃmṄNṅnṆNṇnṈNṉnṊNṋnṌOṍoṎOṏoṐOṑoṒOṓoṔPṕpṖPṗpṘRṙrṚRṛrṜRṝrṞRṟrṠSṡsṢSṣsṤSṥsṦSṧsṨSṩsṪTṫtṬTṭtṮTṯtṰTṱtṲUṳuṴUṵuṶUṷuṸUṹuṺUṻuṼVṽvṾVṿvẀWẁwẂWẃwẄWẅwẆWẇwẈWẉwẊXẋxẌXẍxẎYẏyẐZẑzẒZẓzẔZẕzẖhẗtẘwẙyẠAạaẢAảaẤAấaẦAầaẨAẩaẪAẫaẬAậaẮAắaẰAằaẲAẳaẴAẵaẶAặaẸEẹeẺEẻeẼEẽeẾEếeỀEềeỂEểeỄEễeỆEệeỈIỉiỊIịiỌOọoỎOỏoỐOốoỒOồoỔOổoỖOỗoỘOộoỚOớoỜOờoỞOởoỠOỡoỢOợoỤUụuỦUủuỨUứuỪUừuỬUửuỮUữuỰUựuỲYỳyỴYỵyỶYỷyỸYỹy',
    ).join('').match(/../gu) ?? []
  ).map((pair) => [pair[0]!, pair[1]!]),
);

const SUBSCRIPTS: Record<string, string> = {
  ...Object.fromEntries(Array.from('₀₁₂₃₄₅₆₇₈₉', (c, i) => [c, String(i)])),
  '₊': '+', '₋': '-', 'ₐ': 'a', 'ₑ': 'e', 'ₒ': 'o', 'ₓ': 'x', 'ₕ': 'h', 'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₚ': 'p',
  'ₛ': 's', 'ₜ': 't', 'ᵢ': 'i', 'ⱼ': 'j',
};

const SUPERSCRIPTS: Record<string, string> = {
  ...Object.fromEntries(Array.from('⁰¹²³⁴⁵⁶⁷⁸⁹', (c, i) => [c, String(i)])),
  '⁻': '-', '⁺': '+', 'ⁿ': 'n',
};

const GREEK = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega'.split(' ');

function greek(c: string): string | undefined {
  const cp = c.codePointAt(0)!;
  const index =
    cp >= 0x3b1 && cp <= 0x3c1 ? cp - 0x3b1
    : cp === 0x3c2 ? 17
    : cp >= 0x3c3 && cp <= 0x3c9 ? cp - 0x3b1 - 1
    : cp >= 0x391 && cp <= 0x3a1 ? cp - 0x391
    : cp >= 0x3a3 && cp <= 0x3a9 ? cp - 0x391 - 1
    : -1;
  return GREEK[index];
}

const SYMBOLS: Record<string, string> = {
  '×': 'times', '÷': 'divided by', '≤': 'less than or equal to', '≥': 'greater than or equal to', '≠': 'is not equal to',
  '≈': 'approximately', '√': 'square root of', '∞': 'infinity', '∑': 'sigma', '‰': 'per mille',
};

const DOUBLE_QUOTES = '»«‹›“”„‟″‶⹂❝❞❠〝〞〟＂';
const SINGLE_QUOTES = '‘’‚‛′‵❛❜❟＇ʼʻʹ´`';
const CURRENCY_SYMBOLS = '$€£¥¢';

function isEmoji(c: string): boolean {
  const cp = c.codePointAt(0)!;
  return (cp >= 0x1f000 && cp <= 0x1faff) || (cp >= 0x2600 && cp <= 0x27bf) || (cp >= 0x2b00 && cp <= 0x2bff) || cp === 0xfe0f || cp === 0x200d;
}

const power = (n: string) => (n === '2' ? 'squared' : n === '3' ? 'cubed' : `to the power of ${n}`);

/** Phonon's English normalization (Normalize::for_lang(Lang::En)): the character pass, then the rewrite rules. */
export function normalizeEnglish(input: string): string {
  const res = new Appender();
  const chars = Array.from(input);
  let prev: string | undefined;
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i]!;
    const next = chars[i + 1];
    if (DOUBLE_QUOTES.includes(c)) res.push('"');
    else if (SINGLE_QUOTES.includes(c)) res.push("'");
    else if ('‐‑‒―'.includes(c)) res.push('-');
    else if (c === '★' || c === '☆') {
      let full = c === '★' ? 1 : 0;
      let empty = 1 - full;
      for (; chars[i + 1] === '★' || chars[i + 1] === '☆'; i++) chars[i + 1] === '★' ? full++ : empty++;
      const half = chars[i + 1] === '½';
      if (half) i++;
      const n = half ? `${full}.5` : String(full);
      res.pushSpoken(empty > 0 ? `${n} out of ${full + empty + (half ? 1 : 0)} stars` : `${n} stars`);
    } else if ((c === '°' || c === 'º') && res.endsWithN()) {
      if (res.last() === '.') res.buffer.pop();
      res.pushSpoken(res.buffer.pop() === 'N' ? 'Number' : 'number');
    } else if (c === '°') {
      res.pushSpoken(res.endsWithOne() ? 'degree' : 'degrees');
      const scale = next === 'C' ? 'Celsius' : next === 'F' ? 'Fahrenheit' : null;
      if (scale && !isAlphanumeric(chars[i + 2])) {
        i++;
        res.pushStr(scale);
      }
    } else if (FOLDS.has(c)) res.push(FOLDS.get(c)!);
    else if (SUBSCRIPTS[c]) res.push(SUBSCRIPTS[c]);
    else if (SUPERSCRIPTS[c]) {
      let n = SUPERSCRIPTS[c];
      for (; SUPERSCRIPTS[chars[i + 1] ?? '']; i++) n += SUPERSCRIPTS[chars[i + 1]!];
      if (isAlphanumeric(prev) || prev === ')') res.pushSpoken(power(n));
      else res.pushStr(n);
    } else if (greek(c)) res.pushSpoken(greek(c)!);
    else if (SYMBOLS[c]) res.pushSpoken(SYMBOLS[c]);
    else if (c === '±') res.pushSpoken('plus or minus');
    else if (c === '❤' || c === '♥') {
      if (next === '️') i++;
      res.pushSpoken('love');
    } else if (c === '~' && isDigit(next)) res.pushSpoken('about');
    else if (c === '*' && (prev === undefined || isWhitespace(prev)) && isDigit(next)) res.pushSpoken('star');
    else if ('–*—[]{}•‣◦·→←↑↓➡➜'.includes(c)) res.pushWhitespace();
    // "250 €": a symbol after a number and a space joins the number, so that the currency rule reads it.
    else if (CURRENCY_SYMBOLS.includes(c) && res.endsWithSpacedDigit() && !isDigit(next)) {
      res.popWhitespace();
      res.push(c);
    } else if (c === '…') res.push('.');
    else if (c === '=') res.pushSpoken('equals');
    // A ":" between two non-spaces, as in "10:30", stays; otherwise it, ";" and brackets become a comma.
    else if (c === ':' && prev !== undefined && !isWhitespace(prev) && next !== undefined && !isWhitespace(next)) res.push(':');
    else if (';:()'.includes(c)) {
      res.popWhitespace();
      res.push(',');
      res.pushWhitespace();
    } else if (isEmoji(c) || /^\p{Cc}$/u.test(c) || isWhitespace(c)) res.pushWhitespace();
    else res.push(c);
    prev = c;
  }
  return stripClosingQuote(rewriteText(res.toString()));
}

function stripClosingQuote(text: string): string {
  const trimmed = text.replace(/"+$/, '');
  const rest = trimmed.endsWith("'") ? trimmed.slice(0, -1) : null;
  return rest !== null && /[.!?,;]$/.test(rest) ? rest : trimmed;
}

/**
 * Spell out the "@" and "+" of a word no rule claimed, and give the pieces they leave their own turn at the rules.
 * A slash between words ("and/or") reads as a pause.
 */
function spellSymbols(word: string): string {
  const slashed = splitSlash(word);
  if (slashed !== null) return rewriteEach(slashed);
  if (!word.includes('@') && !word.includes('+')) return word;
  const res = new Appender();
  for (const c of word) {
    if (c === '@') res.pushSpoken('at');
    else if (c === '+') res.pushSpoken('plus');
    else res.push(c);
  }
  return rewriteEach(res.toString());
}

const rewriteEach = (text: string) => text.split(' ').map((w) => rewriteWord(w) ?? w).join(' ');

/** Units written with a slash between letters ("mg/kg"), which keep it. */
const SLASH_UNITS = new Set(
  'km kg mg ml mL dL mol kJ kW kWh Hz min hr hrs sec ft lb lbs oz mi mph kph yd cm mm yr wk mo kcal rpm'.split(' '),
);

/** `word` with its slashes as spaces when every part is a word of two letters or more and not a unit, else null. */
function splitSlash(word: string): string | null {
  let end = word.length;
  while (end > 0 && isAsciiPunctuation(word[end - 1]) && word[end - 1] !== '/') end--;
  const parts = word.slice(0, end).split('/');
  const wordlike = (p: string) => Array.from(p).length >= 2 && Array.from(p).every(isAlphabetic) && !SLASH_UNITS.has(p);
  return parts.length > 1 && parts.every(wordlike) ? parts.join(' ') + word.slice(end) : null;
}

// ---- The rewrite rules (preprocess/rewrite.rs and rewrite/*.rs) ----

/** Rewrite the words of `text`, first letting the readings that span several words claim theirs. */
function rewriteText(text: string): string {
  const words = text.split(' ');
  const out: string[] = [];
  for (let i = 0; i < words.length; ) {
    const rest = words.slice(i);
    const span = scaled(rest) ?? digitRun(rest) ?? endingIn(out[out.length - 1], rest) ?? ranks(rest);
    if (span) {
      out.push(span[0]);
      i += span[1];
    } else {
      out.push(rewriteWord(words[i]!) ?? spellSymbols(words[i]!));
      i++;
    }
  }
  return out.join(' ');
}

/** The first default rule that claims `word`, or null. */
function rewriteWord(word: string): string | null {
  return numbers(word) ?? currency(word) ?? dashedDigits(word) ?? emails(word) ?? urls(word) ?? abbreviations(word) ?? elongations(word);
}

/** The sentence punctuation a rule carries through at the end of a word. */
const PUNCTUATION = '!?.:,;…';

/** `rest` when it is only such punctuation, else null: the rule did not account for the whole word. */
const suffix = (rest: string) => (Array.from(rest).every((c) => PUNCTUATION.includes(c)) ? rest : null);

/** `word` without its trailing punctuation, and that punctuation. */
function splitSuffix(word: string): [string, string] {
  let end = word.length;
  while (end > 0 && PUNCTUATION.includes(word[end - 1]!)) end--;
  return [word.slice(0, end), word.slice(end)];
}

const allDigits = (s: string) => /^[0-9]+$/.test(s);

/** A two-letter part of an address spelled out ("fr" is "F-R", "www" "W-W-W"); anything else as it is. */
function spellShort(word: string): string {
  if (word === 'www' || word === 'WWW') return 'W-W-W';
  return /^[\x00-\x7f]{2}$/.test(word) ? `${word[0]!.toUpperCase()}-${word[1]!.toUpperCase()}` : word;
}

const DIGIT_WORDS = 'zero one two three four five six seven eight nine'.split(' ');

/** Each character a word, digits spelled out, dashes dropped. */
const spell = (s: string) =>
  Array.from(s)
    .filter((c) => c !== '-')
    .map((c) => (isDigit(c) ? DIGIT_WORDS[+c]! : c))
    .join(' ');

/** Numbers with their scale words ("1,234.56" is "1 thousand 234 point 56"); a leading zero marks a code. */
function numbers(word: string): string | null {
  const [body, end] = splitSuffix(word);
  if (body.length >= 2 && body.startsWith('0') && allDigits(body)) return spell(body) + end;
  const fromTo = range(body);
  if (fromTo) return fromTo + end;
  const read = numberWords(word);
  if (read) return read[0] + read[1];
  const dash = word.indexOf('-');
  if (dash < 0) return null;
  const [head, tail] = [word.slice(0, dash), word.slice(dash)];
  if (!isAlphabetic(Array.from(tail.slice(1))[0])) return null;
  const headRead = numberWords(head);
  return headRead && headRead[1] === '' ? headRead[0] + tail : null;
}

/** "20-25" and "1445-1450" (or "2020-21") read "20 to 25", when the range goes up. */
function range(body: string): string | null {
  const dash = body.indexOf('-');
  if (dash < 0) return null;
  const [from, to] = [body.slice(0, dash), body.slice(dash + 1)];
  const small = (s: string) => s.length >= 2 && s.length <= 3 && allDigits(s) && !s.startsWith('0');
  const year = (s: string) => s.length === 4 && allDigits(s) && (s.startsWith('1') || s.startsWith('20'));
  const ok =
    small(from) && small(to) ? +from < +to
    : from.length === 4 && to.length === 4 ? year(from) && year(to) && from < to
    : from.length === 4 && to.length === 2 ? year(from) && allDigits(to) && from.slice(2) < to
    : false;
  return ok ? `${from} to ${to}` : null;
}

/** Runs of digit groups ("+44 20 7946 0958", "555 0142 0199") read digit by digit, a comma between groups. */
function digitRun(words: string[]): [string, number] | null {
  const group = (s: string) => s.length >= 1 && s.length <= 8 && /^[0-9A-Z]+$/.test(s) && /[0-9]/.test(s);
  if (!words.length) return null;
  const plus = words[0]!.startsWith('+');
  const bodies: string[] = [];
  let end = '';
  for (const word of words) {
    const [whole, punctuation] = splitSuffix(word);
    const body = !bodies.length && plus ? whole.slice(1) : whole;
    if (!group(body)) break;
    bodies.push(body);
    if (punctuation) {
      end = punctuation;
      break;
    }
  }
  const years = bodies.every((b) => b.length === 4 && (b.startsWith('19') || b.startsWith('20')));
  const long = bodies.length >= 3 && !years;
  if (!long && !(bodies.length === 2 && (plus || bodies.some((b) => b.startsWith('0'))))) return null;
  const spelled = bodies.map(spell).join(', ');
  return [(plus ? `plus ${spelled}` : spelled) + end, bodies.length];
}

/** The code after "ending in": "-4242" reads "four two four two". */
function endingIn(prev: string | undefined, words: string[]): [string, number] | null {
  if (!prev || !['ending', 'ends', 'ended'].includes(prev.toLowerCase()) || words[0] !== 'in' || words[1] === undefined) return null;
  const [body, end] = splitSuffix(words[1]);
  if (!body.startsWith('-')) return null;
  const code = body.slice(1);
  if (!/[0-9]/.test(code) || !/^[0-9A-Z-]*$/.test(code)) return null;
  return [`in ${spell(code)}${end}`, 2];
}

/** How `word` reads as a number, and the punctuation it carries, apart so that money can put its name between. */
function numberWords(word: string): [string, string] | null {
  if (word[0] === '0' && isDigit(word[1])) return null;
  const negative = word.startsWith('-');
  let rest = negative ? word.slice(1) : word;
  let end = 0;
  while (end < rest.length && (isDigit(rest[end]) || rest[end] === ',')) end++;
  const int = rest.slice(0, end).replace(/,+$/, '');
  if (!int) return null;
  rest = rest.slice(int.length);
  let decimal: string | null = null;
  if (rest.startsWith('.')) {
    const digits = /^[0-9]*/.exec(rest.slice(1))![0];
    if (digits) [decimal, rest] = [digits, rest.slice(1 + digits.length)];
  }
  const punctuation = suffix(rest);
  if (punctuation === null) return null;
  const big = BigInt(int.replaceAll(',', ''));
  if (big >= 1_000_000_000_000n) return null;
  const value = Number(big);
  let words: string;
  if (value < 1000 || (value >= 1900 && value < 2100)) words = String(value);
  else {
    const scales: [number, string][] = [
      [1e9, 'billion'],
      [1e6, 'million'],
      [1e3, 'thousand'],
    ];
    const parts = scales.flatMap(([div, name]) => {
      const count = Math.floor(value / div) % 1000;
      return count ? [`${count} ${name}`] : [];
    });
    if (value % 1000) parts.push(String(value % 1000));
    words = parts.join(' ');
  }
  if (decimal !== null) words = `${words} point ${decimal}`;
  if (negative) words = `minus ${words}`;
  return [words, punctuation];
}

const CURRENCIES: Record<string, [string, string]> = {
  $: ['dollar', 'dollars'],
  '€': ['euro', 'euros'],
  '£': ['pound', 'pounds'],
  '¥': ['yen', 'yen'],
  '¢': ['cent', 'cents'],
};

const SCALE_SUFFIXES: [string, string][] = [
  ['thousand', 'thousand'], ['million', 'million'], ['billion', 'billion'], ['trillion', 'trillion'],
  ['bn', 'billion'], ['BN', 'billion'], ['mn', 'million'], ['tn', 'trillion'], ['k', 'thousand'], ['K', 'thousand'],
  ['m', 'million'], ['M', 'million'], ['b', 'billion'], ['B', 'billion'],
];

/** A number with a currency symbol before or after it ("$4.59" is "4 dollars and 59 cents"). */
function currency(word: string): string | null {
  const [body, end] = splitSuffix(word);
  // "-$500" reads as "$-500" does.
  if (body.startsWith('-') && body.length > 1 && CURRENCY_SYMBOLS.includes(body[1]!)) return currency(`${body[1]}-${body.slice(2)}${end}`);
  const symbol = splitSymbol(body);
  if (!symbol) return null;
  const [sign, amount] = symbol;
  const scale = scaleOf(amount);
  if (scale) {
    const spoken = withScale(scale[0], scale[1], sign);
    return spoken === null ? null : spoken + end;
  }
  const cents = sign === '¢' ? null : withCents(amount, sign);
  if (cents !== null) return cents + end;
  const read = numberWords(amount);
  if (!read) return null;
  // Always the plural, even for 1 ("$1" is "1 dollars"), as the serving stack reads it; only "1¢" is "1 cent".
  const [one, many] = CURRENCIES[sign]!;
  return `${read[0]} ${sign === '¢' && amount === '1' ? one : many}${end}`;
}

function splitSymbol(word: string): [string, string] | null {
  if (word && CURRENCY_SYMBOLS.includes(word[0]!)) return [word[0]!, word.slice(1)];
  if (word && CURRENCY_SYMBOLS.includes(word[word.length - 1]!)) return [word[word.length - 1]!, word.slice(0, -1)];
  return null;
}

/** "40k", "3.5 billion": the number and its scale word. */
function scaleOf(amount: string): [string, string] | null {
  for (const [short, scale] of SCALE_SUFFIXES) {
    if (!amount.endsWith(short)) continue;
    const number = amount.slice(0, -short.length);
    if (isDigit(number[number.length - 1])) return [number, scale];
  }
  return null;
}

function withScale(amount: string, scale: string, sign: string): string | null {
  const read = numberWords(amount);
  return read && read[1] === '' ? `${read[0]} ${scale} ${CURRENCIES[sign]![1]}` : null;
}

/** "$2 million": an amount followed by a scale word. */
function scaled(words: string[]): [string, number] | null {
  if (words.length < 2) return null;
  const [scale, end] = splitSuffix(words[1]!);
  const symbol = splitSymbol(words[0]!);
  if (!symbol || !['thousand', 'million', 'billion', 'trillion'].includes(scale)) return null;
  const spoken = withScale(symbol[1], scale, symbol[0]);
  return spoken === null ? null : [spoken + end, 2];
}

/** An amount with exactly two decimals, read in units and cents. */
function withCents(amount: string, sign: string): string | null {
  const dot = amount.lastIndexOf('.');
  if (dot < 0) return null;
  const [whole, cents] = [amount.slice(0, dot), amount.slice(dot + 1)];
  const grouped = whole.split(',').every((g, i) => allDigits(g) && (i === 0 ? g.length <= 3 : g.length === 3));
  if (!(allDigits(whole) || grouped) || cents.length !== 2 || !allDigits(cents)) return null;
  const read = numberWords(whole);
  if (!read) return null;
  const [one, many] = CURRENCIES[sign]!;
  const units = `${read[0]} ${read[0] === '1' ? one : many}`;
  if (cents === '00') return units;
  // "05" reads "5", and English says cents whatever the currency.
  const count = cents.startsWith('0') ? cents.slice(1) : cents;
  return read[0] === '0' ? `${count} cents` : `${units} and ${count} cents`;
}

/** Digit groups with two or more "-" or "_" ("555-123-4567" is "555 123 4 5 6 7"), but not dates. */
function dashedDigits(word: string): string | null {
  const [body, end] = splitSuffix(word);
  if (!/^[0-9_-]*$/.test(body) || (body.match(/[-_]/g) ?? []).length < 2) return null;
  const groups: [string, string | null][] = [];
  let start = 0;
  for (let i = 0; i < body.length; i++) {
    if (body[i] !== '-' && body[i] !== '_') continue;
    if (i === start) return null;
    groups.push([body.slice(start, i), body[i]!]);
    start = i + 1;
  }
  if (start === body.length) return null;
  groups.push([body.slice(start), null]);
  if (!body.includes('_') && looksLikeDate(groups.map(([g]) => g))) return null;
  const words: string[] = [];
  for (const [group, separator] of groups) {
    if (group.length >= 4) words.push(...group);
    else words.push(group);
    if (separator === '_') words.push('underscore');
  }
  return words.join(' ') + end;
}

function looksLikeDate(groups: string[]): boolean {
  const dayMonth = (x: string, y: string) => {
    if (x.length > 2 || y.length > 2) return false;
    const [a, b] = [+x, +y];
    return (a >= 1 && a <= 31 && b >= 1 && b <= 12) || (a >= 1 && a <= 12 && b >= 1 && b <= 31);
  };
  if (groups.length !== 3) return false;
  const [a, b, c] = groups as [string, string, string];
  return (a.length === 4 && dayMonth(b, c)) || ((c.length === 4 || c.length === 2) && dayMonth(a, b));
}

/** Push the pieces of `part` onto `words`, each separator `spoken` has a word for replaced by it. */
function speakParts(part: string, words: string[], spoken: (c: string) => string | undefined): void {
  let start = 0;
  for (let i = 0; i < part.length; i++) {
    const word = spoken(part[i]!);
    if (word === undefined) continue;
    if (start < i) words.push(part.slice(start, i));
    words.push(word);
    start = i + 1;
  }
  if (start < part.length) words.push(part.slice(start));
}

/** An email address with its separators spoken ("a.b@gra-dium.fr" is "a dot b at gra dash dium dot F-R"). */
function emails(word: string): string | null {
  const at = word.indexOf('@');
  if (at < 0) return null;
  const user = word.slice(0, at);
  if (!/^[A-Za-z0-9_.+-]+$/.test(user)) return null;
  const label = (s: string): [string, string] | null => {
    const n = /^[A-Za-z0-9-]*/.exec(s)![0].length;
    return n ? [s.slice(0, n), s.slice(n)] : null;
  };
  let next = label(word.slice(at + 1));
  if (!next) return null;
  const labels = [next[0]];
  let rest = next[1];
  while (rest.startsWith('.') && (next = label(rest.slice(1)))) {
    labels.push(next[0]);
    rest = next[1];
  }
  const end = suffix(rest);
  if (end === null || labels.length < 2) return null;
  const tld = labels.pop()!;
  const words: string[] = [];
  speakParts(user, words, (c) => ({ '-': 'dash', '.': 'dot', '+': 'plus', _: 'underscore' })[c]);
  words.push('at');
  labels.forEach((l, i) => {
    if (i > 0) words.push('dot');
    speakParts(l, words, (c) => (c === '-' ? 'dash' : undefined));
  });
  words.push('dot');
  return `${words.join(' ')} ${spellShort(tld)}${end}`;
}

/** The run of code points `s` starts with that `keep` accepts, and the rest; null when empty. */
function run(s: string, keep: (c: string) => boolean): [string, string] | null {
  let n = 0;
  for (const c of s) {
    if (!keep(c)) break;
    n += c.length;
  }
  return n ? [s.slice(0, n), s.slice(n)] : null;
}

/** A web address ending in a top-level domain, read with its separators spoken. */
function urls(word: string): string | null {
  const words: string[] = [];
  let rest = word;
  for (const [scheme, spoken] of [['https://', 'H-T-T-P-S'], ['http://', 'H-T-T-P']]) {
    if (!word.startsWith(scheme!)) continue;
    words.push(spoken!, 'colon', 'slash', 'slash');
    rest = word.slice(scheme!.length);
    break;
  }
  const inLabel = (c: string) => isAlphanumeric(c) || c === '-';
  let next = run(rest, inLabel);
  if (!next) return null;
  const labels = [next[0]];
  rest = next[1];
  while (rest.startsWith('.') && (next = run(rest.slice(1), inLabel))) {
    labels.push(next[0]);
    rest = next[1];
  }
  const path = run(rest, (c) => inLabel(c) || c === '/');
  if (path) rest = path[1];
  const end = suffix(rest);
  if (end === null) return null;
  const tld = labels[labels.length - 1]!;
  if (labels.length < 2 || !TLDS.has(tld.toUpperCase())) return null;
  // "it.Now" is a sentence that lost its space, unless a scheme or "www." says otherwise.
  const capitalized = /\p{Uppercase}/u.test(tld) && /\p{Lowercase}/u.test(tld);
  if (capitalized && !words.length && !/^www$/i.test(labels[0]!)) return null;
  labels.forEach((l, i) => {
    if (i > 0) words.push('dot');
    speakAddress(l, words);
  });
  if (path) speakAddress(path[0], words);
  return words.join(' ') + end;
}

/** Dashes and slashes spoken, each digit a word, two-letter pieces spelled out. */
function speakAddress(part: string, words: string[]): void {
  let start = 0;
  for (let i = 0; i < part.length; i++) {
    const c = part[i]!;
    const word = c === '-' ? 'dash' : c === '/' ? 'slash' : isDigit(c) ? c : undefined;
    if (word === undefined) continue;
    if (start < i) words.push(spellShort(part.slice(start, i)));
    words.push(word);
    start = i + 1;
  }
  if (start < part.length) words.push(spellShort(part.slice(start)));
}

const ABBREVIATIONS = new Map<string, string>([
  ['Lt', 'Lieutenant'], ['Lt.', 'Lieutenant'], ['Col.', 'Colonel'], ['Sgt.', 'Sergeant'], ['Sgt', 'Sergeant'],
  ['Capt.', 'Captain'], ['Cpl.', 'Corporal'], ['Gen.', 'General'], ['Adm.', 'Admiral'], ['Sen.', 'Senator'],
  ['Sens.', 'Senators'], ['Rep.', 'Representative'], ['Reps.', 'Representatives'], ['Gov.', 'Governor'],
  ['Pres.', 'President'], ['Det.', 'Detective'], ['Insp.', 'Inspector'], ['Supt.', 'Superintendent'], ['Ft.', 'Fort'],
  ['Mt.', 'Mount'], ['Jan.', 'January'], ['Feb.', 'February'], ['Mar.', 'March'], ['Apr.', 'April'], ['Jun.', 'June'],
  ['Jul.', 'July'], ['Aug.', 'August'], ['Sep.', 'September'], ['Sept.', 'September'], ['Oct.', 'October'],
  ['Nov.', 'November'], ['Dec.', 'December'], ['Mon.', 'Monday'], ['Tue.', 'Tuesday'], ['Tues.', 'Tuesday'],
  ['Wed.', 'Wednesday'], ['Thu.', 'Thursday'], ['Thur.', 'Thursday'], ['Thurs.', 'Thursday'], ['Fri.', 'Friday'],
  ['Sat.', 'Saturday'], ['Sun.', 'Sunday'], ['vol.', 'volume'], ['Vol.', 'Volume'], ['ed.', 'edition'],
  ['approx.', 'approximately'], ['Approx.', 'Approximately'], ['dept.', 'department'], ['Dept.', 'Department'],
  ['vs.', 'versus'], ['w/', 'with'], ['W/', 'With'], ['w/o', 'without'], ['W/o', 'Without'],
]);

function abbreviations(word: string): string | null {
  const [body, end] = splitSuffix(word);
  const dotted = end.startsWith('.');
  const expansion = ABBREVIATIONS.get(dotted ? word.slice(0, body.length + 1) : body);
  return expansion === undefined ? null : expansion + (dotted ? end.slice(1) : end);
}

/** "Lt Col" reads "Lieutenant Colonel". */
function ranks(words: string[]): [string, number] | null {
  if (words.length < 2 || words[0] !== 'Lt') return null;
  const [rank, end] = splitSuffix(words[1]!);
  return rank === 'Col' ? [`Lieutenant Colonel${end}`, 2] : null;
}

/** Stretched words ("Sooooo") lose the letters of runs of three or more vowels, "y", "h" or "w". */
function elongations(word: string): string | null {
  const [body, end] = splitSuffix(word);
  const chars = Array.from(body);
  const stretched = (c: string) => 'aeiouyhw'.includes(c.toLowerCase());
  const triple = (k: number) => chars[k] === chars[k + 1] && chars[k + 1] === chars[k + 2];
  if (
    !/^[A-Za-z]*$/.test(body) ||
    /^[A-Z]*$/.test(body) ||
    !chars.some((c, k) => k + 2 < chars.length && triple(k) && stretched(c))
  ) {
    return null;
  }
  let out = '';
  chars.forEach((c, k) => {
    const repeat = k >= 1 && chars[k - 1] === c && stretched(c);
    const window = chars.slice(Math.max(0, k - 2), Math.min(k + 3, chars.length));
    const long = window.some((_, w) => w + 2 < window.length && window.slice(w, w + 3).every((x) => x === c));
    if (!(repeat && long)) out += c;
  });
  return out + end;
}

// ---- From text to the chunks the model is prompted with (tts_model.rs, plan.rs) ----

/** Capitalize, end with a full stop after a letter or digit, and the frames to run after EOS: 3 for up to 4 words. */
export function prepareTextPrompt(text: string): [string, number] {
  const trimmed = trim(text);
  if (!trimmed) return ['', 3];
  const words = splitWhitespace(trimmed.replace(/[\n\r]/g, ' '));
  const joined = words.join(' ');
  const first = String.fromCodePoint(joined.codePointAt(0)!);
  let prepared = first.toUpperCase() + joined.slice(first.length);
  if (isAlphanumeric(Array.from(prepared).at(-1))) prepared += '.';
  return [prepared, words.length <= 4 ? 3 : 1];
}

/** At most this many frames for `tokens` text tokens: three tokens a second plus two seconds. */
const frameBudget = (tokens: number) => Math.ceil((tokens / 3 + 2) * FRAME_RATE);

interface Chunk extends PhononChunk {
  text: string;
}

function chunk(text: string, tokenizer: PhononTokenizer): Chunk {
  const [prepared, framesAfterEos] = prepareTextPrompt(text);
  const ids = tokenizer.encode(prepared);
  return { text, ids, framesAfterEos, frameBudget: frameBudget(ids.length) };
}

/** Whole sentences grouped into chunks of up to `maxTokens` tokens, cut after the tokenizer's sentence-end tokens. */
function splitIntoBestSentences(text: string, tokenizer: PhononTokenizer, maxTokens: number): string[] {
  const tokens = tokenizer.encode(trim(prepareTextPrompt(text)[0]));
  // The marker's first token carries the leading "▁".
  const marker = tokenizer.encode('.!...?');
  const ends = new Set(marker.length > 1 ? marker.slice(1) : marker);
  const boundaries = [0];
  let afterEnd = false;
  tokens.forEach((id, k) => {
    if (ends.has(id)) afterEnd = true;
    else {
      if (afterEnd) boundaries.push(k);
      afterEnd = false;
    }
  });
  boundaries.push(tokens.length);
  const chunks: string[] = [];
  let current = '';
  let count = 0;
  for (let k = 0; k + 1 < boundaries.length; k++) {
    const [start, end] = [boundaries[k]!, boundaries[k + 1]!];
    const sentence = tokenizer.decode(tokens.slice(start, end));
    if (!current) [current, count] = [sentence, end - start];
    else if (count + end - start > maxTokens) {
      chunks.push(trim(current));
      [current, count] = [sentence, end - start];
    } else {
      current += ' ' + sentence;
      count += end - start;
    }
  }
  if (current) chunks.push(trim(current));
  return chunks;
}

/**
 * Cut every chunk of more than `max` tokens until each fits: at the sentence end nearest its middle, else after the
 * clause mark nearest it, else at the middle word, each side keeping a quarter of `max`; a single word is cut between
 * its characters.
 */
function fit(chunks: Chunk[], max: number, tokenizer: PhononTokenizer): Chunk[] {
  const fitted: Chunk[] = [];
  const todo = [...chunks].reverse();
  for (let next = todo.pop(); next; next = todo.pop()) {
    if (next.ids.length <= max) {
      fitted.push(next);
      continue;
    }
    const words = splitWhitespace(next.text);
    let halves: [string, string];
    if (words.length >= 2) {
      const at = cut(words, Math.ceil(max / 4), tokenizer);
      halves = [words.slice(0, at).join(' '), words.slice(at).join(' ')];
    } else {
      const chars = Array.from(trim(next.text));
      const mid = Math.floor(chars.length / 2);
      if (mid === 0) {
        fitted.push(next);
        continue;
      }
      halves = [chars.slice(0, mid).join(''), chars.slice(mid).join('')];
    }
    todo.push(chunk(halves[1], tokenizer), chunk(halves[0], tokenizer));
  }
  return fitted;
}

/** How many of `words` go before the cut (plan.rs cut). */
function cut(words: string[], minSide: number, tokenizer: PhononTokenizer): number {
  const ends: number[] = [];
  let total = 0;
  for (const word of words) ends.push((total += tokenizer.encode(word).length));
  const inner = ends.slice(0, -1);
  const imbalance = (end: number) => Math.abs(end - (total - end));
  const best = (accept: (k: number) => boolean) => {
    let found = -1;
    inner.forEach((end, k) => {
      if (accept(k) && (found < 0 || imbalance(end) < imbalance(inner[found]!))) found = k;
    });
    return found;
  };
  // A mark can sit inside closing quotes or brackets: `"yes,"`.
  const after = (marks: RegExp) =>
    best((k) => marks.test(words[k]!.replace(/["')\]”’»]+$/u, '')) && inner[k]! >= minSide && total - inner[k]! >= minSide);
  let k = after(/[.!?]$/);
  if (k < 0) k = after(/[,;:]$/);
  if (k < 0) k = best(() => true);
  return k + 1;
}

/**
 * The chunks Phonon's runtime prompts the model with for `text` (plan.rs chunks + fit_or_error): normalized, split into
 * sentence groups of up to 50 tokens, each prepared and tokenized, with its frame budget; any over 200 tokens cut.
 */
export function phononChunks(text: string, tokenizer: PhononTokenizer): PhononChunk[] {
  const sentences = splitIntoBestSentences(normalizeEnglish(text), tokenizer, MAX_TOKENS_PER_CHUNK);
  if (!sentences.length) throw new Error('Nothing to synthesize: the text is empty.');
  const chunks = fit(
    sentences.map((s) => chunk(s, tokenizer)),
    MAX_FIT_TOKENS,
    tokenizer,
  );
  const tooLong = chunks.find((c) => c.ids.length > MAX_FIT_TOKENS);
  if (tooLong) throw new Error(`A text chunk is ${tooLong.ids.length} tokens and cannot fit the ${MAX_FIT_TOKENS}-token limit.`);
  return chunks.map(({ ids, framesAfterEos, frameBudget }) => ({ ids, framesAfterEos, frameBudget }));
}

/** IANA's top-level domains (29 November 2025), as the URL rule's tlds.rs lists them. */
const TLDS = new Set(
  [
    `aaa aarp abb abbott abbvie abc able abogado abudhabi ac academy accenture accountant accountants aco actor ad ads
adult ae aeg aero aetna af afl africa ag agakhan agency ai aig airbus airforce airtel akdn al alibaba alipay
allfinanz allstate ally alsace alstom am amazon americanexpress americanfamily amex amfam amica amsterdam analytics
android anquan anz ao aol apartments app apple aq aquarelle ar arab aramco archi army arpa art arte as asda asia
associates at athleta attorney au auction audi audible audio auspost author auto autos aw aws ax axa az azure ba
baby baidu banamex band bank bar barcelona barclaycard barclays barefoot bargains baseball basketball bauhaus bayern
bb bbc bbt bbva bcg bcn bd be beats beauty beer berlin best bestbuy bet bf bg bh bharti bi bible bid bike bing bingo
bio biz bj black blackfriday blockbuster blog bloomberg blue bm bms bmw bn bnpparibas bo boats boehringer bofa bom
bond boo book booking bosch bostik boston bot boutique box br bradesco bridgestone broadway broker brother brussels
bs bt build builders business buy buzz bv bw by bz bzh ca cab cafe cal call calvinklein cam camera camp canon
capetown capital capitalone car caravan cards care career careers cars casa case cash casino cat catering catholic
cba cbn cbre cc cd center ceo cern cf cfa cfd cg ch chanel channel charity chase chat cheap chintai christmas chrome
church ci cipriani circle cisco citadel citi citic city ck cl claims cleaning click clinic clinique clothing cloud
club clubmed cm cn co coach codes coffee college cologne com commbank community company compare computer comsec
condos construction consulting contact contractors cooking cool coop corsica country coupon coupons courses cpa cr
credit creditcard creditunion cricket crown crs cruise cruises cu cuisinella cv cw cx cy cymru cyou cz dad dance
data date dating datsun day dclk dds de deal dealer deals degree delivery dell deloitte delta democrat dental
dentist desi design dev dhl diamonds diet digital direct directory discount discover dish diy dj dk dm dnp do docs
doctor dog domains dot download drive dtv dubai dupont durban dvag dvr dz earth eat ec eco edeka edu education ee eg
email emerck energy engineer engineering enterprises epson equipment er ericsson erni es esq estate et eu eurovision
eus events exchange expert exposed express extraspace fage fail fairwinds faith family fan fans farm farmers fashion
fast fedex feedback ferrari ferrero fi fidelity fido film final finance financial fire firestone firmdale fish
fishing fit fitness fj fk flickr flights flir florist flowers fly fm fo foo food football ford forex forsale forum
foundation fox fr free fresenius frl frogans frontier ftr fujitsu fun fund furniture futbol fyi ga gal gallery gallo
gallup game games gap garden gay gb gbiz gd gdn ge gea gent genting george gf gg ggee gh gi gift gifts gives giving
gl glass gle global globo gm gmail gmbh gmo gmx gn godaddy gold goldpoint golf goo goodyear goog google gop got gov
gp gq gr grainger graphics gratis green gripe grocery group gs gt gu gucci guge guide guitars guru gw gy hair
hamburg hangout haus hbo hdfc hdfcbank health healthcare help helsinki here hermes hiphop hisamitsu hitachi hiv hk
hkt hm hn hockey holdings holiday homedepot homegoods homes homesense honda horse hospital host hosting hot hotels
hotmail house how hr hsbc ht hu hughes hyatt hyundai ibm icbc ice icu id ie ieee ifm ikano il im imamat imdb immo
immobilien in inc industries infiniti info ing ink institute insurance insure int international intuit investments
io ipiranga iq ir irish is ismaili ist istanbul it itau itv jaguar java jcb je jeep jetzt jewelry jio jll jm jmp jnj
jo jobs joburg jot joy jp jpmorgan jprs juegos juniper kaufen kddi ke kerryhotels kerryproperties kfh kg kh ki kia
kids kim kindle kitchen kiwi km kn koeln komatsu kosher kp kpmg kpn kr krd kred kuokgroup kw ky kyoto kz la lacaixa
lamborghini lamer land landrover lanxess lasalle lat latino latrobe law lawyer lb lc lds lease leclerc lefrak legal
lego lexus lgbt li lidl life lifeinsurance lifestyle lighting like lilly limited limo lincoln link live living lk
llc llp loan loans locker locus lol london lotte lotto love lpl lplfinancial lr ls lt ltd ltda lu lundbeck luxe
luxury lv ly ma madrid maif maison makeup man management mango map market marketing markets marriott marshalls
mattel mba mc mckinsey md me med media meet melbourne meme memorial men menu merckmsd mg mh miami microsoft mil mini
mint mit mitsubishi mk ml mlb mls mm mma mn mo mobi mobile moda moe moi mom monash money monster mormon mortgage
moscow moto motorcycles mov movie mp mq mr ms msd mt mtn mtr mu museum music mv mw mx my mz na nab nagoya name navy
nba nc ne nec net netbank netflix network neustar new news next nextdirect nexus nf nfl ng ngo nhk ni nico nike
nikon ninja nissan nissay nl no nokia norton now nowruz nowtv np nr nra nrw ntt nu nyc nz obi observer office
okinawa olayan olayangroup ollo om omega one ong onl online ooo open oracle orange org organic origins osaka otsuka
ott ovh pa page panasonic paris pars partners parts party pay pccw pe pet pf pfizer pg ph pharmacy phd philips phone
photo photography photos physio pics pictet pictures pid pin ping pink pioneer pizza pk pl place play playstation
plumbing plus pm pn pnc pohl poker politie porn post pr praxi press prime pro prod productions prof progressive
promo properties property protection pru prudential ps pt pub pw pwc py qa qpon quebec quest racing radio re read
realestate realtor realty recipes red redumbrella rehab reise reisen reit reliance ren rent rentals repair report
republican rest restaurant review reviews rexroth rich richardli ricoh ril rio rip ro rocks rodeo rogers room rs
rsvp ru rugby ruhr run rw rwe ryukyu sa saarland safe safety sakura sale salon samsclub samsung sandvik
sandvikcoromant sanofi sap sarl sas save saxo sb sbi sbs sc scb schaeffler schmidt scholarships school schule
schwarz science scot sd se search seat secure security seek select sener services seven sew sex sexy sfr sg sh
shangrila sharp shell shia shiksha shoes shop shopping shouji show si silk sina singles site sj sk ski skin sky
skype sl sling sm smart smile sn sncf so soccer social softbank software sohu solar solutions song sony soy spa
space sport spot sr srl ss st stada staples star statebank statefarm stc stcgroup stockholm storage store stream
studio study style su sucks supplies supply support surf surgery suzuki sv swatch swiss sx sy sydney systems sz tab
taipei talk taobao target tatamotors tatar tattoo tax taxi tc tci td tdk team tech technology tel temasek tennis
teva tf tg th thd theater theatre tiaa tickets tienda tips tires tirol tj tjmaxx tjx tk tkmaxx tl tm tmall tn to
today tokyo tools top toray toshiba total tours town toyota toys tr trade trading training travel travelers
travelersinsurance trust trv tt tube tui tunes tushu tv tvs tw tz ua ubank ubs ug uk unicom university uno uol ups
us uy uz va vacations vana vanguard vc ve vegas ventures verisign versicherung vet vg vi viajes video vig viking
villas vin vip virgin visa vision viva vivo vlaanderen vn vodka volvo vote voting voto voyage vu wales walmart
walter wang wanggou watch watches weather weatherchannel webcam weber website wed wedding weibo weir wf whoswho wien
wiki williamhill win windows wine winners wme wolterskluwer woodside work works world wow ws wtc wtf xbox xerox
xihuan xin xxx xyz yachts yahoo yamaxun yandex ye yodobashi yoga yokohama you youtube yt yun za zappos zara zero zip
zm zone zuerich zw`,
    // The internationalized ones, each after "xn--".
    `11b4c3d 1ck2e1b 1qqw23a 2scrj9c 30rr7y 3bst00m 3ds443g 3e0b707e 3hcrj9c 3pxu8k 42c2d9a 45br5cyl 45brj9c 45q11c
4dbrk0ce 4gbrim 54b7fta0cc 55qw42g 55qx5d 5su34j936bgsg 5tzm5g 6frz82g 6qq986b3xl 80adxhks 80ao21a 80aqecdr1a
80asehdb 80aswg 8y0a063a 90a3ac 90ae 90ais 9dbq2a 9et52u 9krt00a b4w605ferd bck1b9a5dre4c c1avg c2br7g cck2b3b
cckwcxetd cg4bki clchc0ea0b2g2a9gcd czr694b czrs0t czru2d d1acj3b d1alf e1a4c eckvdtc9d efvy88h fct429k fhbei
fiq228c5hs fiq64b fiqs8s fiqz9s fjq720a flw351e fpcrj9c3d fzc2c9e2c fzys8d69uvgm g2xx48c gckr3f0f gecrj9c gk3at1e
h2breg3eve h2brj9c h2brj9c8c hxt814e i1b6b1a6a2e imr513n io0a7i j1aef j1amh j6w193g jlq480n2rg jvr189m kcrx77d1x4a
kprw13d kpry57d kput3i l1acc lgbbat1ad8j mgb9awbf mgba3a3ejt mgba3a4f16a mgba7c0bbn0a mgbaam7a8h mgbab2bd
mgbah1a3hjkrd mgbai9azgqp6j mgbayh7gpa mgbbh1a mgbbh1a71e mgbc0a9azcg mgbca7dzdo mgbcpq6gpa1a mgberp4a5d4ar mgbgu82a
mgbi4ecexp mgbpl2fh mgbt3dhd mgbtx2b mgbx4cd0ab mix891f mk1bu44c mxtq1m ngbc5azd ngbe9e0a ngbrx node nqv7f
nqv7fs00ema nyqy26a o3cw4h ogbpf8fl otu796d p1acf p1ai pgbs0dh pssy2u q7ce6a q9jyb4c qcka1pmc qxa6a qxam rhqv96g
rovu88b rvc1e0am3e s9brj9c ses554g t60b56a tckwe tiq49xqyj unup4y vermgensberater-ctb vermgensberatung-pwb vhquv
vuq861b w4r85el8fhu5dnra w4rs40l wgbh1c wgbl6a xhq521b xkc2al3hye2a xkc2dl3a5ee0h y9a3aq yfro4i67o ygbi2ammx zfr164b`
      .split(/\s+/)
      .map((t) => `xn--${t}`)
      .join(' '),
  ]
    .join(' ')
    .split(/\s+/)
    .map((t) => t.toUpperCase()),
);
