// Supertonic's text preparation, as its Python package does it (supertonic/core.py UnicodeProcessor and
// utils.py chunk_text): clean and normalise the text, split it into chunks of whole sentences, and map each
// character to the model's id through its Unicode index.

const EMOJI =
  /[\u{1f600}-\u{1f64f}\u{1f300}-\u{1f5ff}\u{1f680}-\u{1f6ff}\u{1f700}-\u{1f77f}\u{1f780}-\u{1f7ff}\u{1f800}-\u{1f8ff}\u{1f900}-\u{1f9ff}\u{1fa00}-\u{1fa6f}\u{1fa70}-\u{1faff}\u{2600}-\u{26ff}\u{2700}-\u{27bf}\u{1f1e6}-\u{1f1ff}]+/gu;

const SYMBOLS: [string, string][] = [
  ['–', '-'],
  ['‑', '-'],
  ['—', '-'],
  ['¯', ' '],
  ['_', ' '],
  ['“', '"'],
  ['”', '"'],
  ['‘', "'"],
  ['’', "'"],
  ['´', "'"],
  ['`', "'"],
  ['[', ' '],
  [']', ' '],
  ['|', ' '],
  ['/', ' '],
  ['#', ' '],
  ['→', ' '],
  ['←', ' '],
];

const EXPRESSIONS: [string, string][] = [
  ['@', ' at '],
  ['e.g.,', 'for example, '],
  ['i.e.,', 'that is, '],
];

const ENDING = /[.!?;:,'")\]}…。」』】〉》›»]$/;

/** The text as the model reads it: normalised, cleaned, ending in punctuation, wrapped in its language tag. */
export function prepareText(text: string, language: string): string {
  let t = text.normalize('NFKD').replace(EMOJI, '');
  for (const [from, to] of SYMBOLS) t = t.split(from).join(to);
  t = t.replace(/[♥☆♡©\\]/g, '');
  for (const [from, to] of EXPRESSIONS) t = t.split(from).join(to);
  for (const p of [',', '.', '!', '?', ';', ':', "'"]) t = t.split(` ${p}`).join(p);
  t = t.replace(/(["'`])\1+/g, '$1');
  t = t.replace(/\s+/g, ' ').trim();
  if (!ENDING.test(t)) t += '.';
  return `<${language}>${t}</${language}>`;
}

/** Titles whose full stop doesn't end a sentence. */
const TITLES = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'St.', 'vs.', 'etc.', 'e.g.', 'i.e.'];

/** Paragraphs split into sentences, joined back into chunks of at most `maxLength` characters. */
export function chunkText(text: string, maxLength = 300): string[] {
  const chunks: string[] = [];
  for (const paragraph of text.trim().split(/\n\s*\n+/)) {
    const sentences: string[] = [];
    let current = '';
    for (const piece of paragraph.trim().split(/(?<=[.!?])\s+/)) {
      current = current ? `${current} ${piece}` : piece;
      // A title or a single capital letter ("J. Smith") before the break: not the end of a sentence.
      if (TITLES.some((t) => current.endsWith(t)) || /(^|\s)[A-Z]\.$/.test(current)) continue;
      sentences.push(current);
      current = '';
    }
    if (current) sentences.push(current);
    let chunk = '';
    for (const sentence of sentences.map((s) => s.trim()).filter(Boolean)) {
      if (chunk.length + sentence.length + 1 <= maxLength) chunk = chunk ? `${chunk} ${sentence}` : sentence;
      else {
        if (chunk) chunks.push(chunk);
        chunk = sentence;
      }
    }
    if (chunk.trim()) chunks.push(chunk.trim());
  }
  return chunks;
}

/** The model's ids for a prepared text (characters the index lacks, -1 there, become 0). */
export function textIds(prepared: string, indexer: number[]): BigInt64Array {
  const ids = new BigInt64Array(prepared.length);
  for (let i = 0; i < prepared.length; i++) ids[i] = BigInt(Math.max(0, indexer[prepared.charCodeAt(i)] ?? 0));
  return ids;
}
