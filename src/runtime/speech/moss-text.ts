// MOSS-TTS-Nano's text handling (OpenMOSS-Team/MOSS-TTS-Nano, Apache-2.0), ported from its Python:
// tts_robust_normalizer_single_script.py normalize_tts_text (a clean-up for robustness, not a reading of numbers:
// URLs, e-mails, file names and the like are protected, brackets become quotes, dashes and line breaks sentence ends,
// repeated punctuation collapsed, spaces fixed around Chinese and Japanese) and onnx_tts_runtime.py
// split_voice_clone_text (sentences packed into chunks of at most 75 tokens). Python's Unicode \w and \d are spelled
// out as Unicode classes. Tests compare with the Python on fixtures (moss-text.test.ts).

const CJK_CHARS = '\\u3400-\\u4dbf\\u4e00-\\u9fff\\u3040-\\u30ff';
const CJK = `[${CJK_CHARS}]`;
const PROT = '___PROT\\d+___';
/** Python's \w: letters, digits and numbers, underscore. */
const W = '\\p{L}\\p{N}_';
const LATINISH = `(?:${PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)`;

const PROTECTED: RegExp[] = [
  /https?:\/\/[^\s　，。！？；、）】》〉」』]+/gu,
  new RegExp(`(?<![${W}.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}(?![${W}.-])`, 'gu'),
  /(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}/gu,
  /(?<![A-Za-z0-9_])(?:u|r)\/[A-Za-z0-9_]+/gu,
  /(?<![A-Za-z0-9_])#(?!\s)[^\s#]+/gu,
  /(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+/gu,
  /(?<![A-Za-z0-9_])(?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[./+:-])[A-Za-z0-9][A-Za-z0-9._/+:-]*(?![A-Za-z0-9_])/gu,
];
const TRAILING_CLOSERS = new Set('"\')]}）】》〉」』”’');
const FLOW_ARROWS = /\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*/gu;

/** normalize_tts_text: MOSS-TTS-Nano's robustness clean-up of the text before tokenizing. */
export function normalizeTtsText(text: string): string {
  text = baseCleanup(text);
  text = markdownAndLines(text);
  text = text.replace(FLOW_ARROWS, '，');
  const protectedSpans: string[] = [];
  for (const pattern of PROTECTED) {
    text = text.replace(pattern, (m) => `___PROT${protectedSpans.push(m) - 1}___`);
  }
  text = text
    .split(new RegExp(`(${PROT})`, 'u'))
    .map((part) => (new RegExp(`^${PROT}$`, 'u').test(part) ? part : part.replaceAll('_', ' ')))
    .join('');
  text = normalizeSpaces(text);
  text = structuralPunctuation(text);
  text = repeatedPunctuation(text);
  text = normalizeSpaces(text);
  protectedSpans.forEach((original, i) => (text = text.replaceAll(`___PROT${i}___`, original)));
  text = text.trim();
  if (!text) return text;
  return text
    .split('\n')
    .map((line) => (line.trim() ? terminalPunctuation(line.trim()) : ''))
    .join('\n')
    .trim();
}

function baseCleanup(text: string): string {
  text = text.replaceAll('\r\n', '\n').replaceAll('\r', '\n').replaceAll('　', ' ');
  text = text.replace(/[​-‍﻿]/gu, '');
  return Array.from(text)
    .filter((c) => '\n\t '.includes(c) || !/\p{C}/u.test(c))
    .join('');
}

/** Python's str.splitlines() breaks, of those left after the clean-up (the other control characters are gone). */
const LINE_BREAKS = new RegExp('\\n|\\u2028|\\u2029', 'u');

function markdownAndLines(text: string): string {
  text = text.replace(/\[([^[\]]+?)\]\((https?:\/\/[^)\s]+)\)/gu, '$1 $2');
  const lines: string[] = [];
  for (const raw of text.split(LINE_BREAKS)) {
    let line = raw.trim();
    if (!line) continue;
    line = line.replace(/^#{1,6}\s+/u, '').replace(/^>\s+/u, '').replace(/^[-*+]\s+/u, '').replace(/^\p{Nd}+[.)]\s+/u, '');
    lines.push(line);
  }
  if (lines.length === 0) return '';
  return lines.map((line, i) => (i < lines.length - 1 ? terminalPunctuation(line) : line)).join('');
}

function normalizeSpaces(text: string): string {
  text = text.replace(/[ \t\r\f\v]+/gu, ' ');
  text = text.replace(new RegExp(`(${CJK})\\s+(?=${CJK})`, 'gu'), '$1');
  text = text.replace(new RegExp(`(${CJK})\\s+(?=\\p{Nd})`, 'gu'), '$1');
  text = text.replace(new RegExp(`(\\p{Nd})\\s+(?=${CJK})`, 'gu'), '$1');
  text = text.replace(new RegExp(`(${CJK})(?=(${LATINISH}))`, 'gu'), '$1 ');
  text = text.replace(new RegExp(`((${LATINISH}))(?=${CJK})`, 'gu'), '$1 ');
  text = text.replace(/ {2,}/gu, ' ');
  text = text.replace(/\s+([，。！？；：、”’」』】）》])/gu, '$1');
  text = text.replace(/([（【「『《“‘])\s+/gu, '$1');
  text = text.replace(/([，。！？；：、])\s*/gu, '$1');
  text = text.replace(/\s+([,.;!?])/gu, '$1');
  return text.replace(/ {2,}/gu, ' ').trim();
}

function structuralPunctuation(text: string): string {
  text = text.replace(/\[\s*([^[\]]+?)\s*\]/gu, '"$1"');
  text = text.replace(/\{\s*([^{}]+?)\s*\}/gu, '"$1"');
  text = text.replace(/[【〖『「]\s*([^】〗』」]+?)\s*[】〗』」]/gu, '"$1"');
  text = text.replace(/(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))/gu, '$1$2');
  text = text.replace(FLOW_ARROWS, '，');
  return text.replace(/\s*(?:—|–|―|-){2,}\s*/gu, '。');
}

function repeatedPunctuation(text: string): string {
  text = text.replace(/(?:\.{3,}|…{2,}|……+)/gu, '。');
  text = text.replace(/[。．]{2,}/gu, '。');
  text = text.replace(/[，,]{2,}/gu, '，');
  text = text.replace(/[!！]{2,}/gu, '！');
  text = text.replace(/[?？]{2,}/gu, '？');
  return text.replace(/[!?！？]{2,}/gu, (s) => {
    const q = /[?？]/u.test(s);
    const e = /[!！]/u.test(s);
    return q && e ? '？！' : q ? '？' : '！';
  });
}

/** A full stop (。) unless the text already ends in punctuation (before any closing quotes or brackets). */
function terminalPunctuation(text: string): string {
  if (!text) return text;
  const chars = Array.from(text);
  let i = chars.length - 1;
  while (i >= 0 && /\s/u.test(chars[i]!)) i--;
  while (i >= 0 && TRAILING_CLOSERS.has(chars[i]!)) i--;
  return i >= 0 && /\p{P}/u.test(chars[i]!) ? text : `${text}。`;
}

const SENTENCE_END = new Set('.!?。！？；;');
const CLAUSE_SPLIT = new Set(',，、；;：:');
const CLOSING = new Set('"\'”’)]}）】》」』');

function containsCjk(text: string): boolean {
  return /[一-鿿㐀-䶿぀-ヿ가-힯]/u.test(text);
}

/** Python's str.split(): the whitespace-separated words. */
function words(text: string): string[] {
  return text.split(/\s+/u).filter(Boolean);
}

function prepareForChunking(text: string): string {
  let t = text.trim().replaceAll('\r', ' ').replaceAll('\n', ' ');
  while (t.includes('  ')) t = t.replaceAll('  ', ' ');
  const last = Array.from(t).pop()!;
  if (containsCjk(t)) return SENTENCE_END.has(last) ? t : `${t}。`;
  if (/\p{Ll}/u.test(Array.from(t)[0]!)) t = Array.from(t)[0]!.toUpperCase() + Array.from(t).slice(1).join('');
  if (/[\p{L}\p{N}]/u.test(last)) t += '.';
  return words(t).length < 5 ? `        ${t}` : t;
}

function splitByPunctuation(text: string, punctuation: Set<string>): string[] {
  const chars = Array.from(text);
  const sentences: string[] = [];
  let current: string[] = [];
  let i = 0;
  while (i < chars.length) {
    current.push(chars[i]!);
    if (punctuation.has(chars[i]!)) {
      let ahead = i + 1;
      while (ahead < chars.length && CLOSING.has(chars[ahead]!)) current.push(chars[ahead++]!);
      const sentence = current.join('').trim();
      if (sentence) sentences.push(sentence);
      current = [];
      while (ahead < chars.length && /\s/u.test(chars[ahead]!)) ahead++;
      i = ahead;
      continue;
    }
    i++;
  }
  const tail = current.join('').trim();
  if (tail) sentences.push(tail);
  return sentences;
}

function joinParts(left: string, right: string): string {
  if (!left) return right;
  if (!right) return left;
  return containsCjk(left) || containsCjk(right) ? left + right : `${left} ${right}`;
}

/** Pieces of at most `maxTokens` tokens, each cut at a space or punctuation within its last 25 characters if any. */
function splitByTokenBudget(text: string, maxTokens: number, count: (text: string) => number): string[] {
  let remaining = text.trim();
  const pieces: string[] = [];
  const boundaries = new Set([...CLAUSE_SPLIT, ...SENTENCE_END, ' ']);
  while (remaining) {
    if (count(remaining) <= maxTokens) {
      pieces.push(remaining);
      break;
    }
    const chars = Array.from(remaining);
    let [low, high, best] = [1, chars.length, 1];
    while (low <= high) {
      const middle = (low + high) >> 1;
      const candidate = chars.slice(0, middle).join('').trim();
      if (!candidate) low = middle + 1;
      else if (count(candidate) <= maxTokens) {
        best = middle;
        low = middle + 1;
      } else high = middle - 1;
    }
    let cut = best;
    for (let i = best - 1; i > Math.max(-1, best - 25); i--) {
      if (boundaries.has(chars[i]!)) {
        cut = i + 1;
        break;
      }
    }
    let piece = chars.slice(0, cut).join('').trim();
    if (!piece) {
      piece = chars.slice(0, best).join('').trim();
      cut = best;
    }
    pieces.push(piece);
    remaining = chars.slice(cut).join('').trim();
  }
  return pieces;
}

/**
 * split_voice_clone_text: sentences (then clauses, then token-budget pieces for very long ones) packed into chunks
 * of at most `maxTokens` tokens; a text that fits in one chunk is returned as it was.
 */
export function splitText(text: string, count: (text: string) => number, maxTokens = 75): string[] {
  const normalized = text.trim();
  if (!normalized) return [];
  const prepared = prepareForChunking(normalized);
  const sentences = splitByPunctuation(prepared, SENTENCE_END);
  const slices: [number, string][] = [];
  for (const sentence of sentences.length ? sentences : [prepared.trim()]) {
    const s = sentence.trim();
    if (!s) continue;
    const n = count(s);
    if (n <= maxTokens) {
      slices.push([n, s]);
      continue;
    }
    let clauses = splitByPunctuation(s, CLAUSE_SPLIT);
    if (clauses.length <= 1) clauses = [s];
    for (const clause of clauses) {
      const c = clause.trim();
      if (!c) continue;
      const m = count(c);
      if (m <= maxTokens) slices.push([m, c]);
      else for (const piece of splitByTokenBudget(c, maxTokens, count)) if (piece.trim()) slices.push([count(piece.trim()), piece.trim()]);
    }
  }
  const chunks: string[] = [];
  let chunk = '';
  let chunkTokens = 0;
  for (const [n, s] of slices) {
    if (!chunk) [chunk, chunkTokens] = [s, n];
    else if (chunkTokens + n > maxTokens) {
      chunks.push(chunk.trim());
      [chunk, chunkTokens] = [s, n];
    } else {
      chunk = joinParts(chunk, s);
      chunkTokens = count(chunk);
    }
  }
  if (chunk) chunks.push(chunk.trim());
  return chunks.length > 1 ? chunks : [normalized];
}

/** The pause after a chunk (seconds): longer after a chunk of four words or fewer. */
export function chunkPause(chunk: string): number {
  return words(chunk.trim()).length <= 4 ? 0.4 : 0.24;
}

/** Text as SentencePiece's nmt_nfkc normalizer and extra-whitespace removal leave it, before the BPE tokenizer. */
export function sentencePieceText(text: string): string {
  return text.normalize('NFKC').replace(/\s+/gu, ' ').trim();
}
