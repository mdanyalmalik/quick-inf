// Reads the pronunciation lexicon file (exports/speech/g2p/en-us-lexicon.txt.gz, built by
// tools/export/build_g2p_lexicon.py from Misaki's us_gold.json and us_silver.json). The app downloads it as a model
// file; nothing of it is bundled. Format, one entry per line after a section line (@gold / @silver):
//   <p><rest>\t<phonemes>   the word is the previous word's first p characters (p in base 36) followed by <rest>;
//   phonemes use the ascii stand-ins of the "!symbols <ascii> <ipa>" header line, and heteronyms are written
//   "<default>|TAG=<phonemes>|..." (an empty value means none).

/** A heteronym: phonemes by part-of-speech tag (Misaki's keys: DEFAULT, NOUN, VERB, ADJ, VBD, None, ...). */
export type TaggedPhonemes = Record<string, string | null>;
export type Entry = string | TaggedPhonemes;

export interface LexiconData {
  golds: Map<string, Entry>;
  silvers: Map<string, Entry>;
}

const BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz';

/** Parses the lexicon text and grows each table as Misaki's Lexicon.grow_dictionary does. */
export function parseLexicon(text: string): LexiconData {
  const tables = { gold: new Map<string, Entry>(), silver: new Map<string, Entry>() };
  let table: Map<string, Entry> | undefined;
  let decode = (s: string) => s;
  let prev = '';
  for (const line of text.split('\n')) {
    if (!line || line[0] === '#') continue;
    if (line.startsWith('!symbols ')) {
      const [, ascii = '', ipa = ''] = line.split(' ');
      const map = new Map([...ascii].map((c, i) => [c, [...ipa][i]!]));
      decode = (s) => [...s].map((c) => map.get(c) ?? c).join('');
      continue;
    }
    if (line[0] === '@') {
      table = tables[line.slice(1) as 'gold' | 'silver'];
      prev = '';
      continue;
    }
    const tab = line.indexOf('\t');
    const word = prev.slice(0, BASE36.indexOf(line[0]!)) + line.slice(1, tab);
    prev = word;
    table?.set(word, parseEntry(decode(line.slice(tab + 1))));
  }
  return { golds: grow(tables.gold), silvers: grow(tables.silver) };
}

function parseEntry(value: string): Entry {
  if (!value.includes('|')) return value;
  const [first = '', ...rest] = value.split('|');
  const entry: TaggedPhonemes = { DEFAULT: first || null };
  for (const part of rest) {
    const eq = part.indexOf('=');
    entry[part.slice(0, eq)] = part.slice(eq + 1) || null;
  }
  return entry;
}

const capitalize = (w: string) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();

/** Misaki's grow_dictionary: a capitalised copy of each lower-case word and a lower-case copy of each capitalised one. */
function grow(d: Map<string, Entry>): Map<string, Entry> {
  const extra = new Map<string, Entry>();
  for (const [k, v] of d) {
    if (k.length < 2) continue;
    if (k === k.toLowerCase()) {
      if (k !== capitalize(k)) extra.set(capitalize(k), v);
    } else if (k === capitalize(k)) {
      extra.set(k.toLowerCase(), v);
    }
  }
  for (const [k, v] of d) extra.set(k, v);
  return extra;
}

/** The lexicon from the downloaded file's bytes (gzip, or plain text). */
export async function loadLexicon(bytes: ArrayBuffer | Uint8Array): Promise<LexiconData> {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (data[0] !== 0x1f || data[1] !== 0x8b) return parseLexicon(new TextDecoder().decode(data));
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('gzip'));
  return parseLexicon(await new Response(stream).text());
}
