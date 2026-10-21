// What each small TTS model's own Python frontend feeds its network, rebuilt on the browser G2P: Kitten TTS
// (TextPreprocessor, eSpeak IPA, tokenised and re-joined with spaces), Inflect v2 (normalize_text, eSpeak IPA with
// punctuation) and Paradee (Misaki phonemes per sentence). Each chunk carries its text, the phoneme string and the
// model's token ids.

import { inflectChunks, inflectNormalize } from './inflect-text';
import { kittenChunks, kittenNormalize } from './kitten-text';

export interface PhonemeChunk {
  /** The chunk's text after the model's normalisation (Kitten picks its voice style by this text's length). */
  text: string;
  /** The phoneme string the model's symbol table reads. */
  phonemes: string;
  /** Token ids, with the model's start / end / blank tokens. */
  ids: number[];
}

const IPA_LETTERS = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

/** Symbol → id, the last position winning for repeated symbols (as the Python dicts are built). */
function symbolTable(symbols: string): Map<string, number> {
  const map = new Map<string, number>();
  [...symbols].forEach((c, i) => map.set(c, i));
  return map;
}

/** Kitten TTS's TextCleaner symbols (its punctuation list repeats the straight double quote). */
export const KITTEN_SYMBOLS = symbolTable(`$;:,.!?¡¿—…"«»"" ${LETTERS}${IPA_LETTERS}`);
/** Inflect v2's runtime/text/symbols.py. */
export const INFLECT_SYMBOLS = symbolTable(`_;:,.!?¡¿—…"«»“” ${LETTERS}${IPA_LETTERS}`);

const WORDS_AND_MARKS = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;

/** eSpeak-style IPA for a sentence (see EnglishG2P.ipa). */
export type IpaFn = (text: string) => string;
/** Misaki phonemes for a sentence (see EnglishG2P.misaki). */
export type MisakiFn = (text: string) => string;

/** Kitten TTS: [0, ...ids, 0] per chunk; unknown symbols are dropped, as its TextCleaner does. */
export function kittenFrontend(text: string, ipa: IpaFn): PhonemeChunk[] {
  return kittenChunks(kittenNormalize(text)).map((chunk) => {
    // phonemizer's output, split into words and marks and re-joined with spaces (basic_english_tokenize).
    const phonemes = (ipa(chunk).match(WORDS_AND_MARKS) ?? []).join(' ');
    const ids = [...phonemes].flatMap((c) => (KITTEN_SYMBOLS.has(c) ? [KITTEN_SYMBOLS.get(c)!] : []));
    return { text: chunk, phonemes, ids: [0, ...ids, 0] };
  });
}

/** Inflect v2: ids interleaved with the blank 0 ([0, a, 0, b, ..., 0]) per chunk. */
export function inflectFrontend(text: string, ipa: IpaFn): PhonemeChunk[] {
  return inflectChunks(text).flatMap((chunk) => {
    const normalized = inflectNormalize(chunk);
    const phonemes = [...ipa(normalized).replace(/[“”]/g, '"')].filter((c) => INFLECT_SYMBOLS.has(c)).join('').replace(/\s+/g, ' ').trim();
    if (!phonemes) return [];
    const ids = [...phonemes].flatMap((c) => [0, INFLECT_SYMBOLS.get(c)!]);
    return [{ text: normalized, phonemes, ids: [...ids, 0] }];
  });
}

const PARADEE_MAX = 510;

/**
 * Paradee: one chunk per sentence (split after . ! ? … or at line breaks), long ones cut at the last space before
 * 510 phonemes. `vocab` is the "vocab" of the model's config.json; phonemes outside it are dropped.
 */
export function paradeeFrontend(text: string, misaki: MisakiFn, vocab: Record<string, number>): PhonemeChunk[] {
  const chunks: PhonemeChunk[] = [];
  const push = (sentence: string, ps: string) => {
    const ids = [...ps].flatMap((c) => (c in vocab ? [vocab[c]!] : [])).slice(0, PARADEE_MAX);
    chunks.push({ text: sentence, phonemes: ps, ids: [0, ...ids, 0] });
  };
  for (const sentence of text.trim().split(/(?<=[.!?…])\s+|\n+/)) {
    if (!sentence.trim()) continue;
    let ps = misaki(sentence);
    while (ps.length > PARADEE_MAX) {
      const space = ps.lastIndexOf(' ', PARADEE_MAX - 1);
      const cut = space > 0 ? space : PARADEE_MAX;
      push(sentence, ps.slice(0, cut));
      ps = ps.slice(cut).trimStart();
    }
    if (ps) push(sentence, ps);
  }
  return chunks;
}
