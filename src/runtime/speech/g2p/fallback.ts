// Phonemes for words the lexicon does not know, without eSpeak: short all-capital words are spelled out, compounds
// of known words ("riverbank", "unlockable") are joined from their parts, and anything else goes through the
// letter-to-sound rules with a simple stress guess. Digits inside a word are read as numbers.

import { Lexicon, PRIMARY, VOWELS, applyStress } from './lexicon';
import { letterToSound } from './letter-rules';
import type { Fallback } from './misaki';

/** Affixes that join a known word unstressed (or with secondary stress), as Misaki writes them. */
const PREFIXES = new Map(Object.entries({ un: 'ˌʌn', re: 'ɹˌi', pre: 'pɹˌi', non: 'nˌɑn', dis: 'dˌɪs', mis: 'mˌɪs', anti: 'ˌæntI', sub: 'sˌʌb', super: 'sˌupəɹ', inter: 'ˌɪntəɹ', multi: 'mˌʌlti', micro: 'mˌIkɹO', semi: 'sˌɛmi', over: 'ˌOvəɹ', under: 'ˌʌndəɹ', out: 'ˌWt', co: 'kˌO' }));
const SUFFIXES = new Map(Object.entries({ ness: 'nəs', less: 'ləs', ful: 'fəl', ly: 'li', ment: 'mənt', able: 'əbᵊl', ism: 'ˌɪzəm', ist: 'ɪst', ists: 'ɪsts', er: 'əɹ', ers: 'əɹz', ish: 'ɪʃ', like: 'lˌIk', ship: 'ʃˌɪp', hood: 'hˌʊd', dom: 'dəm', wise: 'wˌIz', ware: 'wˌɛɹ' }));
/** Endings whose stress falls on the syllable just before them. */
const PENULT_ENDINGS = /(tion|sion|cian|ic|ics|ical|ity|ial|ian|ious|eous|uous)$/;
const UNSTRESSED_STARTS = /^(a|be|de|re|con|com|ex|em|en|in|im|pro|pre|per|di|mis|ob|sub|sus|un|e)(?=[a-z])/;

/** Letter names Inflect's normaliser writes out ("AI" → "ay eye") that the lexicon lacks. */
const LETTER_NAMES = new Map([['ay', 'ˈA'], ['ee', 'ˈi'], ['ar', 'ˈɑɹ']]);
const ONSETS =new Set('bl br ch cl cr dr dw fl fr gl gn gr kl kn kr ph pl pn pr ps qu sc sch scr sh shr sk sl sm sn sp sph spl spr squ st str sw th thr tr tw wh wr'.split(' '));

/**
 * Unknown words read letter by letter: short capitals ("NATO" is in the lexicon; "FBI" is not), words of three
 * letters or fewer, words without vowels, and words starting with a cluster English does not ("fbi", "nyc").
 */
function looksSpelled(w: string): boolean {
  const lower = w.toLowerCase();
  if ((w === w.toUpperCase() && w.length <= 5) || lower.length <= 3 || !/[aeiouy]/.test(lower)) return true;
  const onset = /^[^aeiouy]*/.exec(lower)![0];
  return onset.length > 1 && !ONSETS.has(onset);
}

const vowelIndices =(ps: string) => [...ps].flatMap((c, i) => (VOWELS.has(c) ? [i] : []));

/** Puts a primary stress before the vowel the guess picks. */
function stressGuess(ps: string, word: string): string {
  const chars = [...ps];
  const vowels = vowelIndices(ps).filter((i) => chars[i] !== 'ə' && chars[i] !== 'ᵻ');
  if (!vowels.length) return ps;
  let pick = 0;
  if (PENULT_ENDINGS.test(word) && vowels.length > 1) pick = vowels.length - 2;
  else if (vowels.length > 1 && UNSTRESSED_STARTS.test(word)) pick = 1;
  chars.splice(vowels[pick]!, 0, PRIMARY);
  return chars.join('');
}

export function makeFallback(lex: Lexicon): Fallback {
  // Parts of four letters or more: shorter ones split too many unknown words into nonsense ("pho" + "non").
  const known = (w: string) => w.length >= 4 && (lex.golds.has(w) || lex.silvers.has(w));
  const plain = (w: string) => lex.lookup(w, '', null, null);

  /** The word as known parts: [phonemes per part] with the fewest parts (at most three), or null. */
  function parts(w: string, depth = 0): string[] | null {
    if (depth > 0 && known(w)) return [plain(w) ?? ''];
    if (depth > 0 && SUFFIXES.has(w)) return [SUFFIXES.get(w)!];
    if (depth >= 2) return null;
    let best: string[] | null = null;
    for (let i = w.length - 2; i >= 2; i--) {
      const head = w.slice(0, i);
      const first = (depth === 0 ? PREFIXES.get(head) : undefined) ?? (known(head) ? plain(head) : null);
      if (!first) continue;
      const rest = parts(w.slice(i), depth + 1);
      if (rest && (!best || rest.length + 1 < best.length)) best = [first, ...rest];
    }
    return best;
  }

  function word(w: string): string {
    const lower = w.toLowerCase();
    const letterName = LETTER_NAMES.get(lower);
    if (letterName) return letterName;
    // A lower-cased acronym the lexicon has in capitals ("ai"), as Kitten's lower-casing leaves them.
    if (lex.golds.has(w.toUpperCase())) return plain(w.toUpperCase()) ?? '';
    if (looksSpelled(w)) {
      const spelled = lex.getNNP(w);
      if (spelled) return spelled;
    }
    if (lex.isKnown(lower)) return plain(lower) ?? '';
    const split = parts(lower);
    if (split) {
      // The first stressed part keeps its primary stress; the rest are demoted to secondary.
      let seen = false;
      return split.map((p) => {
        if (!p.includes(PRIMARY)) return p;
        if (seen) return applyStress(p, -1)!;
        seen = true;
        return p;
      }).join('');
    }
    return stressGuess(letterToSound(lower), lower);
  }

  return (text) => {
    const out: string[] = [];
    // Accents dropped ("café" → "cafe"), so the lexicon gets a second look.
    const plainText = text.normalize('NFD').replace(/\p{M}/gu, '');
    for (const m of plainText.matchAll(/[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+/g)) {
      const run = m[0];
      out.push(/^[0-9]/.test(run) ? (lex.getNumber(run, null, true, '') ?? '') : word(run.replaceAll("'", '')));
    }
    const ps = out.filter(Boolean).join(' ');
    return ps || null;
  };
}
