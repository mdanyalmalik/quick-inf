// Misaki phonemes rendered in the IPA style eSpeak NG writes for en-us (what Kitten TTS and Inflect were trained
// on), with rules only, no eSpeak data. The mapping:
//   Misaki  A   I   O   W   Y   ʤ   ʧ   T/ɾ  ᵊ   ɐ   ᵻ   ʔ
//   IPA     eɪ  aɪ  oʊ  aʊ  ɔɪ  dʒ  tʃ  ɾ    ə   ɐ   ᵻ   ʔ
//   r-coloured  əɹ → ɚ (ɚɹ before a vowel), ɜɹ → ɜː (ɜːɹ before a vowel), ɔɹ → ɔːɹ, ɑɹ → ɑːɹ, ʊɹ → ʊɹɹ before a vowel
//   length      ɑ → ɑː; u → uː; ɜ → ɜː; i → iː, except unstressed in a last syllable ("happy" hˈæpi) or before a
//               vowel ("video" vˈɪdɪoʊ); ɔ → ɔː where the spelling has au / aw / al / ough / augh / wa ("talk"),
//               else ɔ ("on", "coffee")
//   reduced     a first unstressed ə is ɐ in a word spelled a- ("about" ɐbˈaʊt), ᵻ in re- / be- / de- / e- words
//               ("return" ɹᵻtˈɜːn), ɛ in ex- words ("expect" ɛkspˈɛkt); -ular is jʊl, ə w before a vowel after
//               a palatal is uː ("actual" ˈæktʃuːəl), "-ntion" nʃ, a final ᵻst is ɪst
//               unstressed ə spelled i (or e before one consonant and a vowel, or a final -t / -s / -d) is ɪ
//               ("family" fˈæmɪli, "hundred" hˈʌndɹɪd); "the" before a vowel is ðɪ
//   other       n before k / ɡ is ŋ ("include" ɪŋklˈuːd), ʔn at the end is syllabic ʔn̩ ("button" bˈʌʔn̩)
// Stress marks stay where Misaki puts them (before the vowel, as eSpeak does).

import { VOWELS } from './lexicon';

const SIMPLE: Record<string, string> = { A: 'eɪ', I: 'aɪ', O: 'oʊ', W: 'aʊ', Y: 'ɔɪ', ʤ: 'dʒ', ʧ: 'tʃ', T: 'ɾ', ᵊ: 'ə' };
const STRESS = new Set('ˈˌ');
const LONG_AW = /au|aw|al|ough|augh|wa/;
const isVowel = (c: string | undefined) => !!c && (VOWELS.has(c) || c === 'ᵊ');

/** Whether a vowel comes at index i, skipping stress marks. */
function vowelAt(ps: string[], i: number): boolean {
  while (i < ps.length && STRESS.has(ps[i]!)) i++;
  return isVowel(ps[i]);
}

/** Misaki-level rewrites that need the spelling: reduced first vowels, -ular, -ntion, -est. */
function respell(ps: string[], spelling: string): string[] {
  const out = [...ps];
  const first = out.findIndex(isVowel);
  const unstressedFirst = first >= 0 && !STRESS.has(out[first - 1] ?? '') && out.slice(first + 1).some(isVowel);
  if (unstressedFirst && (out[first] === 'ə' || out[first] === 'i')) {
    if (first === 0 && out[0] === 'ə' && spelling.startsWith('a')) out[0] = 'ɐ';
    else if (/^(re|be|de|e)[^aeiou]/.test(spelling)) out[first] = 'ᵻ';
  }
  if (spelling.startsWith('ex') && unstressedFirst && first === 0 && (out[0] === 'ɪ' || out[0] === 'ə') && 'kɡ'.includes(out[1] ?? '')) out[0] = 'ɛ';
  for (let i = 0; i + 2 < out.length; i++) {
    if (out[i] === 'j' && out[i + 1] === 'ə' && out[i + 2] === 'l' && spelling.includes('ul')) out[i + 1] = 'ʊ';
    if (out[i] === 'n' && out[i + 1] === 'ʧ' && 'əᵊ'.includes(out[i + 2]!) && /ti[ao]/.test(spelling)) out[i + 1] = 'ʃ';
  }
  if (out.slice(-3).join('') === 'ᵻst' && spelling.endsWith('est')) out[out.length - 3] = 'ɪ';
  return unreduce(out, spelling);
}

/**
 * eSpeak keeps an unstressed i, and an e before one consonant and a vowel (or a final -t / -s / -ss / -st / -ct),
 * as ɪ where Misaki writes ə ("family", "target", "element"). Vowels are matched to the spelling's vowel groups
 * when their counts agree.
 */
function unreduce(ps: string[], spelling: string): string[] {
  const groups = [...spelling.replace(/([^aeiouy])e$/, '$1').matchAll(/[aeiouy]+/g)];
  const vowels = ps.flatMap((c, i) => (isVowel(c) ? [i] : []));
  if (groups.length !== vowels.length) return ps;
  vowels.forEach((i, k) => {
    if ((ps[i] !== 'ə' && ps[i] !== 'ᵊ') || STRESS.has(ps[i - 1] ?? '') || ps[i + 1] === 'ɹ') return;
    const g = groups[k]!;
    const after = spelling.slice(g.index + g[0].length);
    if ((g[0] === 'i' || (g[0] === 'y' && after)) && !/^[aeiou]|^ng?$/.test(after)) ps[i] = 'ɪ';
    else if (g[0] === 'e' && (/^[^aeiouy][aeiouy]/.test(after) || /^(t|s|d|ss|st|ct)$/.test(after))) ps[i] = 'ɪ';
  });
  return ps;
}

/** One word's Misaki phonemes in eSpeak-style IPA; `spelling` is the word as written (it decides a few vowels). */
export function wordToIpa(misaki: string, spelling = ''): string {
  if (misaki === 'ði') return 'ðɪ'; // "the" before a vowel
  const s = spelling.toLowerCase();
  const ps = respell([...misaki], s);
  const vowelCount = ps.filter(isVowel).length;
  const lastVowel = ps.reduce((last, c, i) => (isVowel(c) ? i : last), -1);
  const longAw = LONG_AW.test(s);
  let out = '';
  for (let i = 0; i < ps.length; i++) {
    const c = ps[i]!;
    const next = ps[i + 1];
    const stressed = i > 0 && STRESS.has(ps[i - 1]!);
    if (c === 'ə' && next === 'ɹ') {
      out += vowelAt(ps, i + 2) ? 'ɚɹ' : 'ɚ';
      i++;
    } else if (c === 'ə' && next === 'w' && vowelAt(ps, i + 2) && 'ʧʤjʃʒ'.includes(ps[i - 1] ?? '-')) {
      out += 'uː';
      i++;
    } else if (c === 'ɜ') {
      out += 'ɜː';
      if (next === 'ɹ') {
        if (vowelAt(ps, i + 2)) out += 'ɹ';
        i++;
      }
    } else if (c === 'ʊ' && next === 'ɹ' && vowelAt(ps, i + 2)) {
      out += 'ʊɹɹ';
      i++;
    } else if (c === 'ɑ') {
      out += 'ɑː';
    } else if (c === 'ɔ') {
      out += next === 'ɹ' || longAw ? 'ɔː' : 'ɔ';
    } else if (c === 'i') {
      const short = !stressed && ((i === lastVowel && vowelCount > 1) || vowelAt(ps, i + 1));
      out += short ? (vowelAt(ps, i + 1) ? 'ɪ' : 'i') : 'iː';
    } else if (c === 'u') {
      out += 'uː';
    } else if (c === 'n' && (next === 'k' || next === 'ɡ')) {
      out += 'ŋ';
    } else if (c === 'ʔ' && next === 'n' && (i + 2 === ps.length || !vowelAt(ps, i + 2))) {
      out += 'ʔn̩';
      i++;
    } else {
      out += SIMPLE[c] ?? c;
    }
  }
  return out;
}
