// A small rule tagger standing in for spaCy's Penn Treebank tags in Misaki's pipeline. Misaki reads tags for
// punctuation, numbers, currency signs, addresses, a few function words ("a", "the", "in", "I", "am", "used") and
// heteronyms (noun / verb / past forms: "record", "read", "live"); the tags below come from the neighbouring words.

import type { MToken } from './misaki';

const DETERMINERS = new Set(['a', 'an', 'the', 'this', 'these', 'those', 'every', 'each', 'another', 'some', 'any', 'no', 'my', 'your', 'his', 'its', 'our', 'their', 'whose', 'of']);
const PRONOUNS = new Set(['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'us', 'them']);
const PREPOSITIONS = new Set(['in', 'on', 'at', 'by', 'for', 'from', 'with', 'of', 'into', 'onto', 'upon', 'about', 'above', 'after', 'before', 'below', 'between', 'during', 'through', 'under', 'without', 'within', 'across', 'against', 'along', 'among', 'around', 'behind', 'beside', 'beyond', 'near', 'since', 'toward', 'towards', 'until', 'via', 'than', 'vs', 'vs.']);
const MODALS = new Set(['will', 'would', 'can', 'could', 'should', 'shall', 'may', 'might', 'must']);
const BEFORE_BASE_VERB = new Set([...MODALS, 'to', 'do', 'does', 'did', "don't", "doesn't", "didn't", "won't", "can't", 'cannot', "let's", 'please', 'not', "wouldn't", "couldn't", "shouldn't"]);
const BEFORE_PARTICIPLE = new Set(['have', 'has', 'had', 'having', 'been', 'was', 'were', 'is', 'are', 'be', 'being', "i've", "you've", "we've", "they've", "it's", "he's", "she's"]);
const CONJUNCTIONS = new Set(['and', 'or', 'but', 'nor', 'yet', 'so']);
const CLAUSE_STARTS = new Set(['i', 'you', 'he', 'she', 'it', 'we', 'they', 'the', 'a', 'an', 'this', 'there', 'my', 'your', 'his', 'her', 'our', 'their']);

const PUNCT_TAG: [RegExp, string][] = [
  [/^[.!?]+$/, '.'],
  [/^,$/, ','],
  [/^([:;—–-]+|\.\.\.|…)$/, ':'],
  [/^[([{]$/, '-LRB-'],
  [/^[)\]}]$/, '-RRB-'],
  [/^[“«]$/, '``'],
  [/^[”»]$/, "''"],
  [/^[$£€]$/, '$'],
  [/^[%&+@]$/, 'SYM'],
];

const isPunctuation = (s: string) => /^[^\p{L}\p{N}]+$/u.test(s);
const isNumber = (s: string) => /^[-+]?[\d.,]*\d[\d.,]*(?:st|nd|rd|th|s|'s|ing|ed|'d)?$/i.test(s);
const isAddress = (s: string) => /^(https?:\/\/|www\.)\S+$|^[\w.+-]+@[\w-]+(\.[\w-]+)+$/i.test(s);

function punctuationTag(text: string, prev: MToken | undefined): string {
  if (text === '"') return !prev || prev.whitespace || isPunctuation(prev.text) ? '``' : "''";
  return PUNCT_TAG.find(([re]) => re.test(text))?.[1] ?? 'NFP';
}

function wordTag(t: MToken, prevToken: MToken | undefined, next: MToken | undefined, sentenceStart: boolean): string {
  const w = t.text.toLowerCase();
  const prev = prevToken ? prevToken.text.toLowerCase() : '';
  if (w === 'a' && t.text === 'A' && !sentenceStart) return next && /^\p{Ll}/u.test(next.text) ? 'DT' : 'NNP';
  // "that" as a pronoun or determiner ("that is", "that book") is stressed; before a clause ("that he") it is not.
  if (w === 'that') return !next || isPunctuation(next.text) || !CLAUSE_STARTS.has(next.text.toLowerCase()) ? 'DT' : 'IN';
  if (DETERMINERS.has(w) && w !== 'of') return 'DT';
  if (t.text === 'I' || (PRONOUNS.has(w) && w !== 'i')) return 'PRP';
  if (w === 'to') return 'TO';
  if (w === 'am') return isNumber(prev) ? 'NN' : 'VBP';
  if (w === 'used') return 'VBD';
  if (PREPOSITIONS.has(w)) return 'IN';
  if (MODALS.has(w)) return 'MD';
  if (CONJUNCTIONS.has(w)) return 'CC';
  // Short or vowelless capitals are acronyms (spelled unless the lexicon says otherwise); longer ones may be shouting.
  if (t.text.length > 1 && t.text === t.text.toUpperCase() && /\p{L}/u.test(t.text)) return t.text.length <= 3 || !/[AEIOUY]/.test(t.text) ? 'NNP' : '';
  if (!sentenceStart && /^\p{Lu}/u.test(t.text)) return 'NNP';
  if (BEFORE_BASE_VERB.has(prev)) return 'VB';
  if (prev === 'i' || prev === 'you' || prev === 'we' || prev === 'they') return w.endsWith('ed') ? 'VBD' : 'VBP';
  // After a third-person subject: "lives" is present, "wound" past.
  if (prev === 'he' || prev === 'she' || prev === 'it') return w.endsWith('s') ? 'VBZ' : 'VBD';
  if ((prevToken?.tag === 'NN' || prevToken?.tag === 'NNP') && w.endsWith('s')) return 'VBZ';
  if (BEFORE_PARTICIPLE.has(prev)) return 'VBN';
  if (DETERMINERS.has(prev)) return 'NN';
  return '';
}

/** Gives each token a Penn Treebank style tag from its own text and its neighbours. */
export function tagTokens(tokens: MToken[]): MToken[] {
  tokens.forEach((t, i) => {
    const prev = tokens[i - 1];
    if (isPunctuation(t.text)) t.tag = punctuationTag(t.text, prev);
    else if (isNumber(t.text)) t.tag = 'CD';
    else if (isAddress(t.text)) t.tag = 'ADD';
    else {
      let k = i - 1;
      while (k >= 0 && /^["“'‘(]$/.test(tokens[k]!.text)) k--;
      const sentenceStart = k < 0 || /^[.!?…]+$/.test(tokens[k]!.text) || tokens[k]!.text === ':';
      t.tag = wordTag(t, prev, tokens[i + 1], sentenceStart);
    }
  });
  return tokens;
}
