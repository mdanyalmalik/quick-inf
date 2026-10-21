// English grapheme-to-phoneme front end for the small TTS models, without eSpeak: Misaki's lexicon and rules
// (lexicon.ts, misaki.ts) give Misaki / Kokoro phonemes, the fallback reads unknown words, and ipa.ts renders the
// result in eSpeak's style of IPA for the models trained on eSpeak.

import { makeFallback } from './fallback';
import { inflectFrontend, kittenFrontend, paradeeFrontend, type PhonemeChunk } from './frontends';
import { Lexicon } from './lexicon';
import { loadLexicon, parseLexicon, type LexiconData } from './lexicon-file';
import { wordToIpa } from './ipa';
import { misakiPhonemes, phonemizeTokens, type Fallback } from './misaki';

export { loadLexicon, parseLexicon, type LexiconData, type PhonemeChunk };
export { INFLECT_SYMBOLS, KITTEN_SYMBOLS } from './frontends';
export { wordToIpa } from './ipa';

export interface MisakiOptions {
  /** Read unknown words with the fallback (default true); otherwise they become `unk`. */
  fallback?: boolean;
  unk?: string;
}

export class EnglishG2P {
  readonly lexicon: Lexicon;
  private readonly fallback: Fallback;

  constructor(data: LexiconData) {
    this.lexicon = new Lexicon(data);
    this.fallback = makeFallback(this.lexicon);
  }

  /** Misaki's phoneme string for a sentence, as misaki.en.G2P (American English) writes it. */
  misaki(text: string, options: MisakiOptions = {}): string {
    return misakiPhonemes(this.lexicon, text, options.fallback === false ? undefined : this.fallback, options.unk ?? '');
  }

  /** Kitten TTS's model input, chunk by chunk (its text normalisation included). */
  kitten(text: string): PhonemeChunk[] {
    return kittenFrontend(text, (t) => this.ipa(t));
  }

  /** Inflect v2's model input, chunk by chunk (its text normalisation included). */
  inflect(text: string): PhonemeChunk[] {
    return inflectFrontend(text, (t) => this.ipa(t));
  }

  /** Paradee's model input, sentence by sentence; `vocab` is the model config's symbol table. */
  paradee(text: string, vocab: Record<string, number>): PhonemeChunk[] {
    return paradeeFrontend(text, (t) => this.misaki(t), vocab);
  }

  /** The sentence in eSpeak-style IPA: words separated as written, punctuation kept in place. */
  ipa(text: string): string {
    return phonemizeTokens(this.lexicon, text, this.fallback)
      .map((t) => {
        const ps = t.phonemes ?? '';
        const ipa = /^[^\p{L}]*$/u.test(ps) ? ps : ps.split(' ').map((p) => wordToIpa(p, t.text)).join(' ');
        return ipa + t.whitespace;
      })
      .join('');
  }
}
