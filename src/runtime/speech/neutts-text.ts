// NeuTTS Air's text side: the text as eSpeak-style IPA (our G2P in place of eSpeak), and the prompt as its
// _apply_chat_template builds it for phonemes:
//   user: Convert the text to speech:<|TEXT_PROMPT_START|>{voice phonemes} {text phonemes}<|TEXT_PROMPT_END|>
//   assistant:<|SPEECH_GENERATION_START|>
// followed (by the caller) by the voice's codes as <|speech_N|> tokens. Its tokenizer has an added token for each IPA
// symbol, matched before byte-level BPE.

import type { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { encodeWithTags } from './chatterbox-text';
import type { EnglishG2P } from './g2p';

/** The text's phonemes as NeuTTS joins eSpeak's: words separated by single spaces. */
export function neuttsPhonemes(g2p: EnglishG2P, text: string): string {
  return g2p.ipa(text).split(/\s+/).filter(Boolean).join(' ');
}

/** The tokenizer's plain (not special) added tokens other than the speech codes: the IPA symbols and a few more. */
export function neuttsAddedTokens(tokenizerJson: { added_tokens: { content: string; special: boolean }[] }): string[] {
  return tokenizerJson.added_tokens.filter((t) => !t.special && !t.content.startsWith('<|speech_')).map((t) => t.content);
}

/** The prompt's ids up to and including <|SPEECH_GENERATION_START|>; `added` from neuttsAddedTokens. */
export function neuttsPrompt(tokenizer: BpeTokenizer, added: readonly string[], refPhonemes: string, phonemes: string): number[] {
  const id = (token: string) => tokenizer.specialTokenId(token);
  return [
    ...tokenizer.encode('user: Convert the text to speech:'),
    id('<|TEXT_PROMPT_START|>'),
    ...encodeWithTags(tokenizer, `${refPhonemes} ${phonemes}`.normalize('NFC'), added),
    id('<|TEXT_PROMPT_END|>'),
    ...tokenizer.encode('\nassistant:'),
    id('<|SPEECH_GENERATION_START|>'),
  ];
}
