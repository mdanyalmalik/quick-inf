// Zonos v0.1's text side: the text as eSpeak-style IPA (our G2P in place of eSpeak, words joined by single spaces),
// and its phoneme ids as zonos.conditioning.tokenize_phonemes makes them: BOS, one id per character (4 + its index in
// the symbol list; unknown characters 1), EOS.

import type { EnglishG2P } from './g2p';

export interface ZonosSymbols {
  symbols: string[];
  symbol_offset: number;
  unknown_id: number;
  bos_id: number;
  eos_id: number;
}

export function zonosPhonemes(g2p: EnglishG2P, text: string): string {
  return g2p.ipa(text).split(/\s+/).filter(Boolean).join(' ');
}

export function zonosPhonemeIds(config: ZonosSymbols, phonemes: string): number[] {
  const index = new Map(config.symbols.map((s, i) => [s, i + config.symbol_offset]));
  return [config.bos_id, ...Array.from(phonemes, (c) => index.get(c) ?? config.unknown_id), config.eos_id];
}
