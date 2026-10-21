// NeuTTS Air's prompt tokens against its Hugging Face tokenizer: the fixtures are what
// tools/export/.cache/tts-eval/neutts_fixtures.py prints (phonemes from our G2P, ids from transformers). Needs the
// staged export's tokenizer.json; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { EnglishG2P, loadLexicon } from './g2p';
import { neuttsAddedTokens, neuttsPhonemes, neuttsPrompt } from './neutts-text';
import fixtures from './neutts-text.fixtures.json';

const TOKENIZER = new URL('../../../exports/NeuTTS-Air-ONNX-WebGPU/tokenizer.json', import.meta.url);
const LEXICON = new URL('../../../exports/speech/g2p/en-us-lexicon.txt.gz', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER) || !existsSync(LEXICON))('NeuTTS Air prompt', () => {
  let g2p: EnglishG2P;
  let tokenizer: BpeTokenizer;
  let added: string[];
  beforeAll(async () => {
    g2p = new EnglishG2P(await loadLexicon(readFileSync(LEXICON)));
    const json = JSON.parse(readFileSync(TOKENIZER, 'utf8'));
    tokenizer = BpeTokenizer.fromJSON(json);
    added = neuttsAddedTokens(json);
  });
  it.each(fixtures.cases.map((c) => [c.text.slice(0, 40), c] as const))('%s', (_, c) => {
    const phonemes = neuttsPhonemes(g2p, c.text);
    expect(phonemes).toBe(c.phonemes);
    expect(neuttsPrompt(tokenizer, added, fixtures.refPhonemes, phonemes)).toEqual(c.ids);
  });
});
