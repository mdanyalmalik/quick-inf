// Zonos's phoneme ids against its own tokenize_phonemes: the fixtures are what
// tools/export/.cache/tts-eval/zonos_fixtures.py prints (phonemes from our G2P, ids from the zonos package). Needs the
// staged export's zonos.json; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { EnglishG2P, loadLexicon } from './g2p';
import { zonosPhonemeIds, zonosPhonemes, type ZonosSymbols } from './zonos-text';
import fixtures from './zonos-text.fixtures.json';

const CONFIG = new URL('../../../exports/Zonos-v0.1-ONNX-WebGPU/zonos.json', import.meta.url);
const LEXICON = new URL('../../../exports/speech/g2p/en-us-lexicon.txt.gz', import.meta.url);

describe.skipIf(!existsSync(CONFIG) || !existsSync(LEXICON))('Zonos phoneme ids', () => {
  let g2p: EnglishG2P;
  let config: ZonosSymbols;
  beforeAll(async () => {
    g2p = new EnglishG2P(await loadLexicon(readFileSync(LEXICON)));
    config = JSON.parse(readFileSync(CONFIG, 'utf8'));
  });
  it.each(fixtures.cases.map((c) => [c.text.slice(0, 40), c] as const))('%s', (_, c) => {
    const phonemes = zonosPhonemes(g2p, c.text);
    expect(phonemes).toBe(c.phonemes);
    expect(zonosPhonemeIds(config, phonemes)).toEqual(c.ids);
  });
});
