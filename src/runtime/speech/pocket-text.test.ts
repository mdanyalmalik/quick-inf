// Pocket TTS's text handling against Kyutai's Python: the fixtures are what
// tools/export/.cache/tts-eval/pocket_fixtures.py prints from pocket_tts's text_chunking.py (English settings) with
// the model's tokenizer.json. The chunks need that tokenizer in exports/; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { UnigramTokenizer } from '../../tokenizer/unigram-tokenizer';
import { pocketChunks, preparePocketText, splitPocketText } from './pocket-text';
import fixtures from './pocket-text.fixtures.json';

const cases = fixtures.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const);

describe('preparePocketText', () => {
  it.each(cases)('%s', (_, c) => expect(preparePocketText(c.text)[0]).toBe(c.prepared));
});

const TOKENIZER = new URL('../../../exports/Pocket-TTS-ONNX-WebGPU/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER))('pocketChunks', () => {
  const tokenizer = UnigramTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  it.each(cases)('%s', (_, c) => {
    expect(splitPocketText(c.text, tokenizer)).toEqual(c.chunks);
    expect(pocketChunks(c.text, tokenizer)).toEqual(c.out);
  });
});
