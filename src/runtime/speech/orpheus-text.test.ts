// Orpheus 3B's prompt tokens against its Hugging Face tokenizer (fixtures from
// tools/export/.cache/tts-eval/orpheus_fixtures.py), and its audio tokens' SNAC layout. Needs the staged export's
// tokenizer.json for the prompt tests; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import mayaFixtures from './maya1-text.fixtures.json';
import { ORPHEUS_AUDIO_BASE, mayaPrompt, mayaTags, orpheusPrompt, orpheusSnacCodes } from './orpheus-text';
import fixtures from './orpheus-text.fixtures.json';

const TOKENIZER = new URL('../../../exports/Orpheus-3B-ONNX-WebGPU/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER))('orpheusPrompt', () => {
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  it.each(fixtures.cases.map((c) => [`${c.voice}: ${c.text.slice(0, 40)}`, c] as const))('%s', (_, c) => {
    expect(orpheusPrompt(tokenizer, c.voice, c.text)).toEqual(c.ids);
  });
});

const MAYA_TOKENIZER = new URL('../../../exports/Maya1-ONNX-WebGPU/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(MAYA_TOKENIZER))('mayaPrompt', () => {
  const json = JSON.parse(readFileSync(MAYA_TOKENIZER, 'utf8'));
  const tokenizer = BpeTokenizer.fromJSON(json);
  const tags = mayaTags(json);
  it.each(mayaFixtures.cases.map((c) => [c.text.slice(0, 40), c] as const))('%s', (_, c) => {
    expect(mayaPrompt(tokenizer, tags, c.description, c.text)).toEqual(c.ids);
  });
});

describe('orpheusSnacCodes', () => {
  it('splits frames of 7 into the three layers, skipping other tokens and an unfinished frame', () => {
    const frame = (base: number) => Array.from({ length: 7 }, (_, slot) => ORPHEUS_AUDIO_BASE + slot * 4096 + base + slot);
    const tokens = [128258 - 1, ...frame(10), 5, ...frame(20), ...frame(30).slice(0, 3)];
    expect(orpheusSnacCodes(tokens)).toEqual([
      [10, 20],
      [11, 14, 21, 24],
      [12, 13, 15, 16, 22, 23, 25, 26],
    ]);
  });
});
