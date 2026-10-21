// VoxCPM2's text ids against the voxcpm package's tokenizer wrapper: the fixtures are what
// tools/export/.cache/tts-eval/voxcpm2_fixtures.py prints. Needs the staged export's tokenizer.json; skipped if it's
// missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { voxcpmIds, voxcpmSplits } from './voxcpm-text';
import fixtures from './voxcpm-text.fixtures.json';

const TOKENIZER = new URL('../../../exports/VoxCPM2-ONNX-WebGPU/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER))('voxcpmIds', () => {
  const json = JSON.parse(readFileSync(TOKENIZER, 'utf8'));
  const tokenizer = BpeTokenizer.fromJSON(json);
  const splits = voxcpmSplits(json);
  it.each(fixtures.cases.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))('%s', (_, c) => {
    expect(voxcpmIds(tokenizer, splits, c.text)).toEqual(c.ids);
  });
});
