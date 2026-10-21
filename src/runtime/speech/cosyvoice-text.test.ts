// CosyVoice 3's English text handling against upstream: the fixtures are what
// tools/export/.cache/tts-eval/cosyvoice_fixtures.py prints from its frontend (wetext, inflect, split_paragraph)
// and the Qwen tokenizer. The ids need the tokenizer in exports/; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { cardinal, normalizeEnglish, splitParagraph } from './cosyvoice-text';
import fixtures from './cosyvoice-text.fixtures.json';

const TOKENIZER = new URL('../../../exports/tts-int4-small/cosyvoice3/tokenizer/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER))('CosyVoice text', () => {
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  const count = (t: string) => tokenizer.encode(t).length;
  it.each(fixtures.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('%s', (_, c) => {
    const chunks = splitParagraph(normalizeEnglish(c.text), count);
    expect(chunks).toEqual(c.chunks);
    expect(chunks.map((t) => tokenizer.encode(t))).toEqual(c.ids);
  });
});

describe('cardinal', () => {
  it('reads numbers as wetext does', () => {
    expect(cardinal('1234')).toBe('one thousand two hundred and thirty four');
    expect(cardinal('1005')).toBe('one thousand and five');
    expect(cardinal('2000000')).toBe('two million');
    expect(cardinal('0142')).toBe('oh one hundred and forty two');
  });
});
