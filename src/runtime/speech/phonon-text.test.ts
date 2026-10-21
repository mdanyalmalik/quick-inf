// Phonon's text front end against Gradium's Rust runtime: the fixtures are what its dump_phonon example
// (ptts/examples/dump_phonon.rs, --tokens-only --lang en) prints for each text, the normalized text and the chunks it
// prompts the model with. The chunks need the checkpoint's tokenizer.json in the export cache; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { UnigramTokenizer } from '../../tokenizer/unigram-tokenizer';
import { normalizeEnglish, phononChunks, prepareTextPrompt } from './phonon-text';
import fixtures from './phonon-text.fixtures.json';

interface Fixture {
  text: string;
  normalized?: string;
  chunks?: { ids: number[]; framesAfterEos: number; frameBudget: number }[];
  error?: string;
}

const cases = (fixtures as Fixture[]).map((c) => [JSON.stringify(c.text).slice(0, 60), c] as const);
const TOKENIZER = new URL('../../../tools/export/.cache/phonon/src/tokenizer.json', import.meta.url);

describe('normalizeEnglish', () => {
  it.each(cases.filter(([, c]) => c.normalized !== undefined))('normalizes %s', (_, c) =>
    expect(normalizeEnglish(c.text)).toBe(c.normalized),
  );
});

describe('prepareTextPrompt', () => {
  it('capitalizes, ends a trailing letter or digit with a full stop, and gives short text a longer tail', () => {
    expect(prepareTextPrompt('not a thing')).toEqual(['Not a thing.', 3]);
    expect(prepareTextPrompt('one\r\ntwo   three\nfour five')).toEqual(['One two three four five.', 1]);
    expect(prepareTextPrompt('is it?')).toEqual(['Is it?', 3]);
    expect(prepareTextPrompt('éclair au chocolat')).toEqual(['Éclair au chocolat.', 3]);
    expect(prepareTextPrompt('  \n ')).toEqual(['', 3]);
  });
});

describe.skipIf(!existsSync(TOKENIZER))('phononChunks', () => {
  const tokenizer = UnigramTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));

  it.each(cases)('chunks %s', (_, c) => {
    if (c.error) expect(() => phononChunks(c.text, tokenizer)).toThrow(/empty/);
    else expect(phononChunks(c.text, tokenizer)).toEqual(c.chunks);
  });

  it('decodes as Hugging Face does: the first token loses its "▁", byte pieces join into characters', () => {
    const ids = tokenizer.encode('Ł﻿x and ü');
    expect(tokenizer.decode(ids)).toBe('Ł﻿x and ü');
    expect(tokenizer.decode(tokenizer.encode('. b'))).toBe('. b');
  });
});
