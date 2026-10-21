// Chatterbox's text handling against its Python: the fixtures are what
// tools/export/.cache/tts-eval/chatterbox_fixtures.py prints from resemble-ai/chatterbox's punc_norm (tts.py,
// tts_turbo.py) and Hugging Face's tokenizers. The ids need the tokenizers in the export cache; skipped if missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { CharBpeTokenizer } from './char-bpe';
import { encodeWithTags, prepareLanguage, puncNorm } from './chatterbox-text';
import fixtures from './chatterbox-text.fixtures.json';

describe('puncNorm', () => {
  it.each(fixtures.norms.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))('%s', (_, c) => {
    expect(puncNorm(c.text, 'turbo')).toBe(c.turbo);
    expect(puncNorm(c.text, 'original')).toBe(c.original);
    expect(puncNorm(c.text, 'multilingual')).toBe(c.multilingual);
  });
});

const TURBO = new URL('../../../tools/export/.cache/tts-small/chatterbox-turbo/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TURBO))('encodeWithTags (Turbo)', () => {
  const json = JSON.parse(readFileSync(TURBO, 'utf8')) as { added_tokens: { content: string }[] };
  const tokenizer = BpeTokenizer.fromJSON(json);
  const tags = json.added_tokens.map((t) => t.content).filter((t) => t.startsWith('['));
  it.each(fixtures.turbo.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))('%s', (_, c) => {
    // Turbo's tokenizer ends every text with <|endoftext|> twice.
    const eos = tokenizer.specialTokenId('<|endoftext|>');
    expect([...encodeWithTags(tokenizer, c.text, tags), eos, eos]).toEqual(c.ids);
  });
});

const ORIGINAL = new URL('../../../tools/export/.cache/tts-small/chatterbox/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(ORIGINAL))('CharBpeTokenizer (original Chatterbox)', () => {
  const tokenizer = new CharBpeTokenizer(JSON.parse(readFileSync(ORIGINAL, 'utf8')));
  it.each(fixtures.original.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))('%s', (_, c) => {
    // The template around the text: [EXAGGERATION] [START] … [STOP] [START_SPEECH] [START_SPEECH].
    expect([6563, 255, ...tokenizer.encode(c.text), 0, 6561, 6561]).toEqual(c.ids);
  });
});

const MULTILINGUAL = new URL('../../../tools/export/.cache/tts-small/chatterbox-multilingual/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(MULTILINGUAL))('CharBpeTokenizer (Chatterbox Multilingual)', () => {
  const tokenizer = new CharBpeTokenizer(JSON.parse(readFileSync(MULTILINGUAL, 'utf8')));
  it.each(fixtures.multilingual.map((c) => [c.language, c] as const))('%s', (_, c) => {
    expect([6563, 255, ...tokenizer.encode(prepareLanguage(c.text, c.language)), 0, 6561, 6561]).toEqual(c.ids);
  });
});
