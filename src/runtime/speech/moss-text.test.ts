// MOSS-TTS-Nano's text handling against its Python: the fixtures are what
// tools/export/.cache/tts-eval/moss_fixtures.py prints from normalize_tts_text (including its own test cases) and
// split_voice_clone_text, with sentencepiece's ids. The chunks need our tokenizer.json (converted from its
// tokenizer.model by tools/export/.cache/tts-eval/moss_tokenizer.py) in the export cache; skipped if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { normalizeTtsText, sentencePieceText, splitText } from './moss-text';
import fixtures from './moss-text.fixtures.json';

describe('normalizeTtsText', () => {
  it.each(fixtures.normalized.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('%s', (_, c) => {
    expect(normalizeTtsText(c.text)).toBe(c.out);
    // The original's own check: normalizing again changes nothing.
    expect(normalizeTtsText(c.out)).toBe(c.out);
  });
});

const TOKENIZER = new URL('../../../tools/export/.cache/tts-small/moss-tokenizer/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER))('splitText', () => {
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  const count = (text: string) => tokenizer.encode(sentencePieceText(text)).length;
  it.each(fixtures.chunks.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))('%s', (_, c) => {
    expect(normalizeTtsText(c.text)).toBe(c.normalized);
    const chunks = splitText(c.normalized, count);
    expect(chunks).toEqual(c.chunks);
    expect(chunks.map((t) => tokenizer.encode(sentencePieceText(t)))).toEqual(c.ids);
  });
});
