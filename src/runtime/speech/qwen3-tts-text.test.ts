// Qwen3-TTS's prompt tokens against its Hugging Face tokenizer: the fixtures are what
// tools/export/.cache/tts-eval/qwen3_tts_fixtures.py prints. Needs the tokenizer.json in the export cache; skipped
// if it's missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { qwenInstruct, qwenPrompt } from './qwen3-tts-text';
import fixtures from './qwen3-tts-text.fixtures.json';

const TOKENIZER = new URL('../../../tools/export/.cache/tts-medium/qwen3-tts-0.6b-cv-meta/tokenizer.json', import.meta.url);

describe.skipIf(!existsSync(TOKENIZER))('qwenPrompt', () => {
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  it.each(fixtures.cases.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))('%s', (_, c) => {
    expect(qwenPrompt(tokenizer, c.text)).toEqual({ role: c.role, body: c.body });
  });
  it('tokenizes a voice instruction', () => expect(qwenInstruct(tokenizer, fixtures.instruct)).toEqual(fixtures.instructIds));
});
