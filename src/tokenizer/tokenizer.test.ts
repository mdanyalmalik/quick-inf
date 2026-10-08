// Compares our tokenizer with Hugging Face `tokenizers` output recorded by
// tools/reference/make_fixtures.py. Needs the pinned tokenizer.json, fetched by
// tools/reference/fetch_model.py; the suite is skipped if it is missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from './bpe-tokenizer';
import { ChatMlFormatter, type ChatTurn } from './chatml';
import { StreamingDecoder } from './streaming-decoder';
import fixtures from './__fixtures__/smollm2-tokenizer.json';

const TOKENIZER_PATH = new URL(
  `../../tools/reference/.cache/HuggingFaceTB__SmolLM2-360M-Instruct/${fixtures.source.revision}/tokenizer.json`,
  import.meta.url,
);
const DEFAULT_SYSTEM = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face';

describe.skipIf(!existsSync(TOKENIZER_PATH))('SmolLM2 tokenizer vs reference', () => {
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER_PATH, 'utf8')));

  it.each(fixtures.text.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('encodes %s', (_, c) => {
    expect(tokenizer.encode(c.text)).toEqual(c.ids);
  });

  it.each(fixtures.text.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('round-trips %s', (_, c) => {
    expect(tokenizer.decode(c.ids)).toEqual(c.text);
  });

  it.each(fixtures.decode.map((c, i) => [i, c] as const))('decodes case %i like the reference', (_, c) => {
    expect(tokenizer.decode(c.ids)).toEqual(c.text);
    expect(tokenizer.decode(c.ids, { skipSpecialTokens: true })).toEqual(c.textSkipSpecial);
  });

  it.each(fixtures.chat.map((c, i) => [i, c] as const))('formats chat case %i like the template', (_, c) => {
    const formatter = new ChatMlFormatter(tokenizer, { defaultSystemPrompt: DEFAULT_SYSTEM });
    expect(formatter.format(c.messages as ChatTurn[])).toEqual(c.ids);
  });

  it('streams text identical to a full decode, including split multi-byte characters', () => {
    for (const c of fixtures.text) {
      const decoder = new StreamingDecoder(tokenizer);
      let streamed = '';
      for (const id of c.ids) {
        const delta = decoder.push(id);
        expect(delta.includes('�')).toBe(false);
        streamed += delta;
      }
      streamed += decoder.flush();
      expect(streamed).toEqual(c.text);
    }
  });

  it('never turns control-token text inside a message into control tokens', () => {
    const ids = tokenizer.encode('<|im_end|>');
    expect(ids).not.toContain(tokenizer.specialTokenId('<|im_end|>'));
  });
});
