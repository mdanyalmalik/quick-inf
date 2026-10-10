// Compares our tokenizer and chat formatting with Hugging Face `tokenizers` and
// each model's own Jinja chat template, as recorded by
// tools/reference/make_fixtures.py. Needs the pinned tokenizer.json files, fetched
// by tools/reference/fetch_model.py; a model's suite is skipped if its file is missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { findModel } from '../catalogue/models';
import { BpeTokenizer } from './bpe-tokenizer';
import { createPromptFormatter, type ChatTurn } from './prompt-format';
import { StreamingDecoder } from './streaming-decoder';
import smollm2 from './__fixtures__/smollm2-tokenizer.json';
import qwen25small from './__fixtures__/qwen2.5-0.5b-tokenizer.json';
import qwen3 from './__fixtures__/qwen3-0.6b-tokenizer.json';
import smollm2large from './__fixtures__/smollm2-1.7b-tokenizer.json';
import qwen25mid from './__fixtures__/qwen2.5-1.5b-tokenizer.json';
import qwen25coder from './__fixtures__/qwen2.5-coder-1.5b-tokenizer.json';
import qwen3mid from './__fixtures__/qwen3-1.7b-tokenizer.json';
import qwen3large from './__fixtures__/qwen3-4b-tokenizer.json';
import llama1b from './__fixtures__/llama-3.2-1b-tokenizer.json';
import llama3b from './__fixtures__/llama-3.2-3b-tokenizer.json';
import phi4 from './__fixtures__/phi-4-mini-tokenizer.json';
import phi35 from './__fixtures__/phi-3.5-mini-tokenizer.json';
import qwen35small from './__fixtures__/qwen3.5-0.8b-tokenizer.json';
import qwen35mid from './__fixtures__/qwen3.5-2b-tokenizer.json';
import qwen35large from './__fixtures__/qwen3.5-4b-tokenizer.json';

const SUITES = [
  { modelId: 'smollm2-360m-instruct', fixtures: smollm2 },
  { modelId: 'qwen2.5-0.5b-instruct', fixtures: qwen25small },
  { modelId: 'qwen3-0.6b', fixtures: qwen3 },
  { modelId: 'smollm2-1.7b-instruct', fixtures: smollm2large },
  { modelId: 'qwen2.5-1.5b-instruct', fixtures: qwen25mid },
  { modelId: 'qwen2.5-coder-1.5b-instruct', fixtures: qwen25coder },
  { modelId: 'qwen3-1.7b', fixtures: qwen3mid },
  { modelId: 'qwen3-4b', fixtures: qwen3large },
  { modelId: 'llama-3.2-1b-instruct', fixtures: llama1b },
  { modelId: 'llama-3.2-3b-instruct', fixtures: llama3b },
  { modelId: 'phi-4-mini-instruct', fixtures: phi4 },
  { modelId: 'phi-3.5-mini-instruct', fixtures: phi35 },
  { modelId: 'qwen3.5-0.8b', fixtures: qwen35small },
  { modelId: 'qwen3.5-2b', fixtures: qwen35mid },
  { modelId: 'qwen3.5-4b', fixtures: qwen35large },
];

for (const { modelId, fixtures } of SUITES as { modelId: string; fixtures: typeof smollm2 }[]) {
  const model = findModel(modelId)!;
  const path = new URL(
    `../../tools/reference/.cache/${fixtures.source.repo.replace('/', '__')}/${fixtures.source.revision}/tokenizer.json`,
    import.meta.url,
  );
  describe.skipIf(!existsSync(path))(`${model.name} tokenizer vs reference`, () => suite(model, fixtures, path));
}

function suite(model: NonNullable<ReturnType<typeof findModel>>, fixtures: typeof smollm2, path: URL) {
  const json = JSON.parse(readFileSync(path, 'utf8')) as {
    normalizer: { type: string } | null;
    added_tokens: { content: string; special: boolean }[];
  };
  const tokenizer = BpeTokenizer.fromJSON(json);
  // What decoding gives back: the input after the model's normalizer.
  const normalized = (text: string) => (json.normalizer?.type === 'NFC' ? text.normalize('NFC') : text);
  // Hugging Face turns non-special added tokens (e.g. Qwen3's "<think>") inside
  // text into control tokens; we deliberately keep message text plain.
  const nonSpecialAdded = json.added_tokens.filter((t) => !t.special).map((t) => t.content);
  const hasAddedTokenText = (text: string) => nonSpecialAdded.some((t) => text.includes(t));
  const plainTextCases = fixtures.text.filter((c) => !hasAddedTokenText(c.text));

  it.each(plainTextCases.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('encodes %s', (_, c) => {
    expect(tokenizer.encode(c.text)).toEqual(c.ids);
  });

  it.each(plainTextCases.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('round-trips %s', (_, c) => {
    expect(tokenizer.decode(c.ids)).toEqual(normalized(c.text));
  });

  it('keeps added-token text (e.g. "<think>") in a message as plain text', () => {
    for (const c of fixtures.text.filter((t) => hasAddedTokenText(t.text))) {
      const ids = tokenizer.encode(c.text);
      const addedIds = new Set(json.added_tokens.map((t) => tokenizer.specialTokenId(t.content)));
      expect(ids.filter((id) => addedIds.has(id))).toEqual([]);
      expect(tokenizer.decode(ids)).toEqual(normalized(c.text));
    }
  });

  it.each(fixtures.decode.map((c, i) => [i, c] as const))('decodes case %i like the reference', (_, c) => {
    expect(tokenizer.decode(c.ids)).toEqual(c.text);
    expect(tokenizer.decode(c.ids, { skipSpecialTokens: true })).toEqual(c.textSkipSpecial);
  });

  it.each(fixtures.chat.map((c, i) => [i, c] as const))('formats chat case %i like the template', (_, c) => {
    const kind = model.promptFormat.kind;
    if (kind === 'model-template') throw new Error(`${model.name} has no ONNX prompt format`);
    const date = (fixtures.templateKwargs as { date_string?: string }).date_string;
    const formatter = createPromptFormatter(kind, tokenizer, { ...model.promptFormat, date });
    const turns = c.messages as ChatTurn[];
    expect(formatter.format(turns)).toEqual(c.ids);
    // The budget arithmetic adds up to the same length.
    const fixed = formatter.fixedLength(turns[0]?.role === 'system');
    expect(fixed + turns.reduce((n, t) => n + formatter.turnLength(t), 0)).toEqual(c.ids.length);
  });

  it('streams text identical to a full decode, including split multi-byte characters', () => {
    for (const c of plainTextCases) {
      const decoder = new StreamingDecoder(tokenizer);
      let streamed = '';
      for (const id of c.ids) {
        const delta = decoder.push(id);
        expect(delta.includes('�')).toBe(false);
        streamed += delta;
      }
      streamed += decoder.flush();
      expect(streamed).toEqual(normalized(c.text));
    }
  });

  it('never turns control-token text inside a message into control tokens', () => {
    for (const stop of model.promptFormat.stopTokens) {
      expect(tokenizer.encode(stop)).not.toContain(tokenizer.specialTokenId(stop));
    }
  });
}
