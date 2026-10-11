// Compares our tokenizer and chat formatting with Hugging Face `tokenizers` and
// each model's own Jinja chat template, as recorded by
// tools/reference/make_fixtures.py. Needs the pinned tokenizer.json files, fetched
// by tools/reference/fetch_model.py; a model's suite is skipped if its file is missing.

import { existsSync, readFileSync } from 'node:fs';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
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














// New models (2026-10-11)
import fx_deepseek_r1_qwen_1_5b from './__fixtures__/deepseek-r1-qwen-1.5b-tokenizer.json';
import fx_qwen3_4b_2507 from './__fixtures__/qwen3-4b-2507-tokenizer.json';
import fx_gemma_3_4b from './__fixtures__/gemma-3-4b-tokenizer.json';
import fx_smollm2_135m from './__fixtures__/smollm2-135m-tokenizer.json';
import fx_gemma_3_270m from './__fixtures__/gemma-3-270m-tokenizer.json';
import fx_lfm2_5_350m from './__fixtures__/lfm2.5-350m-tokenizer.json';
import fx_granite_4_0_350m from './__fixtures__/granite-4.0-350m-tokenizer.json';
import fx_qwen2_5_coder_0_5b from './__fixtures__/qwen2.5-coder-0.5b-tokenizer.json';
import fx_lfm2_700m from './__fixtures__/lfm2-700m-tokenizer.json';
import fx_gemma_3_1b from './__fixtures__/gemma-3-1b-tokenizer.json';
import fx_granite_4_0_1b from './__fixtures__/granite-4.0-1b-tokenizer.json';
import fx_lfm2_5_1_2b from './__fixtures__/lfm2.5-1.2b-tokenizer.json';
import fx_qwen2_5_math_1_5b from './__fixtures__/qwen2.5-math-1.5b-tokenizer.json';
import fx_lfm2_2_6b from './__fixtures__/lfm2-2.6b-tokenizer.json';
import fx_qwen2_5_coder_3b from './__fixtures__/qwen2.5-coder-3b-tokenizer.json';
import fx_granite_4_0_micro from './__fixtures__/granite-4.0-micro-tokenizer.json';
import fx_phi_3_mini from './__fixtures__/phi-3-mini-tokenizer.json';
import fx_olmo_3_7b from './__fixtures__/olmo-3-7b-tokenizer.json';
import fx_lfm2_8b_a1b from './__fixtures__/lfm2-8b-a1b-tokenizer.json';
import fx_lfm2_24b_a2b from './__fixtures__/lfm2-24b-a2b-tokenizer.json';
import fx_phi_4_mini_reasoning from './__fixtures__/phi-4-mini-reasoning-tokenizer.json';
import fx_mistral_7b_v0_2 from './__fixtures__/mistral-7b-v0.2-tokenizer.json';
import fx_llama_3_1_8b from './__fixtures__/llama-3.1-8b-tokenizer.json';
import fx_deepseek_r1_llama_8b from './__fixtures__/deepseek-r1-llama-8b-tokenizer.json';
import fx_gemma_2_9b from './__fixtures__/gemma-2-9b-tokenizer.json';
import fx_phi_3_medium from './__fixtures__/phi-3-medium-tokenizer.json';
import fx_phi_4 from './__fixtures__/phi-4-tokenizer.json';
import fx_phi_4_reasoning from './__fixtures__/phi-4-reasoning-tokenizer.json';

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
  { modelId: 'smollm2-135m-instruct', fixtures: fx_smollm2_135m },
  { modelId: 'gemma-3-270m-it', fixtures: fx_gemma_3_270m },
  { modelId: 'lfm2.5-350m', fixtures: fx_lfm2_5_350m },
  { modelId: 'granite-4.0-350m', fixtures: fx_granite_4_0_350m },
  { modelId: 'qwen2.5-coder-0.5b-instruct', fixtures: fx_qwen2_5_coder_0_5b },
  { modelId: 'qwen2.5-math-1.5b-instruct', fixtures: fx_qwen2_5_math_1_5b },
  { modelId: 'qwen2.5-coder-3b-instruct', fixtures: fx_qwen2_5_coder_3b },
  { modelId: 'lfm2-700m', fixtures: fx_lfm2_700m },
  { modelId: 'gemma-3-1b-it', fixtures: fx_gemma_3_1b },
  { modelId: 'granite-4.0-1b', fixtures: fx_granite_4_0_1b },
  { modelId: 'lfm2.5-1.2b-instruct', fixtures: fx_lfm2_5_1_2b },
  { modelId: 'lfm2-2.6b', fixtures: fx_lfm2_2_6b },
  { modelId: 'granite-4.0-micro', fixtures: fx_granite_4_0_micro },
  { modelId: 'phi-3-mini-4k-instruct', fixtures: fx_phi_3_mini },
  { modelId: 'lfm2-8b-a1b', fixtures: fx_lfm2_8b_a1b },
  { modelId: 'lfm2-24b-a2b', fixtures: fx_lfm2_24b_a2b },
  { modelId: 'olmo-3-7b-instruct', fixtures: fx_olmo_3_7b },
    { modelId: 'phi-4-mini-reasoning', fixtures: fx_phi_4_mini_reasoning },
  { modelId: 'mistral-7b-instruct-v0.2', fixtures: fx_mistral_7b_v0_2 },
  { modelId: 'llama-3.1-8b-instruct', fixtures: fx_llama_3_1_8b },
  { modelId: 'deepseek-r1-distill-llama-8b', fixtures: fx_deepseek_r1_llama_8b },
  { modelId: 'gemma-2-9b-it', fixtures: fx_gemma_2_9b },
  { modelId: 'phi-3-medium-4k-instruct', fixtures: fx_phi_3_medium },
  { modelId: 'phi-4', fixtures: fx_phi_4 },
  { modelId: 'phi-4-reasoning', fixtures: fx_phi_4_reasoning },
    { modelId: 'smollm2-135m-instruct', fixtures: fx_smollm2_135m },
  { modelId: 'gemma-3-270m-it', fixtures: fx_gemma_3_270m },
  { modelId: 'lfm2.5-350m', fixtures: fx_lfm2_5_350m },
  { modelId: 'granite-4.0-350m', fixtures: fx_granite_4_0_350m },
  { modelId: 'qwen2.5-coder-0.5b-instruct', fixtures: fx_qwen2_5_coder_0_5b },
  { modelId: 'lfm2-700m', fixtures: fx_lfm2_700m },
  { modelId: 'gemma-3-1b-it', fixtures: fx_gemma_3_1b },
  { modelId: 'granite-4.0-1b', fixtures: fx_granite_4_0_1b },
  { modelId: 'lfm2.5-1.2b-instruct', fixtures: fx_lfm2_5_1_2b },
  { modelId: 'qwen2.5-math-1.5b-instruct', fixtures: fx_qwen2_5_math_1_5b },
  { modelId: 'lfm2-2.6b', fixtures: fx_lfm2_2_6b },
  { modelId: 'qwen2.5-coder-3b-instruct', fixtures: fx_qwen2_5_coder_3b },
  { modelId: 'granite-4.0-micro', fixtures: fx_granite_4_0_micro },
  { modelId: 'phi-3-mini-4k-instruct', fixtures: fx_phi_3_mini },
  { modelId: 'olmo-3-7b-instruct', fixtures: fx_olmo_3_7b },
  { modelId: 'lfm2-8b-a1b', fixtures: fx_lfm2_8b_a1b },
  { modelId: 'lfm2-24b-a2b', fixtures: fx_lfm2_24b_a2b },
  { modelId: 'phi-4-mini-reasoning', fixtures: fx_phi_4_mini_reasoning },
  { modelId: 'mistral-7b-instruct-v0.2', fixtures: fx_mistral_7b_v0_2 },
  { modelId: 'llama-3.1-8b-instruct', fixtures: fx_llama_3_1_8b },
  { modelId: 'deepseek-r1-distill-llama-8b', fixtures: fx_deepseek_r1_llama_8b },
  { modelId: 'gemma-2-9b-it', fixtures: fx_gemma_2_9b },
  { modelId: 'phi-3-medium-4k-instruct', fixtures: fx_phi_3_medium },
  { modelId: 'phi-4', fixtures: fx_phi_4 },
  { modelId: 'phi-4-reasoning', fixtures: fx_phi_4_reasoning },
    { modelId: 'deepseek-r1-distill-qwen-1.5b', fixtures: fx_deepseek_r1_qwen_1_5b },
  { modelId: 'qwen3-4b-instruct-2507', fixtures: fx_qwen3_4b_2507 },
  { modelId: 'gemma-3-4b-it', fixtures: fx_gemma_3_4b },
  { modelId: 'smollm2-135m-instruct', fixtures: fx_smollm2_135m },
  { modelId: 'gemma-3-270m-it', fixtures: fx_gemma_3_270m },
  { modelId: 'lfm2.5-350m', fixtures: fx_lfm2_5_350m },
  { modelId: 'granite-4.0-350m', fixtures: fx_granite_4_0_350m },
  { modelId: 'qwen2.5-coder-0.5b-instruct', fixtures: fx_qwen2_5_coder_0_5b },
  { modelId: 'lfm2-700m', fixtures: fx_lfm2_700m },
  { modelId: 'gemma-3-1b-it', fixtures: fx_gemma_3_1b },
  { modelId: 'granite-4.0-1b', fixtures: fx_granite_4_0_1b },
  { modelId: 'lfm2.5-1.2b-instruct', fixtures: fx_lfm2_5_1_2b },
  { modelId: 'qwen2.5-math-1.5b-instruct', fixtures: fx_qwen2_5_math_1_5b },
  { modelId: 'lfm2-2.6b', fixtures: fx_lfm2_2_6b },
  { modelId: 'qwen2.5-coder-3b-instruct', fixtures: fx_qwen2_5_coder_3b },
  { modelId: 'granite-4.0-micro', fixtures: fx_granite_4_0_micro },
  { modelId: 'phi-3-mini-4k-instruct', fixtures: fx_phi_3_mini },
  { modelId: 'olmo-3-7b-instruct', fixtures: fx_olmo_3_7b },
  { modelId: 'lfm2-8b-a1b', fixtures: fx_lfm2_8b_a1b },
  { modelId: 'lfm2-24b-a2b', fixtures: fx_lfm2_24b_a2b },
  { modelId: 'phi-4-mini-reasoning', fixtures: fx_phi_4_mini_reasoning },
  { modelId: 'mistral-7b-instruct-v0.2', fixtures: fx_mistral_7b_v0_2 },
  { modelId: 'llama-3.1-8b-instruct', fixtures: fx_llama_3_1_8b },
  { modelId: 'deepseek-r1-distill-llama-8b', fixtures: fx_deepseek_r1_llama_8b },
  { modelId: 'gemma-2-9b-it', fixtures: fx_gemma_2_9b },
  { modelId: 'phi-3-medium-4k-instruct', fixtures: fx_phi_3_medium },
  { modelId: 'phi-4', fixtures: fx_phi_4 },
  { modelId: 'phi-4-reasoning', fixtures: fx_phi_4_reasoning },
    { modelId: 'deepseek-r1-distill-qwen-1.5b', fixtures: fx_deepseek_r1_qwen_1_5b },
  { modelId: 'qwen3-4b-instruct-2507', fixtures: fx_qwen3_4b_2507 },
  { modelId: 'gemma-3-4b-it', fixtures: fx_gemma_3_4b },
  { modelId: 'smollm2-135m-instruct', fixtures: fx_smollm2_135m },
  { modelId: 'gemma-3-270m-it', fixtures: fx_gemma_3_270m },
  { modelId: 'lfm2.5-350m', fixtures: fx_lfm2_5_350m },
  { modelId: 'granite-4.0-350m', fixtures: fx_granite_4_0_350m },
  { modelId: 'qwen2.5-coder-0.5b-instruct', fixtures: fx_qwen2_5_coder_0_5b },
  { modelId: 'lfm2-700m', fixtures: fx_lfm2_700m },
  { modelId: 'gemma-3-1b-it', fixtures: fx_gemma_3_1b },
  { modelId: 'granite-4.0-1b', fixtures: fx_granite_4_0_1b },
  { modelId: 'lfm2.5-1.2b-instruct', fixtures: fx_lfm2_5_1_2b },
  { modelId: 'qwen2.5-math-1.5b-instruct', fixtures: fx_qwen2_5_math_1_5b },
  { modelId: 'lfm2-2.6b', fixtures: fx_lfm2_2_6b },
  { modelId: 'qwen2.5-coder-3b-instruct', fixtures: fx_qwen2_5_coder_3b },
  { modelId: 'granite-4.0-micro', fixtures: fx_granite_4_0_micro },
  { modelId: 'phi-3-mini-4k-instruct', fixtures: fx_phi_3_mini },
  { modelId: 'olmo-3-7b-instruct', fixtures: fx_olmo_3_7b },
  { modelId: 'lfm2-8b-a1b', fixtures: fx_lfm2_8b_a1b },
  { modelId: 'lfm2-24b-a2b', fixtures: fx_lfm2_24b_a2b },
  { modelId: 'phi-4-mini-reasoning', fixtures: fx_phi_4_mini_reasoning },
  { modelId: 'mistral-7b-instruct-v0.2', fixtures: fx_mistral_7b_v0_2 },
  { modelId: 'llama-3.1-8b-instruct', fixtures: fx_llama_3_1_8b },
  { modelId: 'deepseek-r1-distill-llama-8b', fixtures: fx_deepseek_r1_llama_8b },
  { modelId: 'gemma-2-9b-it', fixtures: fx_gemma_2_9b },
  { modelId: 'phi-3-medium-4k-instruct', fixtures: fx_phi_3_medium },
  { modelId: 'phi-4', fixtures: fx_phi_4 },
  { modelId: 'phi-4-reasoning', fixtures: fx_phi_4_reasoning },
  // New models (2026-10-11)
  { modelId: 'deepseek-r1-distill-qwen-1.5b', fixtures: fx_deepseek_r1_qwen_1_5b },
  { modelId: 'qwen3-4b-instruct-2507', fixtures: fx_qwen3_4b_2507 },
  { modelId: 'gemma-3-4b-it', fixtures: fx_gemma_3_4b },
  { modelId: 'smollm2-135m-instruct', fixtures: fx_smollm2_135m },
  { modelId: 'gemma-3-270m-it', fixtures: fx_gemma_3_270m },
  { modelId: 'lfm2.5-350m', fixtures: fx_lfm2_5_350m },
  { modelId: 'granite-4.0-350m', fixtures: fx_granite_4_0_350m },
  { modelId: 'qwen2.5-coder-0.5b-instruct', fixtures: fx_qwen2_5_coder_0_5b },
  { modelId: 'lfm2-700m', fixtures: fx_lfm2_700m },
  { modelId: 'gemma-3-1b-it', fixtures: fx_gemma_3_1b },
  { modelId: 'granite-4.0-1b', fixtures: fx_granite_4_0_1b },
  { modelId: 'lfm2.5-1.2b-instruct', fixtures: fx_lfm2_5_1_2b },
  { modelId: 'qwen2.5-math-1.5b-instruct', fixtures: fx_qwen2_5_math_1_5b },
  { modelId: 'lfm2-2.6b', fixtures: fx_lfm2_2_6b },
  { modelId: 'qwen2.5-coder-3b-instruct', fixtures: fx_qwen2_5_coder_3b },
  { modelId: 'granite-4.0-micro', fixtures: fx_granite_4_0_micro },
  { modelId: 'phi-3-mini-4k-instruct', fixtures: fx_phi_3_mini },
  { modelId: 'olmo-3-7b-instruct', fixtures: fx_olmo_3_7b },
  { modelId: 'lfm2-8b-a1b', fixtures: fx_lfm2_8b_a1b },
  { modelId: 'lfm2-24b-a2b', fixtures: fx_lfm2_24b_a2b },
  { modelId: 'phi-4-mini-reasoning', fixtures: fx_phi_4_mini_reasoning },
  { modelId: 'mistral-7b-instruct-v0.2', fixtures: fx_mistral_7b_v0_2 },
  { modelId: 'llama-3.1-8b-instruct', fixtures: fx_llama_3_1_8b },
  { modelId: 'deepseek-r1-distill-llama-8b', fixtures: fx_deepseek_r1_llama_8b },
  { modelId: 'gemma-2-9b-it', fixtures: fx_gemma_2_9b },
  { modelId: 'phi-3-medium-4k-instruct', fixtures: fx_phi_3_medium },
  { modelId: 'phi-4', fixtures: fx_phi_4 },
  { modelId: 'phi-4-reasoning', fixtures: fx_phi_4_reasoning },
];

for (const { modelId, fixtures } of SUITES as { modelId: string; fixtures: typeof smollm2 }[]) {
  const model = findModel(modelId)!;
  const path = new URL(
    `../../tools/reference/.cache/${fixtures.source.repo.replace('/', '__')}/${fixtures.source.revision}/${model.tokenizer.path}`,
    import.meta.url,
  );
  describe.skipIf(!existsSync(path))(`${model.name} tokenizer vs reference`, () => suite(model, fixtures, path));
}

interface TokenizerFile {
  normalizer: { type: string } | null;
  added_tokens: { content: string; special: boolean }[];
}

function suite(model: NonNullable<ReturnType<typeof findModel>>, fixtures: typeof smollm2, path: URL) {
  // Built per suite and dropped after it: all the tokenizers at once don't fit in Node's heap.
  let json!: TokenizerFile;
  let tokenizer!: BpeTokenizer;
  let plainTextCases: typeof fixtures.text = [];
  beforeAll(() => {
    json = JSON.parse(readFileSync(path, 'utf8')) as TokenizerFile;
    tokenizer = BpeTokenizer.fromJSON(json);
    // Hugging Face turns non-special added tokens (e.g. Qwen3's "<think>") inside
    // text into control tokens; we deliberately keep message text plain.
    const nonSpecialAdded = json.added_tokens.filter((t) => !t.special).map((t) => t.content);
    plainTextCases = fixtures.text.filter((c) => !hasAddedTokenText(c.text, nonSpecialAdded));
    addedTextCases = fixtures.text.filter((c) => hasAddedTokenText(c.text, nonSpecialAdded));
  });
  afterAll(() => {
    json = undefined as unknown as TokenizerFile;
    tokenizer = undefined as unknown as BpeTokenizer;
  });
  let addedTextCases: typeof fixtures.text = [];
  // What decoding gives back: the input after the model's normalizer.
  const normalized = (text: string) => (json.normalizer?.type === 'NFC' ? text.normalize('NFC') : text);

  it.each(fixtures.text.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('encodes %s', (_, c) => {
    if (!plainTextCases.includes(c)) return;
    expect(tokenizer.encode(c.text)).toEqual(c.ids);
  });

  it.each(fixtures.text.map((c) => [JSON.stringify(c.text).slice(0, 50), c] as const))('round-trips %s', (_, c) => {
    if (!plainTextCases.includes(c)) return;
    expect(tokenizer.decode(c.ids)).toEqual(normalized(c.text));
  });

  it('keeps added-token text (e.g. "<think>") in a message as plain text', () => {
    for (const c of addedTextCases) {
      const ids = tokenizer.encode(c.text);
      // Gemma's whitespace runs are added tokens that ordinary text produces too.
      const addedIds = new Set(json.added_tokens.filter((t) => !/^[\s▁]+$/u.test(t.content)).map((t) => tokenizer.specialTokenId(t.content)));
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

function hasAddedTokenText(text: string, added: string[]): boolean {
  return added.some((t) => text.includes(t));
}
