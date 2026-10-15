import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { flux3PromptIds } from './prompts';

// FLUX 3 Image Klein's prompts as BFL's text_encoder.py prepares them (transformers: the processor's chat template,
// padded with <|endoftext|> to a multiple of 80, padded positions at 1), from tools/export/export_flux3.py's
// tokenizer; skipped without it.
const TOKENIZER = new URL('../../../tools/export/.cache/hf/flux-3-image-klein/text_encoder/tokenizer.json', import.meta.url);
const cases = JSON.parse(readFileSync(new URL('./__fixtures__/flux3-prompts.json', import.meta.url), 'utf8')) as {
  prompt: string;
  ids: number[];
  mask: number[];
  positions: number[];
}[];

describe.skipIf(!existsSync(TOKENIZER))('flux3PromptIds', () => {
  const tokenizer = existsSync(TOKENIZER) ? BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8'))) : null;
  it.each(cases.map((c) => [c.prompt.slice(0, 40), c] as const))('%s', (_, c) => {
    expect(flux3PromptIds(tokenizer!, c.prompt)).toEqual({ ids: c.ids, mask: c.mask, positions: c.positions });
  });
});
