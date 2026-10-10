// Reference fixtures from tools/reference/make_fixtures.py, by catalogue model, for the
// diagnostics pages (development only; not part of the app build).

import type smollm2Tok from '../tokenizer/__fixtures__/smollm2-tokenizer.json';
import type smollm2Ref from '../runtime/__fixtures__/smollm2-q4f16-reference.json';

export type TokenizerFixtures = typeof smollm2Tok;
export type ReferenceFixtures = typeof smollm2Ref;

const TOKENIZERS = import.meta.glob<TokenizerFixtures>('../tokenizer/__fixtures__/*-tokenizer.json', { eager: true, import: 'default' });
const REFERENCES = import.meta.glob<ReferenceFixtures>('../runtime/__fixtures__/*-reference.json', { eager: true, import: 'default' });

/** Fixture file prefixes, as in make_fixtures.py. */
const NAMES: Record<string, string> = {
  'smollm2-360m-instruct': 'smollm2',
  'qwen2.5-0.5b-instruct': 'qwen2.5-0.5b',
  'qwen3-0.6b': 'qwen3-0.6b',
  'smollm2-1.7b-instruct': 'smollm2-1.7b',
  'qwen2.5-1.5b-instruct': 'qwen2.5-1.5b',
  'qwen2.5-coder-1.5b-instruct': 'qwen2.5-coder-1.5b',
  'qwen3-1.7b': 'qwen3-1.7b',
  'qwen3-4b': 'qwen3-4b',
  'llama-3.2-1b-instruct': 'llama-3.2-1b',
  'llama-3.2-3b-instruct': 'llama-3.2-3b',
  'phi-4-mini-instruct': 'phi-4-mini',
  'phi-3.5-mini-instruct': 'phi-3.5-mini',
  'qwen3.5-0.8b': 'qwen3.5-0.8b',
  'qwen3.5-2b': 'qwen3.5-2b',
  'qwen3.5-4b': 'qwen3.5-4b',
};

export function tokenizerFixturesFor(modelId: string): TokenizerFixtures | undefined {
  const name = NAMES[modelId];
  return name ? TOKENIZERS[`../tokenizer/__fixtures__/${name}-tokenizer.json`] : undefined;
}

/** The reference made from the file of `precision` where there is one, else the q4f16 one. */
export function referenceFor(modelId: string, precision: string): ReferenceFixtures | undefined {
  const name = NAMES[modelId];
  if (!name) return undefined;
  return REFERENCES[`../runtime/__fixtures__/${name}-${precision}-reference.json`] ?? REFERENCES[`../runtime/__fixtures__/${name}-q4f16-reference.json`];
}
