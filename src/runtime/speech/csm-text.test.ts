// CSM-1B's prompt ("<|begin_of_text|>[0]{text}<|end_of_text|>", its chat template for speaker 0) against Hugging
// Face's apply_chat_template ids (printed by transformers from exports/CSM-1B-ONNX-WebGPU). Skipped if the staged
// tokenizer is missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';

const TOKENIZER = new URL('../../../exports/CSM-1B-ONNX-WebGPU/tokenizer.json', import.meta.url);
const CASES: [string, number[]][] = [
  ['Hello from Phonon. This is a quick test.', [128000, 58, 15, 60, 9906, 505, 71424, 263, 13, 1115, 374, 264, 4062, 1296, 13, 128001]],
  [
    'Please call me back at 555 0142 before 6 pm on Friday.',
    [128000, 58, 15, 60, 5618, 1650, 757, 1203, 520, 220, 14148, 220, 15901, 17, 1603, 220, 21, 9012, 389, 6740, 13, 128001],
  ],
];

describe.skipIf(!existsSync(TOKENIZER))('CSM prompt', () => {
  const tokenizer = BpeTokenizer.fromJSON(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  it.each(CASES)('%s', (text, ids) => {
    const bos = tokenizer.specialTokenId('<|begin_of_text|>');
    const eos = tokenizer.specialTokenId('<|end_of_text|>');
    expect([bos, ...tokenizer.encode(`[0]${text}`), eos]).toEqual(ids);
  });
});
