// How the language-model-based image models read a prompt (checked against Hugging Face in image-tokenizers.test.ts).

import type { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';

const Z_IMAGE_MAX_TOKENS = 512;

/** Z-Image: the prompt as a user turn in Qwen3's chat format (thinking on, so no empty think block), at most 512 tokens. */
export function zImagePromptIds(t: BpeTokenizer, prompt: string): number[] {
  const start = t.specialTokenId('<|im_start|>');
  const end = t.specialTokenId('<|im_end|>');
  return [start, ...t.encode(`user\n${prompt}`), end, ...t.encode('\n'), start, ...t.encode('assistant\n')].slice(0, Z_IMAGE_MAX_TOKENS);
}

/** FLUX.2 klein: the prompt as a user turn in Qwen3's chat format, thinking off, padded to 512 tokens (with its mask). */
export function flux2PromptIds(t: BpeTokenizer, prompt: string, length = 512): { ids: number[]; mask: number[] } {
  const start = t.specialTokenId('<|im_start|>');
  const end = t.specialTokenId('<|im_end|>');
  const ids = [
    start,
    ...t.encode(`user\n${prompt}`),
    end,
    ...t.encode('\n'),
    start,
    ...t.encode('assistant\n'),
    t.specialTokenId('<think>'),
    ...t.encode('\n\n'),
    t.specialTokenId('</think>'),
    ...t.encode('\n\n'),
  ].slice(0, length);
  const pad = t.specialTokenId('<|endoftext|>');
  return {
    ids: [...ids, ...new Array<number>(length - ids.length).fill(pad)],
    mask: [...ids.map(() => 1), ...new Array<number>(length - ids.length).fill(0)],
  };
}

/** Janus: "<｜begin▁of▁sentence｜><|User|>: {prompt}\n\n<|Assistant|>:<begin_of_image>" (its text_to_image template). */
export function janusPromptIds(t: BpeTokenizer, prompt: string): number[] {
  return [
    t.specialTokenId('<｜begin▁of▁sentence｜>'),
    t.specialTokenId('<|User|>'),
    ...t.encode(`: ${prompt.trim()}\n\n`),
    t.specialTokenId('<|Assistant|>'),
    ...t.encode(':'),
    t.specialTokenId('<begin_of_image>'),
  ];
}
