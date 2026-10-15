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

/**
 * FLUX 3 Image Klein (BFL's text_encoder.py): the prompt as a user turn in Qwen3-VL's chat format, padded with
 * <|endoftext|> to a multiple of 80 tokens (at most 4,096); padded positions are masked out and take position 1.
 */
export function flux3PromptIds(t: BpeTokenizer, prompt: string, multiple = 80, max = 4096): { ids: number[]; mask: number[]; positions: number[] } {
  const start = t.specialTokenId('<|im_start|>');
  const end = t.specialTokenId('<|im_end|>');
  const ids = [start, ...t.encode(`user\n${prompt}`), end, ...t.encode('\n'), start, ...t.encode('assistant\n')].slice(0, max);
  const length = Math.min(Math.ceil(ids.length / multiple) * multiple, max);
  const padding = length - ids.length;
  const pad = t.specialTokenId('<|endoftext|>');
  return {
    ids: [...ids, ...new Array<number>(padding).fill(pad)],
    mask: [...ids.map(() => 1), ...new Array<number>(padding).fill(0)],
    positions: [...ids.map((_, i) => i), ...new Array<number>(padding).fill(1)],
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
