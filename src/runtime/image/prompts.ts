// How the language-model-based image models read a prompt (checked against Hugging Face in image-tokenizers.test.ts).

import type { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';

const Z_IMAGE_MAX_TOKENS = 512;

/** Z-Image: the prompt as a user turn in Qwen3's chat format (thinking on, so no empty think block), at most 512 tokens. */
export function zImagePromptIds(t: BpeTokenizer, prompt: string): number[] {
  const start = t.specialTokenId('<|im_start|>');
  const end = t.specialTokenId('<|im_end|>');
  return [start, ...t.encode(`user\n${prompt}`), end, ...t.encode('\n'), start, ...t.encode('assistant\n')].slice(0, Z_IMAGE_MAX_TOKENS);
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
