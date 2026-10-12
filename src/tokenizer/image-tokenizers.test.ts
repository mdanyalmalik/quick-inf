// The image models' tokenizers and prompt formats against Hugging Face (fixtures made with transformers'
// CLIPTokenizer/CLIPTokenizerFast and the tokenizers library). Each needs that model's tokenizer files in the
// reference cache (tools/reference/.cache); skipped if they're missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { janusPromptIds, zImagePromptIds } from '../runtime/image/prompts';
import { BpeTokenizer } from './bpe-tokenizer';
import { ClipTokenizer } from './clip-tokenizer';
import fixtures from './__fixtures__/image-tokenizers.json';

const cache = (path: string) => new URL(`../../tools/reference/.cache/${path}`, import.meta.url);
const SD15 = cache('microsoft__stable-diffusion-v1.5-webnn/d7c229b23e94f49b4a74c279ba17ff1afb367c31/tokenizer/resolve/main/');
const SDXL = cache('webnn__sdxl-turbo/51606a881f73accd9e69e3638c5b160a76d0eec3/');
const JANUS = cache('onnx-community__Janus-Pro-1B-ONNX/04efdf2e36cb07a034b0d94f7322356b292f0418/tokenizer.json');
const ZIMAGE = cache('webnn__Z-Image-Turbo/02e1d8cc5d10d9ff6c7ed1f54b173bd981381bc2/tokenizer/tokenizer.json');
const json = (url: URL): unknown => JSON.parse(readFileSync(url, 'utf8'));

describe.skipIf(!existsSync(new URL('vocab.json', SD15)))('CLIP, SD 1.5 (pads with <|endoftext|>)', () => {
  const t = new ClipTokenizer(
    json(new URL('vocab.json', SD15)) as Record<string, number>,
    readFileSync(new URL('merges.txt', SD15), 'utf8'),
    '<|endoftext|>',
  );
  it.each(fixtures.sd15.map((c) => [c.text.slice(0, 40), c] as const))('encodes %s', (_, c) => expect(t.encode(c.text)).toEqual(c.ids));
});

describe.skipIf(!existsSync(new URL('tokenizer/tokenizer.json', SDXL)))('CLIP, SDXL’s two tokenizers (from tokenizer.json)', () => {
  const one = ClipTokenizer.fromJSON(json(new URL('tokenizer/tokenizer.json', SDXL)), '<|endoftext|>');
  const two = ClipTokenizer.fromJSON(json(new URL('tokenizer_2/tokenizer.json', SDXL)), '!');
  it.each(fixtures.sdxl1.map((c) => [c.text.slice(0, 40), c] as const))('first encodes %s', (_, c) =>
    expect(one.encode(c.text)).toEqual(c.ids),
  );
  it.each(fixtures.sdxl2.map((c) => [c.text.slice(0, 40), c] as const))('second encodes %s', (_, c) =>
    expect(two.encode(c.text)).toEqual(c.ids),
  );
});

describe.skipIf(!existsSync(JANUS))('Janus prompt', () => {
  const t = BpeTokenizer.fromJSON(json(JANUS));
  it.each(fixtures.janus.map((c) => [c.prompt.slice(0, 40), c] as const))('reads %s', (_, c) =>
    expect(janusPromptIds(t, c.prompt)).toEqual(c.ids),
  );
});

describe.skipIf(!existsSync(ZIMAGE))('Z-Image prompt (Qwen3 chat format)', () => {
  const t = BpeTokenizer.fromJSON(json(ZIMAGE));
  it.each(fixtures.zimage.map((c) => [c.prompt.slice(0, 40), c] as const))('reads %s', (_, c) =>
    expect(zImagePromptIds(t, c.prompt)).toEqual(c.ids),
  );
});
