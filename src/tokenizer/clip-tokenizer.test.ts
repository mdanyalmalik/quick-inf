// Compares the CLIP tokenizer with Hugging Face's CLIPTokenizer (fixtures from tools/reference). Needs SD Turbo's
// tokenizer files in the reference cache; skipped if they're missing.

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ClipTokenizer } from './clip-tokenizer';
import fixtures from './__fixtures__/clip-sd-turbo-tokenizer.json';

const dir = new URL(
  '../../tools/reference/.cache/schmuell__sd-turbo-ort-web/ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1/tokenizer/',
  import.meta.url,
);

describe.skipIf(!existsSync(new URL('vocab.json', dir)))('CLIP tokenizer (SD Turbo) vs reference', () => {
  const tokenizer = new ClipTokenizer(
    JSON.parse(readFileSync(new URL('vocab.json', dir), 'utf8')) as Record<string, number>,
    readFileSync(new URL('merges.txt', dir), 'utf8'),
  );
  it.each(fixtures.cases.map((c) => [c.text.slice(0, 40), c] as const))('encodes %s', (_, c) => {
    expect(tokenizer.encode(c.text)).toEqual(c.ids);
  });
});
