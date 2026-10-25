import { describe, expect, it } from 'vitest';
import type { BpeTokenizer } from './bpe-tokenizer';
import { MuseGlimmerStream } from './harmony';

// A tokenizer with one token per piece of text and Muse Glimmer's control tokens.
const SPECIAL = ['<|start|>', '<|message|>', '<|eom|>', '<|eot|>'];
const pieces = [...SPECIAL, ' to', '=self', 'assistant', ' to=user', 'Let me think', '.', 'Paris', ' is the capital', ' to=web'];
const tokenizer = {
  specialTokenId: (text: string) => SPECIAL.indexOf(text),
  decode: (ids: number[], options?: { skipSpecialTokens?: boolean }) =>
    ids.map((id) => (options?.skipSpecialTokens && id < SPECIAL.length ? '' : pieces[id])).join(''),
} as unknown as BpeTokenizer;
const ids = (...texts: string[]) => texts.map((t) => pieces.indexOf(t));
const run = (tokens: number[]) => {
  const stream = new MuseGlimmerStream(tokenizer);
  return tokens.map((id) => stream.push(id)).join('') + stream.flush();
};

describe('MuseGlimmerStream', () => {
  it('turns the reasoning (to=self) into thinking and the message to the user into the answer', () => {
    // The generation prompt ends at <|start|>assistant: the reply starts with the rest of the header.
    const reply = ids(' to', '=self', '<|message|>', 'Let me think', '.', '<|eom|>', '<|start|>', 'assistant', ' to=user', '<|message|>', 'Paris', ' is the capital', '<|eot|>');
    expect(run(reply)).toBe('<think>Let me think.</think>Paris is the capital');
  });

  it('answers straight away when the first message is to the user', () => {
    expect(run(ids(' to=user', '<|message|>', 'Paris', '<|eot|>'))).toBe('Paris');
  });

  it('counts messages to anyone else as thinking, and closes thinking a reply ends in', () => {
    expect(run(ids(' to=web', '<|message|>', 'Let me think'))).toBe('<think>Let me think</think>');
  });
});
