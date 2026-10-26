import { describe, expect, it } from 'vitest';
import type { BpeTokenizer } from './bpe-tokenizer';
import { Gemma4Stream, MuseGlimmerStream } from './harmony';

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

describe('Gemma4Stream', () => {
  const special = ['<|channel>', '<channel|>'];
  const words = [...special, 'thought', '\n', 'Let me think', '.', 'Paris', ' is the capital'];
  const gemma = {
    specialTokenId: (text: string) => special.indexOf(text),
    decode: (ids: number[], options?: { skipSpecialTokens?: boolean }) =>
      ids.map((id) => (options?.skipSpecialTokens && id < special.length ? '' : words[id])).join(''),
  } as unknown as BpeTokenizer;
  const runGemma = (...texts: string[]) => {
    const stream = new Gemma4Stream(gemma);
    return texts.map((t) => stream.push(words.indexOf(t))).join('') + stream.flush();
  };

  it('turns the thought channel into thinking, without its name line', () => {
    expect(runGemma('<|channel>', 'thought', '\n', 'Let me think', '.', '<channel|>', 'Paris', ' is the capital')).toBe(
      '<think>Let me think.</think>Paris is the capital',
    );
  });

  it('leaves a reply without a channel as it is, and closes thinking a reply ends in', () => {
    expect(runGemma('Paris', ' is the capital')).toBe('Paris is the capital');
    expect(runGemma('<|channel>', 'thought', '\n', 'Let me think')).toBe('<think>Let me think</think>');
  });
});
