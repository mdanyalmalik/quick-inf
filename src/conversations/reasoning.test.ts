import { describe, expect, it } from 'vitest';
import { historyText, splitReasoning } from './reasoning';

describe('splitReasoning', () => {
  it('splits thinking from the answer', () => {
    expect(splitReasoning('<think>\nLet me add.\n</think>\n\nThe answer is 4.')).toEqual({
      thinking: 'Let me add.',
      thinkingOpen: false,
      answer: 'The answer is 4.',
    });
  });

  it('shows thinking in progress while streaming', () => {
    expect(splitReasoning('<think>\nFirst, ')).toEqual({ thinking: 'First,', thinkingOpen: true, answer: '' });
  });

  it('handles a missing opening tag for reasoning models', () => {
    expect(splitReasoning('Hmm, 2 + 2.\n</think>\n4', true)).toEqual({ thinking: 'Hmm, 2 + 2.', thinkingOpen: false, answer: '4' });
    expect(splitReasoning('Hmm, 2 + 2.', true)).toEqual({ thinking: 'Hmm, 2 + 2.', thinkingOpen: true, answer: '' });
  });

  it('leaves ordinary replies alone, even ones that mention the tags', () => {
    expect(splitReasoning('Use <think> tags like this: </think>.')).toEqual({ thinking: null, thinkingOpen: false, answer: 'Use <think> tags like this: </think>.' });
    expect(splitReasoning('Hello!')).toEqual({ thinking: null, thinkingOpen: false, answer: 'Hello!' });
  });

  it('keeps only the answer for the history', () => {
    expect(historyText('<think>long reasoning</think>Paris.')).toBe('Paris.');
  });
});
