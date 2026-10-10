import { describe, expect, it } from 'vitest';
import { extendsByOne, stripThinking } from './text';

describe('stripThinking', () => {
  it('removes the empty thinking block once text follows it', () => {
    expect(stripThinking('<think>\n\n</think>\n\nParis.')).toBe('Paris.');
  });

  it('holds back text that may still become the block', () => {
    for (const partial of ['', '<', '<think', '<think>\n\n</th', '<think>\n\n</think>\n\n']) expect(stripThinking(partial)).toBeNull();
  });

  it('passes other text through', () => {
    expect(stripThinking('Hello')).toBe('Hello');
    expect(stripThinking('<b>bold</b>')).toBe('<b>bold</b>');
  });

  it('at the end, gives whatever is left', () => {
    expect(stripThinking('<think>\n\n</think>\n\n', true)).toBe('');
    expect(stripThinking('<thi', true)).toBe('<thi');
  });
});

describe('extendsByOne', () => {
  const a = { role: 'user' as const, content: 'Hi' };
  const b = { role: 'assistant' as const, content: 'Hello!' };
  const c = { role: 'user' as const, content: 'More' };
  it('is true only for the previous messages plus exactly one', () => {
    expect(extendsByOne([a, b, c], [a, b])).toBe(true);
    expect(extendsByOne([a, b], [a, b])).toBe(false);
    expect(extendsByOne([a, b, c, c], [a, b])).toBe(false);
    expect(extendsByOne([a, { ...b, content: 'Hello' }, c], [a, b])).toBe(false);
  });
});
