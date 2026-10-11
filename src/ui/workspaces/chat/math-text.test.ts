import { describe, expect, it } from 'vitest';
import { normalizeMath } from './math-text';

describe('normalizeMath', () => {
  it('converts LaTeX delimiters to dollars', () => {
    expect(normalizeMath('So \\(a^2\\) and \\[x = 1\\]')).toBe('So $$a^2$$ and $$x = 1$$');
  });
  it('wraps a bare \\boxed answer, but not one already in maths', () => {
    expect(normalizeMath('Answer: \\boxed{391}')).toBe('Answer: $$\\boxed{391}$$');
    expect(normalizeMath('$$\\boxed{\\frac{1}{2}}$$')).toBe('$$\\boxed{\\frac{1}{2}}$$');
    expect(normalizeMath('\\boxed{\\frac{1}{2}}')).toBe('$$\\boxed{\\frac{1}{2}}$$');
    expect(normalizeMath('It costs $5 and $10.')).toBe('It costs $5 and $10.');
  });
  it('leaves code alone', () => {
    expect(normalizeMath('`\\(x\\)` and\n```\n\\boxed{1}\n```')).toBe('`\\(x\\)` and\n```\n\\boxed{1}\n```');
  });
});
