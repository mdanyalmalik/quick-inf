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
  it('reads $ … $ as maths when it looks like maths, and prices as text', () => {
    expect(normalizeMath('using the $\\epsilon$-$N$ definition')).toBe('using the $$\\epsilon$$-$$N$$ definition');
    expect(normalizeMath('so $f(x) = \\frac{1}{x}$ and $x$, $M_1$, $f(x)$.')).toBe('so $$f(x) = \\frac{1}{x}$$ and $$x$$, $$M_1$$, $$f(x)$$.');
    expect(normalizeMath('$\\left| \\frac{1}{x} - 0 \\right| < \\epsilon$')).toBe('$$\\left| \\frac{1}{x} - 0 \\right| < \\epsilon$$');
    expect(normalizeMath('It costs $5 and $10.')).toBe('It costs $5 and $10.');
    expect(normalizeMath('Between $5 and $6, or $20 for both.')).toBe('Between $5 and $6, or $20 for both.');
    expect(normalizeMath('A $5 fee, then 3$ more')).toBe('A $5 fee, then 3$ more');
    expect(normalizeMath('Prices: $5$ each')).toBe('Prices: $5$ each');
    expect(normalizeMath('\\$x$ is escaped')).toBe('\\$x$ is escaped');
    expect(normalizeMath('$$x^2$$ stays')).toBe('$$x^2$$ stays');
  });
  it('leaves code alone', () => {
    expect(normalizeMath('`\\(x\\)` and\n```\n\\boxed{1}\n```')).toBe('`\\(x\\)` and\n```\n\\boxed{1}\n```');
  });
});
