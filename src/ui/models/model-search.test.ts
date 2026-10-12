import { describe, expect, it } from 'vitest';
import { matchesSearch } from './model-search';

const qwen = { name: 'Qwen3 4B', publisher: 'Alibaba Qwen', parameters: '4B' };
const phi = { name: 'Phi-4 Reasoning', publisher: 'Microsoft', parameters: '14B', reasoning: true };

describe('matchesSearch', () => {
  it('matches every word against name, publisher and size, ignoring case', () => {
    expect(matchesSearch(qwen, 'qwen 4b')).toBe(true);
    expect(matchesSearch(qwen, 'ALIBABA')).toBe(true);
    expect(matchesSearch(qwen, 'qwen 8b')).toBe(false);
  });
  it('finds reasoning models by kind', () => {
    expect(matchesSearch(phi, 'thinking')).toBe(true);
    expect(matchesSearch(qwen, 'thinking')).toBe(false);
  });
  it('an empty query matches everything', () => {
    expect(matchesSearch(qwen, '  ')).toBe(true);
  });
});
