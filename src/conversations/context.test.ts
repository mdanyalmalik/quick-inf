import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../runtime/contract';
import { boundedHistory } from './context';

const turn = (role: ChatTurn['role'], length: number, tag = ''): ChatTurn => ({ role, content: tag + 'x'.repeat(length - tag.length) });

describe('boundedHistory', () => {
  it('keeps everything when it fits', () => {
    const turns = [turn('user', 10), turn('assistant', 10), turn('user', 10)];
    expect(boundedHistory(turns, 30)).toEqual(turns);
  });

  it('keeps the newest turns within the budget', () => {
    const turns = [turn('user', 10, 'a'), turn('assistant', 10, 'b'), turn('user', 10, 'c'), turn('assistant', 10, 'd')];
    expect(boundedHistory(turns, 25).map((t) => t.content[0])).toEqual(['c', 'd']);
  });

  it('always keeps the last turn, even when it alone is over budget', () => {
    const turns = [turn('user', 10), turn('user', 100, 'z')];
    expect(boundedHistory(turns, 50).map((t) => t.content[0])).toEqual(['z']);
  });

  it('stops at the first turn that does not fit (no gaps)', () => {
    const turns = [turn('user', 5, 'a'), turn('assistant', 50, 'b'), turn('user', 5, 'c')];
    expect(boundedHistory(turns, 20).map((t) => t.content[0])).toEqual(['c']);
  });

  it('handles an empty history', () => {
    expect(boundedHistory([], 100)).toEqual([]);
  });
});
