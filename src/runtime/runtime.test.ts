import { describe, expect, it } from 'vitest';
import { fitTurns } from './context';
import { RuntimeError, type ChatTurn } from './contract';
import { applyRepetitionPenalty, argmax, createRng, sampleToken } from './sampling';

describe('sampling', () => {
  const logits = [0.1, 3.0, 2.9, -5, 1.0];

  it('is greedy at temperature 0', () => {
    expect(sampleToken(logits, { temperature: 0, topP: 0.9 }, Math.random)).toBe(argmax(logits));
  });

  it('is reproducible for a fixed seed', () => {
    const run = () => {
      const rng = createRng(42);
      return Array.from({ length: 20 }, () => sampleToken(logits, { temperature: 1, topP: 1 }, rng));
    };
    expect(run()).toEqual(run());
  });

  it('restricts choices to the top-p nucleus', () => {
    // Token 1 alone holds ~48% of the probability mass, so a 0.4 nucleus is just {1}.
    const rng = createRng(7);
    const seen = new Set(Array.from({ length: 500 }, () => sampleToken(logits, { temperature: 1, topP: 0.4 }, rng)));
    expect([...seen]).toEqual([1]);
  });

  it('can sample beyond the top token with a wide nucleus', () => {
    const rng = createRng(3);
    const seen = new Set(Array.from({ length: 500 }, () => sampleToken(logits, { temperature: 1, topP: 1 }, rng)));
    expect(seen.has(1) && seen.has(2)).toBe(true);
  });
});

describe('applyRepetitionPenalty', () => {
  it('pushes positive and negative logits of recent tokens down, once per id', () => {
    const logits = new Float32Array([2, -2, 4, 1]);
    applyRepetitionPenalty(logits, [0, 1, 0, 0], 2);
    expect([...logits]).toEqual([1, -4, 4, 1]);
  });

  it('does nothing at penalty 1', () => {
    const logits = new Float32Array([2, -2]);
    applyRepetitionPenalty(logits, [0, 1], 1);
    expect([...logits]).toEqual([2, -2]);
  });
});

describe('fitTurns', () => {
  const len = (t: ChatTurn) => t.content.length;
  const turns: ChatTurn[] = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'aaaa' },
    { role: 'assistant', content: 'bbbb' },
    { role: 'user', content: 'cccc' },
  ];

  it('keeps everything that fits', () => {
    expect(fitTurns(turns, 100, len, 0)).toEqual({ turns, droppedTurns: 0 });
  });

  it('drops the oldest turns first and keeps the system turn', () => {
    const plan = fitTurns(turns, 3 + 4 + 4, len, 0);
    expect(plan.turns.map((t) => t.content)).toEqual(['sys', 'cccc']);
    expect(plan.droppedTurns).toBe(2);
  });

  it('does not start the kept history with an assistant turn', () => {
    const plan = fitTurns(turns, 3 + 4 + 4 + 1, len, 0);
    expect(plan.turns[1]?.role).toBe('user');
  });

  it('rejects a latest message that cannot fit', () => {
    expect(() => fitTurns(turns, 5, len, 0)).toThrow(RuntimeError);
  });
});
