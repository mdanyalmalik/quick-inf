import { describe, expect, it } from 'vitest';
import { IMAGE_MODELS } from '../catalogue/image-models';
import { catalogue } from '../catalogue/models';
import { chatQuality, imageQuality, INTELLIGENCE_FLOOR } from './quality';

describe('chatQuality', () => {
  const q = chatQuality(catalogue);
  it('scores every chat model, saying what each score rests on', () => {
    for (const m of catalogue) {
      const s = q.get(m.id)!;
      expect(Number.isFinite(s.score), m.id).toBe(true);
      const arena = m.quality.arena !== undefined;
      const bench = (m.quality.intelligence ?? 0) >= INTELLIGENCE_FLOOR;
      expect(s.basis, m.id).toBe(arena && bench ? 'arena+benchmarks' : arena ? 'arena' : bench ? 'benchmarks' : 'estimate');
    }
  });
  it('an arena rating alone is the score; with benchmarks too, the score lies between the two signals', () => {
    for (const m of catalogue) {
      const s = q.get(m.id)!;
      if (s.basis === 'arena') expect(s.score).toBe(m.quality.arena);
    }
    // gpt-oss 20B: strong on academic benchmarks, weaker with people (LMArena): the blend sits above its arena rating.
    const oss = q.get('gpt-oss-20b')!;
    expect(oss.basis).toBe('arena+benchmarks');
    expect(oss.score).toBeGreaterThan(1287);
    // …but below models people and benchmarks both rate higher.
    expect(oss.score).toBeLessThan(q.get('gemma-3-12b-it')!.score);
    expect(oss.score).toBeLessThan(q.get('qwen3.8-27b')!.score);
  });
  it('estimated scores keep the hand-set order', () => {
    const est = catalogue.filter((m) => q.get(m.id)!.basis === 'estimate');
    for (const a of est) for (const b of est) if (a.quality.rank > b.quality.rank) expect(q.get(a.id)!.score).toBeGreaterThan(q.get(b.id)!.score);
  });
});

describe('imageQuality', () => {
  const q = imageQuality(IMAGE_MODELS);
  it('uses arena ratings where they exist, and estimates the rest from the hand ranks', () => {
    for (const m of IMAGE_MODELS) {
      const s = q.get(m.id)!;
      if (m.arena !== undefined) expect(s).toEqual({ score: m.arena, basis: 'arena' });
      else expect(s.basis).toBe('estimate');
    }
    expect(q.get('flux.1-dev')!.score).toBeGreaterThan(q.get('sdxl-1.0')!.score);
  });
});
