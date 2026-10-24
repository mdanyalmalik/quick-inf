import { describe, expect, it } from 'vitest';
import type { Plan } from './planner';
import { sharedPrefillChunk } from './share-plan';

const GB = 1e9;
const plan = (shares: [string, number][]): Plan =>
  ({ shares: shares.map(([id, gb], i) => ({ id, from: i, to: i + 1, bytes: gb * GB, ms: 1 })), cuts: [], msPerToken: 1 }) as unknown as Plan;

describe('sharedPrefillChunk', () => {
  it('keeps the usual passes when no phone or tablet takes part', () => {
    expect(sharedPrefillChunk(plan([['host', 14], ['mac', 10]]), [{ id: 'host', label: 'Mac' }, { id: 'mac', label: 'Mac' }])).toBeUndefined();
  });

  it('keeps a phone’s passes short in proportion to its share', () => {
    const devices = [{ id: 'host', label: 'Mac' }, { id: 'phone', label: 'Android phone' }, { id: 'pc', label: 'Windows PC' }];
    // A 3.7 GB share (Nemotron Super 49B's): 64 / 3.7 = 17 → 16 tokens.
    expect(sharedPrefillChunk(plan([['host', 14], ['phone', 3.7], ['pc', 3.4]]), devices)).toBe(16);
    // A small share keeps longer passes.
    expect(sharedPrefillChunk(plan([['host', 2], ['phone', 0.5]]), devices)).toBe(128);
  });

  it('never goes below 8 tokens', () => {
    expect(sharedPrefillChunk(plan([['host', 2], ['ipad', 20]]), [{ id: 'host' }, { id: 'ipad', label: 'iPad' }])).toBe(8);
  });
});
