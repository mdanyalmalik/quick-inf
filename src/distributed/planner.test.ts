import { describe, expect, it } from 'vitest';
import { planSplit, type PlanInput } from './planner';

const GB = 1e9;
const base = (devices: PlanInput['devices'], hopMs = 0): PlanInput => ({
  layerBytes: Array.from({ length: 10 }, () => 0.1 * GB),
  hostBytes: 0.5 * GB,
  kvBytesPerLayer: 0.01 * GB,
  overheadBytes: 0.1 * GB,
  devices,
  hopMs,
});

describe('planSplit', () => {
  it('keeps everything on the host when it fits and hops cost time', () => {
    const plan = planSplit(base([{ id: 'host', memoryBytes: 4 * GB, bandwidthGBs: 100 }, { id: 'a', memoryBytes: 4 * GB, bandwidthGBs: 100 }], 2))!;
    expect(plan.shares.map((s) => [s.id, s.from, s.to])).toEqual([['host', 0, 10]]);
    expect(plan.cuts).toEqual([]);
  });

  it('gives each device what fits, the faster one more', () => {
    const plan = planSplit(
      base([
        { id: 'host', memoryBytes: 1.2 * GB, bandwidthGBs: 100 },
        { id: 'a', memoryBytes: 1 * GB, bandwidthGBs: 200 },
        { id: 'b', memoryBytes: 1 * GB, bandwidthGBs: 50 },
      ]),
    )!;
    // a is fastest and takes all it holds (0.9 GB / 0.11 GB per layer: 8 layers); the host, next fastest, the rest.
    expect(plan.shares.map((s) => [s.id, s.from, s.to])).toEqual([
      ['host', 0, 2],
      ['a', 2, 10],
    ]);
    expect(plan.cuts).toEqual([2]);
    for (const s of plan.shares) expect(s.bytes).toBeLessThanOrEqual(s.id === 'host' ? 1.2 * GB : 1 * GB);
  });

  it('uses every device when the model only fits across all of them', () => {
    // Each other device holds 3 layers (0.45 GB less the margin), the host 3: only all four reach 10.
    const tight = { memoryBytes: 0.45 * GB, bandwidthGBs: 100 };
    const plan = planSplit(base([{ id: 'host', ...tight, memoryBytes: 0.95 * GB }, { id: 'a', ...tight }, { id: 'b', ...tight }, { id: 'c', ...tight }]))!;
    expect(plan.shares.map((s) => s.id)).toEqual(['host', 'a', 'b', 'c']);
    expect(plan.shares.at(-1)!.to).toBe(10);
  });

  it('says when the devices can not hold the model', () => {
    expect(planSplit(base([{ id: 'host', memoryBytes: 1 * GB, bandwidthGBs: 100 }, { id: 'a', memoryBytes: 0.3 * GB, bandwidthGBs: 100 }]))).toBeNull();
  });

  it('counts a hop to each used device and back', () => {
    const plan = planSplit(base([{ id: 'host', memoryBytes: 1 * GB, bandwidthGBs: 100 }, { id: 'a', memoryBytes: 2 * GB, bandwidthGBs: 100 }], 3))!;
    expect(plan.shares).toHaveLength(2);
    expect(plan.msPerToken).toBeCloseTo(10 + 2 * 3);
  });
});
