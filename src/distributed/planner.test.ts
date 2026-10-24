import { describe, expect, it } from 'vitest';
import { COMFORT_FILL, MAX_FILL, planSplit, type PlanInput } from './planner';

const GB = 1e9;
const base = (devices: PlanInput['devices'], hopMs = 0): PlanInput => ({
  layerBytes: Array.from({ length: 10 }, () => 0.1 * GB),
  hostBytes: 0.5 * GB,
  kvBytes: Array.from({ length: 10 }, () => 0.01 * GB),
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

  it('gives the faster device more while every device stays comfortably within its memory', () => {
    const roomy = { memoryBytes: 4 * GB };
    const plan = planSplit({
      ...base([
        { id: 'host', ...roomy, bandwidthGBs: 50 },
        { id: 'a', ...roomy, bandwidthGBs: 200 },
      ]),
      everyDevice: true,
    })!;
    // The host keeps the one layer it must run; a, four times faster, takes the rest.
    expect(plan.shares.map((s) => [s.id, s.from, s.to])).toEqual([
      ['host', 0, 1],
      ['a', 1, 10],
    ]);
  });

  it(`never plans a device past ${MAX_FILL * 100}% of its memory`, () => {
    // a holds 8 layers (0.98 GB) in its 1 GB, but may use only 0.85 GB of it: 6 layers, the host the other 4.
    const plan = planSplit(base([{ id: 'host', memoryBytes: 1.5 * GB, bandwidthGBs: 100 }, { id: 'a', memoryBytes: 1 * GB, bandwidthGBs: 100 }]))!;
    for (const s of plan.shares) expect(s.bytes).toBeLessThanOrEqual((s.id === 'host' ? 1.5 : 1) * GB * MAX_FILL);
    expect(plan.shares.find((s) => s.id === 'a')!.to - plan.shares.find((s) => s.id === 'a')!.from).toBeLessThanOrEqual(6);
  });

  it('spreads a model that fills the devices instead of filling the faster one', () => {
    const plan = planSplit(
      base([
        { id: 'host', memoryBytes: 1.6 * GB, bandwidthGBs: 100 },
        { id: 'a', memoryBytes: 0.8 * GB, bandwidthGBs: 200 },
      ]),
    )!;
    // No split keeps both within COMFORT_FILL. The fastest gives a (twice as fast) 5 layers, 81% full; the host's 6
    // and a's 4 leave the fullest at 79%.
    expect(plan.shares.map((s) => [s.id, s.from, s.to])).toEqual([
      ['host', 0, 6],
      ['a', 6, 10],
    ]);
    expect(Math.max(...plan.shares.map((s) => s.bytes / (s.id === 'host' ? 1.6 : 0.8) / GB))).toBeGreaterThan(COMFORT_FILL);
  });

  it('uses every device when the model only fits across all of them', () => {
    // Each other device holds 3 layers (0.45 GB less the margin, within MAX_FILL), the host 3: only all four reach 10.
    const tight = { memoryBytes: 0.45 * GB / MAX_FILL, bandwidthGBs: 100 };
    const plan = planSplit(base([{ id: 'host', ...tight, memoryBytes: 0.95 * GB / MAX_FILL }, { id: 'a', ...tight }, { id: 'b', ...tight }, { id: 'c', ...tight }]))!;
    expect(plan.shares.map((s) => s.id)).toEqual(['host', 'a', 'b', 'c']);
    expect(plan.shares.at(-1)!.to).toBe(10);
  });

  it('says when the devices can not hold the model', () => {
    expect(planSplit(base([{ id: 'host', memoryBytes: 1 * GB, bandwidthGBs: 100 }, { id: 'a', memoryBytes: 0.3 * GB, bandwidthGBs: 100 }]))).toBeNull();
  });

  it('gives every device some layers when asked, even if fewer would do', () => {
    const roomy = { memoryBytes: 4 * GB, bandwidthGBs: 100 };
    const plan = planSplit({ ...base([{ id: 'host', ...roomy }, { id: 'a', ...roomy }, { id: 'b', ...roomy }], 2), everyDevice: true })!;
    expect(plan.shares.map((s) => s.id)).toEqual(['host', 'a', 'b']);
    expect(plan.shares.every((s) => s.to > s.from)).toBe(true);
  });

  it('gives devices alike even shares', () => {
    const alike = { memoryBytes: 4 * GB, bandwidthGBs: 100 };
    const plan = planSplit({ ...base([{ id: 'host', ...alike }, { id: 'a', ...alike }]), hostBytes: 0, everyDevice: true })!;
    expect(plan.shares.map((s) => s.to - s.from)).toEqual([5, 5]);
  });

  it('gives devices alike even shares when the host is fullest in every plan', () => {
    const alike = { memoryBytes: 4 * GB, bandwidthGBs: 100 };
    const plan = planSplit({ ...base([{ id: 'host', ...alike }, { id: 'a', ...alike }, { id: 'b', ...alike }]), hostBytes: 3 * GB, everyDevice: true })!;
    const [host, a, b] = plan.shares.map((s) => s.to - s.from);
    expect(host).toBe(1);
    expect(Math.abs(a! - b!)).toBeLessThanOrEqual(1);
  });

  it('counts a hop to each used device and back', () => {
    const plan = planSplit(base([{ id: 'host', memoryBytes: 1 * GB, bandwidthGBs: 100 }, { id: 'a', memoryBytes: 2 * GB, bandwidthGBs: 100 }], 3))!;
    expect(plan.shares).toHaveLength(2);
    expect(plan.msPerToken).toBeCloseTo(10 + 2 * 3);
  });
});
