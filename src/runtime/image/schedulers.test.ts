import { describe, expect, it } from 'vitest';
import reference from './__fixtures__/diffusers-schedules.json';
import {
  amusedStep,
  eulerSchedule,
  flowMatchSchedule,
  flux1Schedule,
  fromHalf,
  gaussian,
  guidanceEmbedding,
  lcmScalings,
  lcmTimesteps,
  toHalf,
  toRgba,
} from './schedulers';

// Reference values from diffusers 0.36 (tools/reference/.venv): EulerDiscreteScheduler, LCMScheduler and
// FlowMatchEulerDiscreteScheduler with Stable Diffusion's betas, and LCM's guidance embedding.
const close = (actual: number[], expected: number[], digits = 3) => {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, digits));
};

describe('eulerSchedule', () => {
  it.each([
    ['trailing', 1],
    ['trailing', 4],
    ['linspace', 25],
    ['linspace', 4],
  ] as const)('matches diffusers (%s, %i steps)', (spacing, steps) => {
    const ref = reference[`euler-${spacing}-${steps}` as keyof typeof reference] as { timesteps: number[]; sigmas: number[] };
    const { timesteps, sigmas } = eulerSchedule(steps, spacing);
    close(timesteps, ref.timesteps);
    close(sigmas, ref.sigmas);
  });
});

describe('eulerSchedule, leading (SDXL 1.0: steps_offset 1)', () => {
  it('matches diffusers for 30 steps', () => {
    // EulerDiscreteScheduler(timestep_spacing="leading", steps_offset=1).set_timesteps(30), diffusers 0.36.
    const { timesteps, sigmas } = eulerSchedule(30, 'leading');
    close(timesteps.slice(0, 4), [958, 925, 892, 859]);
    close(sigmas.slice(0, 3), [11.4769, 9.5436, 8.0043]);
    expect(sigmas.at(-2)).toBeCloseTo(0.0413, 3);
  });
});

describe('LCM', () => {
  it('chooses diffusers’ timesteps', () => {
    expect(lcmTimesteps(4)).toEqual(reference['lcm-4']);
    expect(lcmTimesteps(2)).toEqual(reference['lcm-2']);
  });
  it('scales like diffusers', () => {
    const { cSkip, cOut } = lcmScalings(759);
    close([cSkip, cOut], reference['lcm-scalings-759'], 6);
  });
  it('embeds guidance like diffusers', () => {
    // torch computes the sines in float32 (arguments up to 7,000), so agreement is to about 1e-4.
    close([...guidanceEmbedding(8)], reference['lcm-guidance-8'], 3);
  });
});

describe('flowMatchSchedule (Z-Image)', () => {
  it('matches diffusers’ sigmas, and feeds 1 − sigma (the last as 1)', () => {
    const { sigmas, timesteps } = flowMatchSchedule(9);
    close(sigmas, reference['flow-9'].sigmas);
    close(
      timesteps.slice(0, -1),
      reference['flow-9'].timesteps.slice(0, -1).map((t) => 1 - t / 1000),
    );
    expect(timesteps.at(-1)).toBe(1);
  });
});

describe('flux1Schedule (FLUX.1)', () => {
  // diffusers 0.41 FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(1, 1/n, n), mu=calculate_shift(...)).
  it('schnell: evenly from 1 to 1/n, unshifted', () => {
    expect(flux1Schedule(4, 4096, false)).toEqual([1, 0.75, 0.5, 0.25, 0]);
  });
  it('dev: shifted for the image size, as diffusers', () => {
    const dev = flux1Schedule(28, 4096, true);
    [1.0, 0.988409, 0.976222, 0.963394, 0.949873].forEach((v, i) => expect(dev[i]).toBeCloseTo(v, 5));
    expect(dev[27]).toBeCloseTo(0.104721, 5);
    expect(dev[28]).toBe(0);
    const small = flux1Schedule(4, 1024, true);
    [1.0, 0.849235, 0.652489, 0.384945, 0].forEach((v, i) => expect(small[i]).toBeCloseTo(v, 5));
  });
});

describe('gaussian', () => {
  it('is repeatable per seed and roughly standard normal', () => {
    const a = gaussian(20000, 42);
    expect(Array.from(gaussian(5, 42))).toEqual(Array.from(a.slice(0, 5)));
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const variance = a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length;
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(variance).toBeGreaterThan(0.95);
    expect(variance).toBeLessThan(1.05);
  });
});

describe('half floats', () => {
  it('round-trip exactly representable values and round others to nearest', () => {
    const values = [0, 1, -2, 0.5, 65504, -(2 ** -14), 1 / 3, 14.6146, 1e-8, 70000];
    const back = [...fromHalf(toHalf(values))];
    expect(back.slice(0, 6)).toEqual([0, 1, -2, 0.5, 65504, -(2 ** -14)]);
    expect(back[6]).toBeCloseTo(1 / 3, 3);
    expect(back[7]).toBeCloseTo(14.6146, 2);
    expect(back[8]).toBe(0);
    expect(back[9]).toBe(Infinity);
  });
  it('matches Float16Array where the runtime has it', () => {
    const F16 = (globalThis as { Float16Array?: new (v: number[]) => ArrayLike<number> & { buffer: ArrayBuffer } }).Float16Array;
    if (!F16) return;
    const values = Array.from({ length: 1000 }, (_, i) => (i - 500) * 0.0137 + Math.sin(i) * 3);
    expect([...toHalf(values)]).toEqual([...new Uint16Array(new F16(values).buffer)]);
  });
});

describe('toRgba', () => {
  it('maps [-1, 1] planes to opaque bytes', () => {
    const data = Float32Array.from([-1, 1, 0, 0, 1, -1]); // 1 × 2 pixels, R G B planes
    expect(Array.from(toRgba(data, 2, 1))).toEqual([0, 128, 255, 255, 255, 128, 0, 255]);
  });
});

describe('amusedStep (aMUSEd)', () => {
  const random = (seed: number) => () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) + 0.5) / 2 ** 31;
  const MASK = 9;

  it('fills every masked position and, before the last step, masks the cosine share again', () => {
    const positions = 64;
    const codebook = 8;
    const logits = Float32Array.from({ length: positions * codebook }, (_, i) => Math.sin(i));
    const tokens = new Int32Array(positions).fill(MASK);
    const { next, predicted } = amusedStep(logits, tokens, 0, 4, MASK, random(1));
    expect(predicted.every((t) => t >= 0 && t < codebook)).toBe(true);
    // cos(1/4 · π/2) · 64 = 59.1 → 59 stay masked.
    expect(next.filter((t) => t === MASK).length).toBe(59);
    expect(next.every((t, p) => t === MASK || t === predicted[p])).toBe(true);
  });

  it('keeps known tokens and unmasks everything on the last step', () => {
    const logits = new Float32Array(4 * 3);
    const tokens = Int32Array.from([2, MASK, 1, MASK]);
    const { next } = amusedStep(logits, tokens, 3, 4, MASK, random(2));
    expect(next[0]).toBe(2);
    expect(next[2]).toBe(1);
    expect([...next].includes(MASK)).toBe(false);
  });

  it('samples from the softmax: a dominant logit wins', () => {
    const logits = Float32Array.from([0, 30, 0]);
    const { predicted } = amusedStep(logits, Int32Array.of(MASK), 1, 2, MASK, random(3));
    expect(predicted[0]).toBe(1);
  });
});
