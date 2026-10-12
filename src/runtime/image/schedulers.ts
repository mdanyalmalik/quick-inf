// Noise schedules for the image pipelines, as diffusers computes them (checked in schedulers.test.ts), and the
// small numeric helpers the pipelines share: seeded Gaussian noise, float16 conversion and pixels.

const TRAIN_STEPS = 1000;

/** ᾱ_t for t = 0…999: scaled-linear betas from 0.00085 to 0.012 (Stable Diffusion's training schedule). */
export function alphasCumprod(): Float64Array {
  const betaStart = Math.sqrt(0.00085);
  const betaEnd = Math.sqrt(0.012);
  const out = new Float64Array(TRAIN_STEPS);
  let product = 1;
  for (let t = 0; t < TRAIN_STEPS; t++) {
    const beta = (betaStart + ((betaEnd - betaStart) * t) / (TRAIN_STEPS - 1)) ** 2;
    product *= 1 - beta;
    out[t] = product;
  }
  return out;
}

/**
 * Euler's timesteps and noise levels (sigmas) for `steps` steps, plus a final sigma of 0. "trailing" counts back
 * from the last training step (SD Turbo, SDXL Turbo); "linspace" spreads steps evenly from 999 to 0, with sigmas
 * interpolated between training steps (Stable Diffusion 1.5).
 */
export function eulerSchedule(steps: number, spacing: 'trailing' | 'linspace' = 'trailing'): { timesteps: number[]; sigmas: number[] } {
  const acp = alphasCumprod();
  const sigmaAt = (t: number) => Math.sqrt((1 - acp[t]!) / acp[t]!);
  const timesteps: number[] = [];
  for (let i = 0; i < steps; i++) {
    if (spacing === 'trailing') timesteps.push(Math.round(TRAIN_STEPS - (i * TRAIN_STEPS) / steps) - 1);
    else timesteps.push(steps === 1 ? TRAIN_STEPS - 1 : ((TRAIN_STEPS - 1) * (steps - 1 - i)) / (steps - 1));
  }
  const sigmas = timesteps.map((t) => {
    const low = Math.floor(t);
    const high = Math.min(TRAIN_STEPS - 1, low + 1);
    return sigmaAt(low) + (sigmaAt(high) - sigmaAt(low)) * (t - low);
  });
  return { timesteps, sigmas: [...sigmas, 0] };
}

/**
 * LCM's timesteps for `steps` steps: evenly chosen from its 50 training-time steps (19, 39, …, 999), largest first,
 * as diffusers' LCMScheduler (4 steps: 999, 759, 499, 259).
 */
export function lcmTimesteps(steps: number, originalSteps = 50): number[] {
  const ratio = TRAIN_STEPS / originalSteps;
  const origin = Array.from({ length: originalSteps }, (_, i) => (i + 1) * ratio - 1).reverse();
  return Array.from({ length: steps }, (_, i) => origin[Math.floor((i * originalSteps) / steps)]!);
}

/** LCM's boundary-condition scalings at timestep `t` (sigma_data 0.5, timestep scaling 10). */
export function lcmScalings(t: number): { cSkip: number; cOut: number } {
  const scaled = t * 10;
  const sigmaData2 = 0.25;
  return { cSkip: sigmaData2 / (scaled * scaled + sigmaData2), cOut: scaled / Math.sqrt(scaled * scaled + sigmaData2) };
}

/** LCM's guidance input: a sinusoidal embedding of (guidance − 1) × 1000, `dim` wide (sines, then cosines). */
export function guidanceEmbedding(guidance: number, dim = 256): Float32Array {
  const w = (guidance - 1) * 1000;
  const half = dim / 2;
  const scale = Math.log(10000) / (half - 1);
  const out = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    const v = w * Math.exp(-scale * i);
    out[i] = Math.sin(v);
    out[half + i] = Math.cos(v);
  }
  return out;
}

/**
 * Flow matching's noise levels for `steps` steps (Z-Image: shift 3), plus a final 0, and the timestep input of
 * each step (1 − sigma, the last forced to 1), as Microsoft's WebNN Z-Image pipeline.
 */
export function flowMatchSchedule(steps: number, shift = 3): { sigmas: number[]; timesteps: number[] } {
  const shifted = (s: number) => (shift * s) / (1 + (shift - 1) * s);
  const sigmaMax = shifted(1);
  const sigmaMin = shifted(1 / TRAIN_STEPS);
  const sigmas: number[] = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? sigmaMax : sigmaMax + ((sigmaMin - sigmaMax) * i) / (steps - 1);
    sigmas.push(shifted(t));
  }
  const timesteps = sigmas.map((s, i) => (i === steps - 1 ? 1 : 1 - s));
  return { sigmas: [...sigmas, 0], timesteps };
}

/** `n` standard normal values from `seed` (xorshift + Box–Muller; not the same numbers as PyTorch). */
export function gaussian(n: number, seed: number): Float32Array {
  let s = seed >>> 0 || 1;
  const uniform = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) + 1) / 4294967297;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(uniform()));
    const theta = 2 * Math.PI * uniform();
    out[i] = r * Math.cos(theta);
    if (i + 1 < n) out[i + 1] = r * Math.sin(theta);
  }
  return out;
}

/** [1, 3, h, w] in [−1, 1] → RGBA bytes. */
export function toRgba(data: ArrayLike<number>, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const plane = width * height;
  const out = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    out[i * 4] = (data[i]! / 2 + 0.5) * 255;
    out[i * 4 + 1] = (data[plane + i]! / 2 + 0.5) * 255;
    out[i * 4 + 2] = (data[2 * plane + i]! / 2 + 0.5) * 255;
    out[i * 4 + 3] = 255;
  }
  return out;
}

const view = new DataView(new ArrayBuffer(4));

/** float32 → IEEE half bits (round to nearest even). */
export function toHalf(values: ArrayLike<number>): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    view.setFloat32(0, values[i]!);
    const x = view.getUint32(0);
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) {
      out[i] = sign | 0x7c00 | (mant ? 0x200 : 0);
      continue;
    }
    const e = exp - 127 + 15;
    if (e >= 0x1f) {
      out[i] = sign | 0x7c00;
      continue;
    }
    if (e <= 0) {
      if (e < -10) {
        out[i] = sign;
        continue;
      }
      mant |= 0x800000;
      const shift = 14 - e;
      let half = mant >>> shift;
      const rem = mant & ((1 << shift) - 1);
      const mid = 1 << (shift - 1);
      if (rem > mid || (rem === mid && half & 1)) half++;
      out[i] = sign | half;
      continue;
    }
    let half = (e << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half++;
    out[i] = sign | half;
  }
  return out;
}

/** IEEE half bits (or a Float16Array) → float32. */
export function fromHalf(values: ArrayLike<number>): Float32Array<ArrayBuffer> {
  if (!(values instanceof Uint16Array)) return Float32Array.from(values);
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const h = values[i]!;
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >>> 10) & 0x1f;
    const mant = h & 0x3ff;
    out[i] =
      exp === 0 ? sign * mant * 2 ** -24 : exp === 0x1f ? (mant ? NaN : sign * Infinity) : sign * (1 + mant / 1024) * 2 ** (exp - 15);
  }
  return out;
}
