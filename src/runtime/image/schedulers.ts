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
 * from the last training step (SD Turbo, SDXL Turbo, SDXL Lightning); "leading" from the first, offset by one (SDXL 1.0);
 * "linspace" spreads steps evenly from 999 to 0, with sigmas interpolated between training steps (Stable Diffusion 1.5).
 */
export function eulerSchedule(
  steps: number,
  spacing: 'trailing' | 'linspace' | 'leading' = 'trailing',
): { timesteps: number[]; sigmas: number[] } {
  const acp = alphasCumprod();
  const sigmaAt = (t: number) => Math.sqrt((1 - acp[t]!) / acp[t]!);
  const timesteps: number[] = [];
  for (let i = 0; i < steps; i++) {
    if (spacing === 'trailing') timesteps.push(Math.round(TRAIN_STEPS - (i * TRAIN_STEPS) / steps) - 1);
    // Leading, with SDXL's steps_offset of 1.
    else if (spacing === 'leading') timesteps.push((steps - 1 - i) * Math.floor(TRAIN_STEPS / steps) + 1);
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

/**
 * FLUX.1's noise levels for `steps` steps, plus a final 0: evenly from 1 to 1/steps (as diffusers' FluxPipeline), and for
 * dev shifted towards noise by the image's token count (calculate_shift: mu from 0.5 at 256 tokens to 1.15 at 4096).
 */
export function flux1Schedule(steps: number, imageTokens: number, shift: boolean): number[] {
  const sigmas = Array.from({ length: steps }, (_, i) => (steps === 1 ? 1 : 1 + ((1 / steps - 1) * i) / (steps - 1)));
  if (!shift) return [...sigmas, 0];
  const mu = 0.5 + ((1.15 - 0.5) * (imageTokens - 256)) / (4096 - 256);
  return [...sigmas.map((s) => Math.exp(mu) / (Math.exp(mu) + (1 / s - 1))), 0];
}

/**
 * FLUX.2's timesteps for `steps` steps (from 1 to 0), shifted towards noise for larger images by its empirical
 * `mu`, as BFL's reference code (computeEmpiricalMu, generalized time-SNR shift).
 */
export function flux2Schedule(steps: number, imageTokens: number): number[] {
  const a1 = 8.73809524e-5;
  const b1 = 1.89833333;
  const a2 = 0.00016927;
  const b2 = 0.45666666;
  let mu: number;
  if (imageTokens > 4300) mu = a2 * imageTokens + b2;
  else {
    const m200 = a2 * imageTokens + b2;
    const m10 = a1 * imageTokens + b1;
    const a = (m200 - m10) / 190;
    mu = a * steps + (m200 - 200 * a);
  }
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = 1 - i / steps;
    return t <= 0 ? 0 : Math.exp(mu) / (Math.exp(mu) + (1 / t - 1));
  });
}

/**
 * FLUX 3 Image Klein's times for `steps` flow-map steps, 1 to 0, shifted towards noise: t = αs / (1 + (α − 1)s) over
 * evenly spaced s (BFL's sampling.py, α 6.93 by default).
 */
export function flux3Schedule(steps: number, alpha = 6.93): number[] {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const s = 1 - i / steps;
    return (alpha * s) / (1 + (alpha - 1) * s);
  });
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

/**
 * One step of aMUSEd's unmasking (diffusers' AmusedScheduler, cosine schedule, temperature from 2 down to 0): for each
 * still-masked position, sample a token from its logits; keep the tokens already known. Before the last step, mask
 * again the least confident of the new tokens (confidence = log probability + temperature × Gumbel noise), leaving
 * cos((step + 1) / steps · π/2) of the grid masked. Returns the next tokens and the full prediction (for previews).
 */
export function amusedStep(
  logits: ArrayLike<number>,
  tokens: Int32Array,
  step: number,
  steps: number,
  maskToken: number,
  random: () => number,
): { next: Int32Array; predicted: Int32Array } {
  const positions = tokens.length;
  const codebook = logits.length / positions;
  const predicted = tokens.slice();
  const confidence = new Float64Array(positions).fill(Infinity);
  const temperature = steps > 1 ? 2 - (2 * step) / (steps - 1) : 0;
  let unknown = 0;
  for (let p = 0; p < positions; p++) {
    if (tokens[p] !== maskToken) continue;
    unknown++;
    const row = p * codebook;
    let max = -Infinity;
    for (let k = 0; k < codebook; k++) max = Math.max(max, logits[row + k]!);
    let sum = 0;
    for (let k = 0; k < codebook; k++) sum += Math.exp(logits[row + k]! - max);
    let r = random() * sum;
    let token = codebook - 1;
    for (let k = 0; k < codebook; k++) {
      r -= Math.exp(logits[row + k]! - max);
      if (r <= 0) {
        token = k;
        break;
      }
    }
    predicted[p] = token;
    const gumbel = -Math.log(Math.max(-Math.log(Math.max(random(), 1e-20)), 1e-20));
    confidence[p] = logits[row + token]! - max - Math.log(sum) + temperature * gumbel;
  }
  if (step === steps - 1) return { next: predicted, predicted };
  const ratio = Math.cos((((step + 1) / steps) * Math.PI) / 2);
  const maskLength = Math.max(1, Math.min(unknown - 1, Math.floor(positions * ratio)));
  const cutOff = Float64Array.from(confidence).sort()[maskLength]!;
  const next = predicted.slice();
  for (let p = 0; p < positions; p++) if (confidence[p]! < cutOff) next[p] = maskToken;
  return { next, predicted };
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
