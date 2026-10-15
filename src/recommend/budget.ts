// How much memory this device can be trusted with, and what fits in it — without knowing the device in advance.
//
// Browsers don't tell a page how much memory it may use, and mobile browsers end a tab that goes over with no
// error to catch. So, like LM Studio's model-loading guardrails and llama.cpp's --fit:
//   - every device starts with a conservative budget for its class, from signals every browser gives;
//   - the budget is earned upward: a model that ran here proves the device can hold its peak;
//   - and pushed down by failure: running out of memory, losing the GPU, or the page being killed (see
//     runtime/breadcrumbs.ts) caps it below what failed;
//   - before a load, a model's estimated peak memory is checked against the budget: it fits, fits with a shorter
//     context or smaller image (as llama.cpp shrinks its context to fit), or is too big — and the person can
//     still choose "Try anyway", which records the outcome either way.

import { imageModelBytes, type ImageModelSpec } from '../catalogue/image-models';
import type { ModelArtifact, ModelSpec } from '../catalogue/models';
import { memoryNeeded } from './estimate';

export type DeviceClass = 'ios' | 'android' | 'shared-memory' | 'discrete-gpu' | 'no-gpu';

export interface DeviceSignals {
  /** A usable WebGPU adapter. */
  webgpu: boolean;
  /** iPhone/iPad (every iOS browser is WebKit, with Safari's per-tab memory limit). */
  ios: boolean;
  /** Android phone or tablet. */
  android: boolean;
  /** WebGPU adapter vendor ("apple", "intel", "nvidia", …), lower-case. */
  gpuVendor?: string;
  /** navigator.deviceMemory (GB; Chromium only, coarse, capped at 8). */
  deviceMemoryGb?: number;
}

export function deviceClass(s: DeviceSignals): DeviceClass {
  if (s.ios) return 'ios';
  if (s.android) return 'android';
  if (!s.webgpu) return 'no-gpu';
  // A separate graphics card has its own memory and reports running out cleanly; everything else shares RAM.
  return /nvidia|amd|ati/.test(s.gpuVendor ?? '') ? 'discrete-gpu' : 'shared-memory';
}

const GB = 1e9;

/**
 * Where a device of each class starts, in the units of the app's estimates (memoryNeeded, imagePeakBytes), which
 * run above the memory actually used. iOS: Safari's page process is killed at about 1.5 GB of real use once memory
 * is tight (up to about 3 GB on a fresh tab); the 1.7B chat models, estimated at about 2.9 GB, run on recent
 * iPhones, so 3 GB. Android: a quarter of the reported memory. Computers: shared-memory machines swap rather than crash, discrete GPUs fail cleanly, so
 * they start higher; `deviceMemory` (capped at 8 GB) only lowers it.
 */
export function startingBudget(cls: DeviceClass, deviceMemoryGb?: number): number {
  switch (cls) {
    case 'ios':
      return 3 * GB;
    case 'android':
      return Math.min(2, (deviceMemoryGb ?? 4) / 4) * GB;
    case 'no-gpu':
      return 1 * GB;
    case 'shared-memory':
      return deviceMemoryGb === undefined ? 4 * GB : deviceMemoryGb >= 8 ? 6 * GB : deviceMemoryGb >= 4 ? 2.5 * GB : 1 * GB;
    case 'discrete-gpu':
      return 8 * GB;
  }
}

/** What has happened on this device: the largest peak that ran, and the smallest that failed for lack of memory. */
export interface DeviceEvidence {
  provenBytes?: number;
  failedBytes?: number;
}

export type Guardrail = 'strict' | 'balanced' | 'relaxed' | 'off';
const GUARDRAIL_FACTOR: Record<Guardrail, number> = { strict: 0.75, balanced: 1, relaxed: 1.5, off: Infinity };

/** Room left below a size that failed (the estimate isn't exact). */
export const BELOW_FAILURE = 0.85;

export interface Budget {
  bytes: number;
  /** Why it is what it is, for the person. */
  basis: 'class' | 'proven' | 'failure';
  cls: DeviceClass;
}

export function deviceBudget(cls: DeviceClass, deviceMemoryGb: number | undefined, evidence: DeviceEvidence, guardrail: Guardrail = 'balanced'): Budget {
  let bytes = startingBudget(cls, deviceMemoryGb) * GUARDRAIL_FACTOR[guardrail];
  let basis: Budget['basis'] = 'class';
  if (evidence.provenBytes && evidence.provenBytes > bytes) {
    bytes = evidence.provenBytes;
    basis = 'proven';
  }
  if (evidence.failedBytes && bytes > evidence.failedBytes * BELOW_FAILURE) {
    bytes = evidence.failedBytes * BELOW_FAILURE;
    basis = 'failure';
  }
  return { bytes, basis, cls };
}

export type Fit<T> = { status: 'fits'; value: T; bytes: number } | { status: 'reduced'; value: T; bytes: number } | { status: 'too-big'; bytes: number };

/** The shortest conversation still worth having (llama.cpp's --fit-ctx minimum). */
export const MIN_CONTEXT_TOKENS = 2048;

/** The model's default context if it fits the budget, otherwise the longest that does (≥ MIN_CONTEXT_TOKENS). */
export function fitChat(model: ModelSpec, artifact: ModelArtifact, budget: number): Fit<number> {
  const wanted = model.generation.contextTokens;
  const full = memoryNeeded(model, artifact, wanted);
  if (full <= budget) return { status: 'fits', value: wanted, bytes: full };
  for (let context = Math.floor(wanted / 2 / 512) * 512; context >= MIN_CONTEXT_TOKENS; context = Math.floor(context / 2 / 512) * 512) {
    const bytes = memoryNeeded(model, artifact, context);
    if (bytes <= budget) return { status: 'reduced', value: context, bytes };
  }
  return { status: 'too-big', bytes: memoryNeeded(model, artifact, Math.min(wanted, MIN_CONTEXT_TOKENS)) };
}

/**
 * Peak memory to make one `width` × `height` image: every stage's weights (they stay loaded, and take about 1.5×
 * their file size once on the GPU) plus the working memory of the largest stage, which grows with the image and is
 * dominated by the decoder at full size. Measured browser peaks at 512 × 512 (renderer + GPU process, above an
 * empty page): SDXS (tiny decoder) 1.9 GB, Tiny SD 5.0 GB, aMUSEd 5.6 GB; these constants come out 5–15% above.
 */
export function imagePeakBytes(model: ImageModelSpec, width: number, height: number): number {
  const kind = model.pipeline?.kind ?? 'sd';
  const tinyDecoder = model.pipeline?.kind === 'sd' && model.pipeline.vaeScales === true;
  const perPixel = (tinyDecoder ? BYTES_PER_PIXEL.tiny : BYTES_PER_PIXEL[kind]) ?? DEFAULT_BYTES_PER_PIXEL;
  return imageModelBytes(model) * LOADED_WEIGHTS_FACTOR + width * height * perPixel + IMAGE_OVERHEAD;
}

const LOADED_WEIGHTS_FACTOR = 1.5;
const IMAGE_OVERHEAD = 256 * 1024 * 1024;
/** Full-size decoders (measured: Stable Diffusion's VAE and aMUSEd's VQ decoder, about 13 KB per pixel). */
const DEFAULT_BYTES_PER_PIXEL = 13_000;
const BYTES_PER_PIXEL: Record<string, number> = {
  sd: 13_000,
  amused: 13_000,
  /** The tiny decoder (TAESD) of SDXS and similar (measured about 1.9 KB per pixel). */
  tiny: 2_000,
};

/** The requested size if it fits, otherwise the largest of `sizes` that does. */
export function fitImage(
  model: ImageModelSpec,
  budget: number,
  wanted: { width: number; height: number },
  sizes: { width: number; height: number }[],
): Fit<{ width: number; height: number }> {
  const at = (s: { width: number; height: number }) => imagePeakBytes(model, s.width, s.height);
  if (at(wanted) <= budget) return { status: 'fits', value: wanted, bytes: at(wanted) };
  const smaller = sizes.filter((s) => s.width * s.height < wanted.width * wanted.height).sort((a, b) => b.width * b.height - a.width * a.height);
  for (const s of smaller) if (at(s) <= budget) return { status: 'reduced', value: s, bytes: at(s) };
  const smallest = [...sizes, wanted].sort((a, b) => a.width * a.height - b.width * b.height)[0]!;
  return { status: 'too-big', bytes: at(smallest) };
}
