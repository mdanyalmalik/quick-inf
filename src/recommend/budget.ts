// The memory limit used to say a model may not fit, and to shorten a chat or shrink an image to fit:
//   - none before the benchmark's memory check has run (nothing is advised from a guess);
//   - then the larger of what that check held here (it holds GPU memory until the device runs short) and the largest
//     estimated peak of any model that has run here (memory-evidence.ts), since a model that ran shows the device
//     can hold at least that.
// A model's estimated peak memory is checked against it: it fits, fits with a shorter context or smaller image (as
// llama.cpp shrinks its context to fit), or may not fit (it can still be loaded).

import { imageModelBytes, type ImageModelSpec } from '../catalogue/image-models';
import type { ModelArtifact, ModelSpec } from '../catalogue/models';
import { memoryNeeded } from './estimate';

const GB = 1e9;

/** What has happened on this device. */
export interface DeviceEvidence {
  /** What the benchmark's memory check held (the latest run). */
  measuredBytes?: number;
  /** The largest estimated peak that ran. */
  provenBytes?: number;
  /** The smallest that failed for lack of memory (kept for the record; the limit doesn't use it). */
  failedBytes?: number;
}

export interface Budget {
  bytes: number;
  /** 'measured': what the benchmark held; 'proven': the largest model run here, which needed more. */
  basis: 'measured' | 'proven';
}

/** The memory limit, or null before the benchmark's memory check has run. */
export function deviceBudget(evidence: DeviceEvidence): Budget | null {
  if (evidence.measuredBytes === undefined) return null;
  const proven = evidence.provenBytes ?? 0;
  return proven > evidence.measuredBytes ? { bytes: proven, basis: 'proven' } : { bytes: evidence.measuredBytes, basis: 'measured' };
}

/** "the benchmark held about 14 GB" or "the most a model has used here is about 15 GB". */
export function limitText(budget: Budget, format: (bytes: number) => string): string {
  return budget.basis === 'measured' ? `the benchmark held about ${format(budget.bytes)}` : `the most a model has used here is about ${format(budget.bytes)}`;
}

/** Experts the cache holds at least when streaming (several layers' worth of chosen experts). */
export const STREAM_MIN_CACHE_EXPERTS = 64;

/** The expert cache without a measured memory limit: what Nemotron 3 Nano was tested with. */
export const STREAM_DEFAULT_CACHE_BYTES = 4 * GB;

/**
 * The share of what the measured limit leaves that the expert cache takes: the limit was measured once, with little
 * else running, so the rest is headroom for other tabs and apps (the whole of it swapped a test run).
 */
export const STREAM_CACHE_SHARE = 0.75;

/**
 * GPU memory for recently used experts when a mixture-of-experts model runs from storage (runtime/onnx/moe-stream.ts),
 * which the person chooses when loading it: STREAM_CACHE_SHARE of what `limit` leaves after the model's other weights
 * and its conversation (STREAM_DEFAULT_CACHE_BYTES without a limit), at least STREAM_MIN_CACHE_EXPERTS experts and at
 * most all of them.
 * Null for a model without experts.
 */
export function streamCacheBytes(model: ModelSpec, artifact: ModelArtifact, limit: number | null, contextTokens?: number): number | null {
  const experts = artifact.experts;
  if (!experts) return null;
  const rest = memoryNeeded(model, artifact, contextTokens) - experts.bytes;
  const wanted = limit === null ? STREAM_DEFAULT_CACHE_BYTES : STREAM_CACHE_SHARE * (limit - rest);
  return Math.min(Math.max(wanted, STREAM_MIN_CACHE_EXPERTS * experts.largestBytes), experts.bytes);
}

/** Peak memory of a run with streamed experts: the other weights, the conversation and the expert cache. */
export function streamedPeakBytes(model: ModelSpec, artifact: ModelArtifact, cacheBytes: number, contextTokens?: number): number {
  const experts = artifact.experts;
  const whole = memoryNeeded(model, artifact, contextTokens);
  return experts ? whole - experts.bytes + Math.min(cacheBytes, experts.bytes) : whole;
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
 * Peak memory to make one `width` × `height` image: the download (every stage's weights stay loaded) plus a little
 * extra, which grows with the image. Kept simple on purpose: per-pixel costs extrapolated from 512² runs put SDXL at
 * 1024² near 24 GB and FLUX.1 at 44 GB, yet both ran on a computer with less memory than that. Measured browser peaks at 512² (renderer +
 * GPU process, above an empty page): SDXS 1.9 GB, Tiny SD 5.0, aMUSEd 5.6, SD Turbo 5.2, Z-Image 7.0; this gives
 * 3.9, 4.1, 4.8, 5.5, 9.0.
 */
export function imagePeakBytes(model: ImageModelSpec, width: number, height: number): number {
  return imageModelBytes(model) + IMAGE_EXTRA_BYTES + (width * height * IMAGE_EXTRA_PER_512_SQUARED) / (512 * 512);
}

const IMAGE_EXTRA_BYTES = 1.5 * GB;
/** Working memory per 512 × 512 pixels of image. */
const IMAGE_EXTRA_PER_512_SQUARED = 1.5 * GB;

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
