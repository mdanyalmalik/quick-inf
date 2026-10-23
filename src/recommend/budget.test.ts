import { describe, expect, it } from 'vitest';
import { findImageModel } from '../catalogue/image-models';
import { findModel } from '../catalogue/models';
import {
  deviceBudget,
  fitChat,
  fitImage,
  imagePeakBytes,
  MIN_CONTEXT_TOKENS,
  STREAM_CACHE_SHARE,
  STREAM_DEFAULT_CACHE_BYTES,
  STREAM_MIN_CACHE_EXPERTS,
  streamCacheBytes,
  streamedPeakBytes,
} from './budget';
import { memoryNeeded } from './estimate';

const GB = 1e9;

describe('deviceBudget', () => {
  it('has no limit before the benchmark has measured memory', () => {
    expect(deviceBudget({})).toBeNull();
    expect(deviceBudget({ provenBytes: 8 * GB, failedBytes: 4 * GB })).toBeNull();
  });

  it('is what the benchmark held, unless a model that ran here needed more', () => {
    expect(deviceBudget({ measuredBytes: 14 * GB })).toEqual({ bytes: 14 * GB, basis: 'measured' });
    expect(deviceBudget({ measuredBytes: 14 * GB, provenBytes: 6 * GB })).toEqual({ bytes: 14 * GB, basis: 'measured' });
    expect(deviceBudget({ measuredBytes: 14 * GB, provenBytes: 16 * GB })).toEqual({ bytes: 16 * GB, basis: 'proven' });
  });

  it("isn't lowered by failures", () => {
    expect(deviceBudget({ measuredBytes: 14 * GB, failedBytes: 2 * GB })!.bytes).toBe(14 * GB);
  });
});

describe('fitChat', () => {
  const model = findModel('qwen3-0.6b')!;
  const artifact = model.artifacts[0]!;

  it('keeps the default context when it fits', () => {
    expect(fitChat(model, artifact, 100 * GB)).toEqual({ status: 'fits', value: model.generation.contextTokens, bytes: memoryNeeded(model, artifact) });
  });

  it('shortens the context to fit, but not below the minimum', () => {
    const smaller = memoryNeeded(model, artifact, MIN_CONTEXT_TOKENS) + 1;
    const fit = fitChat(model, artifact, smaller);
    expect(fit.status).toBe('reduced');
    if (fit.status === 'reduced') expect(fit.value).toBeGreaterThanOrEqual(MIN_CONTEXT_TOKENS);
    expect(fitChat(model, artifact, 1).status).toBe('too-big');
  });
});

describe('fitImage', () => {
  it('drops to the largest size option that fits', () => {
    const sdxl = findImageModel('sdxl-1.0')!;
    const at = (s: number) => imagePeakBytes(sdxl, s, s);
    const fit = fitImage(sdxl, at(768) + 1, { width: 1024, height: 1024 }, [
      { width: 768, height: 768 },
      { width: 512, height: 512 },
    ]);
    expect(fit).toEqual({ status: 'reduced', value: { width: 768, height: 768 }, bytes: at(768) });
    expect(fitImage(sdxl, 1, { width: 1024, height: 1024 }, []).status).toBe('too-big');
  });

  it('expects the one-step SDXS (tiny decoder) to need less than a same-size full decoder', () => {
    const sdxs = findImageModel('sdxs-512-dreamshaper')!;
    const tiny = findImageModel('tiny-sd')!;
    expect(imagePeakBytes(sdxs, 512, 512)).toBeLessThan(imagePeakBytes(tiny, 512, 512));
  });
});

describe('streamCacheBytes and streamedPeakBytes', () => {
  const nemotron = findModel('nemotron-3-nano-30b-a3b')!;
  const artifact = nemotron.artifacts.find((a) => a.experts)!;
  const experts = artifact.experts!;
  const rest = memoryNeeded(nemotron, artifact) - experts.bytes;

  it('without a measured limit, gives the default cache', () => {
    expect(streamCacheBytes(nemotron, artifact, null)).toBe(Math.min(STREAM_DEFAULT_CACHE_BYTES, experts.bytes));
  });

  it('with a limit, gives a share of what is left after the other weights, between a minimum cache and all the experts', () => {
    expect(streamCacheBytes(nemotron, artifact, rest + 4e9)).toBeCloseTo(STREAM_CACHE_SHARE * 4e9, -3);
    expect(streamCacheBytes(nemotron, artifact, rest)).toBe(STREAM_MIN_CACHE_EXPERTS * experts.largestBytes);
    expect(streamCacheBytes(nemotron, artifact, 1e15)).toBe(experts.bytes);
  });

  it('is null for a model without experts', () => {
    const dense = findModel('smollm2-360m-instruct')!;
    expect(streamCacheBytes(dense, dense.artifacts[0]!, null)).toBeNull();
  });

  it('counts only the cache of the experts in a streamed run, so it never proves the whole model fits', () => {
    const peak = streamedPeakBytes(nemotron, artifact, 4e9);
    expect(peak).toBeCloseTo(rest + 4e9, -3);
    expect(peak).toBeLessThan(memoryNeeded(nemotron, artifact));
  });
});
