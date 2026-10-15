import { describe, expect, it } from 'vitest';
import { findImageModel } from '../catalogue/image-models';
import { findModel } from '../catalogue/models';
import { deviceBudget, deviceClass, fitChat, fitImage, imagePeakBytes, MIN_CONTEXT_TOKENS, startingBudget } from './budget';
import { memoryNeeded } from './estimate';

const GB = 1e9;

describe('deviceClass', () => {
  it('sorts devices by what limits them', () => {
    expect(deviceClass({ webgpu: true, ios: true, android: false })).toBe('ios');
    expect(deviceClass({ webgpu: true, ios: false, android: true })).toBe('android');
    expect(deviceClass({ webgpu: false, ios: false, android: false })).toBe('no-gpu');
    expect(deviceClass({ webgpu: true, ios: false, android: false, gpuVendor: 'apple' })).toBe('shared-memory');
    expect(deviceClass({ webgpu: true, ios: false, android: false, gpuVendor: 'intel' })).toBe('shared-memory');
    expect(deviceClass({ webgpu: true, ios: false, android: false, gpuVendor: 'nvidia' })).toBe('discrete-gpu');
  });
});

describe('deviceBudget', () => {
  it('starts conservative for the class', () => {
    expect(startingBudget('ios')).toBe(3 * GB);
    expect(startingBudget('android', 8)).toBe(2 * GB);
    expect(startingBudget('android', 4)).toBe(1 * GB);
    expect(startingBudget('shared-memory', 8)).toBe(6 * GB);
    expect(deviceBudget('ios', undefined, {})).toEqual({ bytes: 3 * GB, basis: 'class', cls: 'ios' });
  });

  it('rises with what has run here and falls below what failed', () => {
    expect(deviceBudget('shared-memory', 8, { provenBytes: 14 * GB }).bytes).toBe(14 * GB);
    const failed = deviceBudget('shared-memory', 8, { provenBytes: 14 * GB, failedBytes: 10 * GB });
    expect(failed.basis).toBe('failure');
    expect(failed.bytes).toBeCloseTo(8.5 * GB);
    expect(deviceBudget('ios', undefined, { failedBytes: 1 * GB }).bytes).toBeCloseTo(0.85 * GB);
  });

  it('follows the guardrail setting', () => {
    expect(deviceBudget('ios', undefined, {}, 'strict').bytes).toBeCloseTo(2.25 * GB);
    expect(deviceBudget('ios', undefined, {}, 'off').bytes).toBe(Infinity);
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
