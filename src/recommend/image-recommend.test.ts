import { describe, expect, it } from 'vitest';
import { IMAGE_MODELS, findImageModel } from '../catalogue/image-models';
import type { DeviceTestResult } from '../device/device-test';
import { recommendedImage } from './image-recommend';
import { imageCorrection, imageSeconds, type ImageSpeed } from './image-speed';

// The reference device: the image models' reference times were measured there.
const REFERENCE = { computeGflops: 809, features: ['shader-f16'] } as unknown as DeviceTestResult;
const none = () => null;

describe('recommendedImage', () => {
  it('picks the best model that makes an image in 30 s or less', () => {
    const pick = recommendedImage(IMAGE_MODELS, REFERENCE, none)!;
    expect(pick.why).toBe('runs-well');
    expect(pick.time!.seconds).toBeLessThanOrEqual(30);
    // No better-ranked model runs that fast here.
    for (const m of IMAGE_MODELS) {
      const time = imageSeconds(m, null, REFERENCE);
      if (m.pipeline && time && time.seconds <= 30) expect(m.rank).toBeLessThanOrEqual(pick.model.rank);
    }
  });

  it('on a slow GPU, the quickest model when none makes an image in 30 s', () => {
    // Slow enough that even the one-step SDXS takes over 30 s.
    const slow = { ...REFERENCE, computeGflops: 5 };
    const pick = recommendedImage(IMAGE_MODELS, slow, none)!;
    expect(pick.why).toBe('quickest');
    for (const m of IMAGE_MODELS) {
      const time = imageSeconds(m, null, slow);
      if (m.pipeline && time) expect(time.seconds).toBeGreaterThanOrEqual(pick.time!.seconds);
    }
  });

  it('recommends nothing before a benchmark', () => {
    expect(recommendedImage(IMAGE_MODELS, null, none)).toBeNull();
  });

  it('never picks a model the GPU cannot run', () => {
    const noF16 = { ...REFERENCE, features: [] } as unknown as DeviceTestResult;
    const pick = recommendedImage(IMAGE_MODELS, noF16, none);
    if (pick) expect(pick.model.requiredGpuFeatures ?? []).not.toContain('shader-f16');
  });

  it('every image model that runs in the app has reference times and a rank', () => {
    for (const m of IMAGE_MODELS.filter((m) => m.pipeline)) {
      expect(m.reference, m.id).toBeDefined();
      expect(m.rank, m.id).toBeGreaterThan(0);
    }
  });
});

describe('imageCorrection', () => {
  it('scales estimates by how measured runs here compare with them', () => {
    const turbo = findImageModel('sd-turbo')!;
    const estimate = imageSeconds(turbo, null, REFERENCE)!.seconds * 1000;
    // Measured at twice the estimate: other models' estimates double too.
    const twice: ImageSpeed = { fixedMs: estimate * 2, perStepMs: 0, measuredAt: 0 };
    const speeds = (id: string) => (id === 'sd-turbo' ? twice : null);
    expect(imageCorrection(IMAGE_MODELS, REFERENCE, speeds)).toBeCloseTo(2, 5);
    expect(imageCorrection(IMAGE_MODELS, REFERENCE, none)).toBe(1);
    // Clamped to between 1/3 and 3.
    const far: ImageSpeed = { fixedMs: estimate * 50, perStepMs: 0, measuredAt: 0 };
    expect(imageCorrection(IMAGE_MODELS, REFERENCE, (id) => (id === 'sd-turbo' ? far : null))).toBe(3);
  });
});
