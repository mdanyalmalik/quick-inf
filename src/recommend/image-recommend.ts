// Which image models run well here, and the one to recommend: as for chat models, the best one (by quality.ts's
// score) that runs well;
// with none, the quickest that runs at all; nothing before a benchmark. Pure: the caller supplies the measured speeds.

import { runsInApp, type ImageModelSpec } from '../catalogue/image-models';
import type { DeviceTestResult } from '../device/device-test';
import { imageQuality } from './quality';
import { imageCorrection, imageSeconds, measuredImageSpeed, type ImageSpeed } from './image-speed';

/** Above this, an image is "slow" (seconds at the default steps). */
export const SLOW_IMAGE_SECONDS = 30;

export type ImageVerdict = 'good' | 'slow' | 'unsupported' | 'untested' | 'preview';

export function imageVerdict(model: ImageModelSpec, test: DeviceTestResult | null, seconds: number | undefined): ImageVerdict {
  if (!runsInApp(model)) return 'preview';
  if (test && (model.requiredGpuFeatures ?? []).some((f) => !test.features.includes(f))) return 'unsupported';
  if (seconds === undefined) return 'untested';
  return seconds > SLOW_IMAGE_SECONDS ? 'slow' : 'good';
}

export interface ImagePick {
  model: ImageModelSpec;
  /** Time per image here, when known. */
  time: { seconds: number; source: 'measured' | 'estimated' } | null;
  why: 'runs-well' | 'quickest';
}

export function recommendedImage(
  models: readonly ImageModelSpec[],
  test: DeviceTestResult | null,
  speeds: (id: string) => ImageSpeed | null = measuredImageSpeed,
): ImagePick | null {
  // Nothing is recommended before a benchmark.
  if (!test) return null;
  const correction = imageCorrection(models, test, speeds);
  const judged = models
    .filter(runsInApp)
    .map((model) => {
      const time = imageSeconds(model, speeds(model.id), test, correction);
      return { model, time, verdict: imageVerdict(model, test, time?.seconds) };
    })
    .filter((m) => m.verdict !== 'unsupported');
  const byTime = (a: (typeof judged)[number], b: (typeof judged)[number]) => a.time!.seconds - b.time!.seconds;
  const quality = imageQuality(models);
  const score = (m: ImageModelSpec) => quality.get(m.id)?.score ?? 0;
  const good = judged.filter((m) => m.verdict === 'good').sort((a, b) => score(b.model) - score(a.model) || byTime(a, b));
  if (good[0]) return { model: good[0].model, time: good[0].time, why: 'runs-well' };
  const timed = judged.filter((m) => m.time).sort(byTime);
  return timed[0] ? { model: timed[0].model, time: timed[0].time, why: 'quickest' } : null;
}
