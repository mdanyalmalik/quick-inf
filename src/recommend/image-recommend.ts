// Which image models run well here, and the one to recommend: as for chat models, the best one (by quality.ts's
// score) that runs well;
// with none, the quickest that runs at all; nothing before a benchmark. Pure: the caller supplies the measured speeds.

import { imagePeakBytes, type Budget } from './budget';
import { runsInApp, type ImageModelSpec } from '../catalogue/image-models';
import type { DeviceTestResult } from '../device/device-test';
import { imageQuality } from './quality';
import { imageCorrection, imageSeconds, measuredImageSpeed, type ImageSpeed } from './image-speed';

/** Above this, an image is "slow" (seconds at the default steps). */
export const SLOW_IMAGE_SECONDS = 30;

export type ImageVerdict = 'good' | 'slow' | 'may-not-fit' | 'unsupported' | 'untested' | 'preview';

/** `budget`: the device's memory limit (budget.ts), if the benchmark's memory check has run. */
export function imageVerdict(model: ImageModelSpec, test: DeviceTestResult | null, seconds: number | undefined, budget?: Budget | null): ImageVerdict {
  if (!runsInApp(model)) return 'preview';
  // Nothing is judged before a benchmark.
  if (!test) return 'untested';
  if ((model.requiredGpuFeatures ?? []).some((f) => !test.features.includes(f))) return 'unsupported';
  if (budget && imageMemoryShortfall(model, budget)) return 'may-not-fit';
  if (seconds === undefined) return 'untested';
  return seconds > SLOW_IMAGE_SECONDS ? 'slow' : 'good';
}

/** Peak memory of the model's smallest image (the least it can run with). */
export function smallestImagePeakBytes(model: ImageModelSpec): number {
  const { width, height } = smallestSize(model);
  return imagePeakBytes(model, width, height);
}

/** What the model's smallest image needs, when that's over the device's memory limit; undefined when it fits. */
export function imageMemoryShortfall(model: ImageModelSpec, budget: Budget): number | undefined {
  if (!model.pipeline) return undefined;
  const needed = smallestImagePeakBytes(model);
  return needed > budget.bytes ? needed : undefined;
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
  /** The device's memory limit (budget.ts): models whose smallest image doesn't fit aren't picked. */
  budget?: Budget | null,
): ImagePick | null {
  // Nothing is recommended before a benchmark.
  if (!test) return null;
  const correction = imageCorrection(models, test, speeds);
  const judged = models
    .filter(runsInApp)
    .map((model) => {
      const time = imageSeconds(model, speeds(model.id), test, correction);
      return { model, time, verdict: imageVerdict(model, test, time?.seconds, budget) };
    })
    .filter((m) => m.verdict !== 'unsupported' && m.verdict !== 'may-not-fit');
  const byTime = (a: (typeof judged)[number], b: (typeof judged)[number]) => a.time!.seconds - b.time!.seconds;
  const quality = imageQuality(models);
  const score = (m: ImageModelSpec) => quality.get(m.id)?.score ?? 0;
  const good = judged.filter((m) => m.verdict === 'good').sort((a, b) => score(b.model) - score(a.model) || byTime(a, b));
  if (good[0]) return { model: good[0].model, time: good[0].time, why: 'runs-well' };
  const timed = judged.filter((m) => m.time).sort(byTime);
  return timed[0] ? { model: timed[0].model, time: timed[0].time, why: 'quickest' } : null;
}

/** A model's smallest image size option (its fixed size if it has no choice). */
function smallestSize(model: ImageModelSpec): { width: number; height: number } {
  const size = model.inputs.find((i) => i.kind === 'select' && i.id === 'size');
  const options = size?.kind === 'select' ? size.options.map((o) => o.value) : [];
  const all = [...options, model.sizeLabel.split(' or ')[0]!.replace(/ × /, 'x')].map((v) => {
    const [w, h] = v.split('x').map(Number);
    return { width: w || 512, height: h || 512 };
  });
  return all.sort((a, b) => a.width * a.height - b.width * b.height)[0]!;
}
