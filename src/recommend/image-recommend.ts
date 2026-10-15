// Which image models run well here, and the one to recommend: as for chat models, the best one (by quality.ts's
// score) that runs well;
// with none, the quickest that runs at all; nothing before a benchmark. Pure: the caller supplies the measured speeds.

import { fitImage, imagePeakBytes } from './budget';
import { runsInApp, type ImageModelSpec } from '../catalogue/image-models';
import type { DeviceTestResult } from '../device/device-test';
import { imageQuality } from './quality';
import { imageCorrection, imageSeconds, measuredImageSpeed, type ImageSpeed } from './image-speed';

/** Above this, an image is "slow" (seconds at the default steps). */
export const SLOW_IMAGE_SECONDS = 30;

export type ImageVerdict = 'good' | 'slow' | 'may-not-fit' | 'unsupported' | 'untested' | 'preview';

export function imageVerdict(model: ImageModelSpec, test: DeviceTestResult | null, seconds: number | undefined): ImageVerdict {
  if (!runsInApp(model)) return 'preview';
  // Nothing is judged before a benchmark.
  if (!test) return 'untested';
  if ((model.requiredGpuFeatures ?? []).some((f) => !test.features.includes(f))) return 'unsupported';
  if (imageMemoryShortfall(model, test)) return 'may-not-fit';
  if (seconds === undefined) return 'untested';
  return seconds > SLOW_IMAGE_SECONDS ? 'slow' : 'good';
}

/** Peak memory of the model's smallest image (the least it can run with). */
export function smallestImagePeakBytes(model: ImageModelSpec): number {
  const { width, height } = smallestSize(model);
  return imagePeakBytes(model, width, height);
}

/**
 * When even the model's smallest image needs more memory than the benchmark's memory check held here: what it needs,
 * what was held, and whether the check stopped there without running out (it held all it tried). Undefined when it fits.
 */
export function imageMemoryShortfall(
  model: ImageModelSpec,
  test: DeviceTestResult,
): { neededBytes: number; heldBytes: number; checkedOnly: boolean } | undefined {
  if (!model.pipeline) return undefined;
  const { memory } = test;
  const heldBytes = memory.status === 'failed' ? memory.failedAtBytes : memory.status === 'ok' ? memory.testedBytes : undefined;
  if (heldBytes === undefined) return undefined;
  const neededBytes = smallestImagePeakBytes(model);
  return neededBytes > heldBytes ? { neededBytes, heldBytes, checkedOnly: memory.status === 'ok' } : undefined;
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
  /** The device's memory budget (budget.ts): models whose smallest image doesn't fit aren't picked. */
  budgetBytes?: number,
): ImagePick | null {
  // Nothing is recommended before a benchmark.
  if (!test) return null;
  const correction = imageCorrection(models, test, speeds);
  const fits = (m: ImageModelSpec) => budgetBytes === undefined || fitImage(m, budgetBytes, smallestSize(m), []).status !== 'too-big';
  const judged = models
    .filter(runsInApp)
    .filter(fits)
    .map((model) => {
      const time = imageSeconds(model, speeds(model.id), test, correction);
      return { model, time, verdict: imageVerdict(model, test, time?.seconds) };
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
