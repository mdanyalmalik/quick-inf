// How long image models take here: the last measured run per model (from the image tab), and estimates for the
// rest from the device test. Estimates scale times measured on a reference Mac by compute speed, since a
// diffusion model's steps are compute-bound (large convolutions and attention over the whole image).

import { useSyncExternalStore } from 'react';
import { findImageModel, type ImageModelSpec, type ImageSettings } from '../catalogue/image-models';
import type { DeviceTestResult } from '../device/device-test';

export interface ImageSpeed {
  /** Text encoder and image decoder: the same whatever the steps. */
  fixedMs: number;
  perStepMs: number;
  measuredAt: number;
}

const key = (modelId: string) => `quick-inf:image-speed:${modelId}`;
const listeners = new Set<() => void>();
const cache = new Map<string, ImageSpeed | null>();

export function measuredImageSpeed(modelId: string): ImageSpeed | null {
  if (cache.has(modelId)) return cache.get(modelId)!;
  let speed: ImageSpeed | null = null;
  try {
    speed = JSON.parse(localStorage.getItem(key(modelId)) ?? 'null') as ImageSpeed | null;
  } catch {
    speed = null;
  }
  cache.set(modelId, speed);
  return speed;
}

/** Records a finished generation's timings (at the default size only: larger images take longer per step). */
export function recordImageSpeed(
  modelId: string,
  steps: number,
  settings: ImageSettings,
  t: { promptMs: number; denoiseMs: number; decodeMs: number },
): void {
  const model = findImageModel(modelId);
  const sizeInput = model?.inputs.find((i) => i.id === 'size');
  if (sizeInput?.kind === 'select' && settings.size !== undefined && settings.size !== sizeInput.default) return;
  const speed: ImageSpeed = { fixedMs: t.promptMs + t.decodeMs, perStepMs: t.denoiseMs / Math.max(1, steps), measuredAt: Date.now() };
  cache.set(modelId, speed);
  try {
    localStorage.setItem(key(modelId), JSON.stringify(speed));
  } catch {
    // Kept for this visit.
  }
  for (const l of listeners) l();
}

export function useImageSpeed(modelId: string): ImageSpeed | null {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => measuredImageSpeed(modelId),
  );
}

/** The model's default number of steps (1 for a model without a steps setting). */
export function defaultSteps(model: ImageModelSpec): number {
  const steps = model.inputs.find((i) => i.id === 'steps');
  return steps && steps.kind === 'range' ? steps.default : (model.steps ?? 1);
}

/** Seconds per image at the default steps, estimated from the reference times and the device test (or null). */
function estimatedSeconds(model: ImageModelSpec, test: DeviceTestResult | null): number | null {
  const ref = model.reference;
  if (!ref || !test) return null;
  const scale = ref.computeGflops / test.computeGflops;
  return ((ref.fixedMs + ref.perStepMs * defaultSteps(model)) * scale) / 1000;
}

function measuredSeconds(model: ImageModelSpec, measured: ImageSpeed): number {
  return (measured.fixedMs + measured.perStepMs * defaultSteps(model)) / 1000;
}

/**
 * How far this device is from the estimates, from image models that have run here: the geometric mean of measured
 * over estimated time, between 1/3 and 3 (1 when none has run). As for chat models, it corrects the estimates of
 * the models that haven't run here.
 */
export function imageCorrection(
  models: readonly ImageModelSpec[],
  test: DeviceTestResult | null,
  speeds: (id: string) => ImageSpeed | null = measuredImageSpeed,
): number {
  const ratios: number[] = [];
  for (const model of models) {
    const measured = speeds(model.id);
    const estimate = estimatedSeconds(model, test);
    if (measured && estimate) ratios.push(measuredSeconds(model, measured) / estimate);
  }
  if (!ratios.length) return 1;
  return Math.min(3, Math.max(1 / 3, Math.exp(ratios.reduce((s, r) => s + Math.log(r), 0) / ratios.length)));
}

/** Seconds per image at the default steps: measured here, else estimated (corrected by `correction`), or null. */
export function imageSeconds(
  model: ImageModelSpec,
  measured: ImageSpeed | null,
  test: DeviceTestResult | null,
  correction = 1,
): { seconds: number; source: 'measured' | 'estimated' } | null {
  if (measured) return { seconds: measuredSeconds(model, measured), source: 'measured' };
  const estimate = estimatedSeconds(model, test);
  return estimate === null ? null : { seconds: estimate * correction, source: 'estimated' };
}
