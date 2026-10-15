// What this browser has shown it can hold (see budget.ts): the largest estimated peak that ran, and the smallest
// that failed for lack of memory or got the page killed. Kept in localStorage (a convenience: losing it only means
// starting again from the class budget), updated from the runtime's outcomes and from leftover breadcrumbs.

import { defaultSettings, findImageModel, outputSize, type ImageModelSpec, type ImageSettings } from '../catalogue/image-models';
import { catalogue, findModel } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import { isMobileDevice } from '../device/mobile';
import type { Breadcrumb } from '../runtime/breadcrumbs';
import type { RuntimeClient } from '../runtime/client';
import { deviceBudget, deviceClass, fitChat, fitImage, imagePeakBytes, MIN_CONTEXT_TOKENS, type Budget, type DeviceEvidence, type Fit } from './budget';
import { memoryNeeded } from './estimate';
import { pickArtifact } from './recommend';

const KEY = 'quick-inf:memory-evidence';

/** Failures that say the device ran out of memory (not network, storage or user problems). */
export const MEMORY_FAILURES = new Set(['out-of-memory', 'device-lost', 'page-killed']);

export interface EvidenceLog extends DeviceEvidence {
  /** The latest few memory failures, for explaining the budget. */
  failures: { modelId: string; bytes: number; code: string; at: number }[];
}

const listeners = new Set<() => void>();
let current: EvidenceLog = read();
/** This device, once probed (see trackMemoryEvidence). */
let deviceReport: DeviceReport | null = null;

let cachedBudget: Budget | null = null;

/** The device's memory budget now, or null until the device has been probed (a stable object between changes). */
export function currentBudget(): Budget | null {
  return cachedBudget;
}

function changed(): void {
  cachedBudget = deviceReport ? budgetFor(deviceReport, current) : null;
  for (const l of listeners) l();
}

/** The context length chat should use with `modelId` here: its default, or less if that doesn't fit the budget. */
export function fittedContext(modelId: string): number | undefined {
  const model = findModel(modelId);
  const budget = currentBudget();
  if (!model || !budget || !deviceReport) return model?.generation.contextTokens;
  const artifact = (deviceReport.gpu && pickArtifact(model, deviceReport.gpu.features)) || model.artifacts[0]!;
  const fit = fitChat(model, artifact, budget.bytes);
  return fit.status === 'too-big' ? MIN_CONTEXT_TOKENS : fit.value;
}

function read(): EvidenceLog {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) ?? 'null') as EvidenceLog | null;
    if (parsed && Array.isArray(parsed.failures)) return parsed;
  } catch {
    // Unreadable or unavailable: start from nothing.
  }
  return { failures: [] };
}

function write(next: EvidenceLog): void {
  current = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Kept for this visit only.
  }
  changed();
}

export const memoryEvidence = {
  get: (): EvidenceLog => current,
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  /** A run with this estimated peak worked here. */
  ran(bytes: number): void {
    if (bytes > (current.provenBytes ?? 0)) write({ ...current, provenBytes: bytes });
  },
  /** A run with this estimated peak failed for lack of memory (or killed the page). */
  failed(modelId: string, bytes: number, code: string): void {
    const failures = [...current.failures, { modelId, bytes, code, at: Date.now() }].slice(-10);
    const failedBytes = Math.min(current.failedBytes ?? Infinity, bytes);
    // A failure below what once ran means the earlier success was luckier than the device is: trust the failure.
    const provenBytes = current.provenBytes !== undefined && current.provenBytes >= failedBytes ? undefined : current.provenBytes;
    write({ failures, failedBytes, provenBytes });
  },
  /** The benchmark's memory check held this much here: it replaces the starting guess, and outdates failures above it. */
  measured(bytes: number): void {
    const failedBytes = current.failedBytes !== undefined && current.failedBytes > bytes ? current.failedBytes : undefined;
    write({ ...current, measuredBytes: bytes, failedBytes });
  },
  reset(): void {
    write({ failures: [] });
  },
};

/** Estimated peak memory of running `modelId` (a chat model at `contextTokens`, an image model at its settings). */
export function peakFor(
  modelId: string,
  device: Pick<DeviceReport, 'gpu'> | null,
  options: { contextTokens?: number; settings?: ImageSettings } = {},
): number | undefined {
  const chat = findModel(modelId);
  if (chat) {
    const artifact = (device?.gpu && pickArtifact(chat, device.gpu.features)) || chat.artifacts[0];
    return artifact ? memoryNeeded(chat, artifact, options.contextTokens) : undefined;
  }
  const image = findImageModel(modelId);
  if (!image?.pipeline) return undefined;
  const { width, height } = outputSize(image, options.settings ?? {});
  return imagePeakBytes(image, width, height);
}

/** Reads a device report into budget signals. */
export function budgetFor(device: DeviceReport, evidence: DeviceEvidence = current): Budget {
  const ua = device.userAgent;
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && isMobileDevice());
  const cls = deviceClass({
    webgpu: device.webgpu === 'available',
    ios,
    android: /Android/.test(ua),
    gpuVendor: device.gpu?.vendor.toLowerCase(),
    deviceMemoryGb: device.deviceMemoryGb,
  });
  return deviceBudget(cls, device.deviceMemoryGb, evidence);
}

/** Records the risky steps that killed the page last visit as memory failures. */
export function noteCrashes(crashes: readonly Breadcrumb[], device: Pick<DeviceReport, 'gpu'> | null): void {
  for (const crash of crashes) {
    // The benchmark's memory check: what it tried to hold was too much.
    if (crash.step === 'benchmark' && crash.bytes) memoryEvidence.failed('benchmark', crash.bytes, 'page-killed');
    if (!crash.modelId) continue;
    const bytes = peakFor(crash.modelId, device);
    if (bytes) memoryEvidence.failed(crash.modelId, bytes, 'page-killed');
  }
}

/**
 * Keeps the evidence up to date from the runtime: what loaded and ran raises the proven peak, out-of-memory and
 * lost-GPU failures (and last visit's page kills) lower the limit. Call once at start-up.
 */
export function trackMemoryEvidence(client: RuntimeClient): void {
  void client
    .check(catalogue[0]!.id)
    .then((check) => {
      deviceReport = check.device;
      changed();
      noteCrashes(client.getCrashes(), check.device);
    })
    .catch(() => noteCrashes(client.getCrashes(), null));
  client.onMeasurement((m) => {
    if (m.kind === 'load') {
      // A chat load proves the weights fit, and a short conversation's memory is small beside them (longer ones are
      // proven by their replies). An image model is proven by its first image.
      if (!findModel(m.modelId)) return;
      const bytes = peakFor(m.modelId, null, { contextTokens: MIN_CONTEXT_TOKENS });
      if (!bytes) return;
      if (m.ok) memoryEvidence.ran(bytes);
      else if (MEMORY_FAILURES.has(m.errorCode)) memoryEvidence.failed(m.modelId, bytes, m.errorCode);
    } else if (m.kind === 'reply') {
      const bytes = peakFor(m.modelId, null, { contextTokens: m.result.stats.promptTokens + m.result.stats.generatedTokens });
      if (bytes) memoryEvidence.ran(bytes);
    } else if (m.kind === 'reply-failed') {
      const bytes = peakFor(m.modelId, null);
      if (bytes && MEMORY_FAILURES.has(m.errorCode)) memoryEvidence.failed(m.modelId, bytes, m.errorCode);
    } else if (m.kind === 'image') {
      const bytes = peakFor(m.modelId, null, { settings: m.settings });
      if (!bytes) return;
      if (m.ok) memoryEvidence.ran(bytes);
      else if (m.errorCode && MEMORY_FAILURES.has(m.errorCode)) memoryEvidence.failed(m.modelId, bytes, m.errorCode);
    }
  });
}

const sizeOf = (value: string) => {
  const [width, height] = value.split('x').map(Number);
  return { width: width || 512, height: height || 512 };
};

/** How an image model's default size fits this device's budget (null until probed, or for a fixed-size model). */
export function imageFit(spec: ImageModelSpec): Fit<{ width: number; height: number }> | null {
  const budget = currentBudget();
  if (!budget || !spec.pipeline) return null;
  const size = spec.inputs.find((i) => i.kind === 'select' && i.id === 'size');
  const wanted = outputSize(spec, defaultSettings(spec));
  const sizes = size?.kind === 'select' ? size.options.map((o) => sizeOf(o.value)) : [];
  return fitImage(spec, budget.bytes, wanted, sizes);
}

/** The model's default settings, with the image size reduced to what fits this device (see fitImage). */
export function fittedImageSettings(spec: ImageModelSpec): ImageSettings {
  const settings = defaultSettings(spec);
  const fit = imageFit(spec);
  if (fit?.status === 'reduced') settings.size = `${fit.value.width}x${fit.value.height}`;
  return settings;
}
