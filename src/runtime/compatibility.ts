// Which of a model's artifacts this device can run, and on what: the GPU (WebGPU) when it's usable, otherwise
// ONNX Runtime's CPU engine for models small enough for WebAssembly memory.

import { engineName, weightBytes, type ModelArtifact, type ModelSpec } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import type { CompatibilityReport } from './contract';

/**
 * Weights the CPU engine will take: they live in WebAssembly memory (4 GB at most, with the runtime's own
 * buffers on top), and a bigger model would be too slow on a CPU to be useful anyway.
 */
export const CPU_MAX_WEIGHT_BYTES = 2e9;

export interface CompatibilityOptions {
  /** Only this artifact id (development: testing a particular build). */
  onlyArtifact?: string;
  /** Run on the CPU even if WebGPU is usable (no usable GPU path in this browser, e.g. WebKit without JSPI). */
  cpuOnly?: boolean;
}

/** Whether models run on the GPU here or only on the CPU. */
export function processorFor(device: DeviceReport, cpuOnly = false): 'gpu' | 'cpu' {
  return !cpuOnly && device.webgpu === 'available' && device.gpu ? 'gpu' : 'cpu';
}

/**
 * The artifact to run and where. On the GPU: the first artifact (they're listed preferred first) whose GPU
 * features this device has. On the CPU: the smallest-memory full-precision-maths artifact (q4 before q4f16:
 * ONNX Runtime's CPU kernels cover float32 best), if it's small enough.
 */
export function checkCompatibility(model: ModelSpec, device: DeviceReport, options: CompatibilityOptions = {}): CompatibilityReport {
  const candidates = model.artifacts.filter((a) => !options.onlyArtifact || a.id === options.onlyArtifact);
  if (processorFor(device, options.cpuOnly) === 'cpu') return cpuCompatibility(model, candidates, device);
  const features = new Set(device.gpu!.features);
  const reasons: string[] = [];
  for (const artifact of candidates) {
    const missing = artifact.requiredGpuFeatures.filter((f) => !features.has(f));
    if (missing.length === 0) return { compatible: true, artifact, processor: 'gpu', reasons };
    reasons.push(`${engineName(artifact.engine)} ${artifact.precision} needs GPU feature(s): ${missing.join(', ')}`);
  }
  return { compatible: false, reasons: [...new Set(reasons)] };
}

function cpuCompatibility(model: ModelSpec, candidates: ModelArtifact[], device: DeviceReport): CompatibilityReport {
  const why = noGpuReason(device);
  const fit = candidates.filter((a) => weightBytes(a) <= CPU_MAX_WEIGHT_BYTES);
  if (fit.length === 0) {
    return { compatible: false, reasons: [`${why} ${model.name} is too big to run on the CPU here.`] };
  }
  const artifact = [...fit].sort((a, b) => cpuPreference(a) - cpuPreference(b) || weightBytes(a) - weightBytes(b))[0]!;
  return { compatible: true, artifact, processor: 'cpu', reasons: [`${why} It runs on the CPU instead, more slowly.`] };
}

/** Lower is better: float32 maths first. */
function cpuPreference(artifact: ModelArtifact): number {
  return /f16/.test(artifact.precision) ? 1 : 0;
}

export function noGpuReason(device: DeviceReport): string {
  if (device.webgpu === 'missing-api') return 'This browser doesn’t offer WebGPU.';
  if (device.webgpu === 'no-adapter') return 'WebGPU is present but no GPU is available to it.';
  return 'The GPU can’t be used safely in this browser.';
}
