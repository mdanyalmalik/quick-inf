// Which of a model's artifacts this device can run, whichever engine runs it.

import { engineName, type EngineId, type ModelSpec } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import type { CompatibilityReport } from './contract';

export interface CompatibilityOptions {
  /** Engines not to use for this model on this device (they failed here before). */
  avoid?: ReadonlySet<EngineId>;
  /** Only this engine (development: comparing engines). */
  only?: EngineId;
  /** Only this artifact id (development: testing a particular build). */
  onlyArtifact?: string;
}

/**
 * The first artifact (they're listed preferred first) whose GPU features this
 * device has. Engines in `avoid` are skipped only while another engine can run
 * the model; with nothing to fall back to, the failed engine is tried again.
 */
export function checkCompatibility(model: ModelSpec, device: DeviceReport, options: CompatibilityOptions = {}): CompatibilityReport {
  const report = firstRunnable(model, device, options);
  if (report.compatible || !options.avoid?.size) return report;
  const retry = firstRunnable(model, device, { ...options, avoid: undefined });
  return retry.compatible ? retry : report;
}

function firstRunnable(model: ModelSpec, device: DeviceReport, options: CompatibilityOptions): CompatibilityReport {
  if (device.webgpu !== 'available' || !device.gpu) {
    return {
      compatible: false,
      reasons: [
        device.webgpu === 'missing-api'
          ? 'This browser does not expose WebGPU.'
          : 'WebGPU is present but no GPU adapter is available.',
      ],
    };
  }
  const features = new Set(device.gpu.features);
  const reasons: string[] = [];
  for (const artifact of model.artifacts) {
    if (options.only && artifact.engine !== options.only) continue;
    if (options.onlyArtifact && artifact.id !== options.onlyArtifact) continue;
    if (options.avoid?.has(artifact.engine)) {
      reasons.push(`${engineName(artifact.engine)} failed on this device before.`);
      continue;
    }
    const missing = artifact.requiredGpuFeatures.filter((f) => !features.has(f));
    if (missing.length === 0) return { compatible: true, artifact, reasons };
    reasons.push(`${engineName(artifact.engine)} ${artifact.precision} needs GPU feature(s): ${missing.join(', ')}`);
  }
  return { compatible: false, reasons: [...new Set(reasons)] };
}
