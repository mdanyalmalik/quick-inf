// Which of a model's artifacts this device can run.

import { engineName, type ModelSpec } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import type { CompatibilityReport } from './contract';

export interface CompatibilityOptions {
  /** Only this artifact id (development: testing a particular build). */
  onlyArtifact?: string;
}

/** The first artifact (they're listed preferred first) whose GPU features this device has. */
export function checkCompatibility(model: ModelSpec, device: DeviceReport, options: CompatibilityOptions = {}): CompatibilityReport {
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
    if (options.onlyArtifact && artifact.id !== options.onlyArtifact) continue;
    const missing = artifact.requiredGpuFeatures.filter((f) => !features.has(f));
    if (missing.length === 0) return { compatible: true, artifact, reasons };
    reasons.push(`${engineName(artifact.engine)} ${artifact.precision} needs GPU feature(s): ${missing.join(', ')}`);
  }
  return { compatible: false, reasons: [...new Set(reasons)] };
}
