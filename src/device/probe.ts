// Device capability probe. Results are hints for choosing an artifact, not
// proof that a model fits: only loading and running it proves that.

import { getGpuAdapter } from './gpu';
import { simulatedLimit, simulation } from './simulation';

export interface GpuReport {
  vendor: string;
  architecture: string;
  description: string;
  features: string[];
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number };
  isFallbackAdapter: boolean;
}

export interface DeviceReport {
  webgpu: 'available' | 'missing-api' | 'no-adapter';
  gpu?: GpuReport;
  /** navigator.deviceMemory (GB, coarse and capped by browsers), when exposed. */
  deviceMemoryGb?: number;
  hardwareConcurrency?: number;
  userAgent: string;
}

export async function probeDevice(): Promise<DeviceReport> {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const base = {
    userAgent: nav.userAgent,
    deviceMemoryGb: simulation()?.ramGb ?? nav.deviceMemory,
    hardwareConcurrency: nav.hardwareConcurrency,
  };
  if (!('gpu' in nav) || !nav.gpu) return { ...base, webgpu: 'missing-api' };

  // The same adapter the runtime's device is created from (see gpu.ts).
  const adapter = await getGpuAdapter();
  if (!adapter) return { ...base, webgpu: 'no-adapter' };

  const info = adapter.info;
  return {
    ...base,
    webgpu: 'available',
    gpu: {
      vendor: info?.vendor ?? '',
      architecture: info?.architecture ?? '',
      description: info?.description ?? '',
      features: [...adapter.features].sort(),
      limits: {
        maxBufferSize: simulatedLimit('maxBufferSize', adapter.limits.maxBufferSize),
        maxStorageBufferBindingSize: simulatedLimit('maxStorageBufferBindingSize', adapter.limits.maxStorageBufferBindingSize),
      },
      isFallbackAdapter: (info as GPUAdapterInfo & { isFallbackAdapter?: boolean })?.isFallbackAdapter ?? false,
    },
  };
}
