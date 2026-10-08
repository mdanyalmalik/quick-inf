// The one WebGPU adapter and device this worker uses. The device probe reads
// the adapter, and the runtime is handed the device created from it, so the
// GPU whose features we check is the GPU that runs the model (on dual-GPU
// laptops, separate requests could otherwise land on different GPUs).
//
// A WebGPU adapter can create only one device, and a lost device can't be
// revived; after a loss the whole worker is replaced, which starts afresh.

let adapterPromise: Promise<GPUAdapter | null> | null = null;
let devicePromise: Promise<GPUDevice> | null = null;

/** The adapter (physical GPU) chosen by the browser, preferring a discrete GPU. Null if none is usable. */
export function getGpuAdapter(): Promise<GPUAdapter | null> {
  adapterPromise ??= (async () => {
    if (!('gpu' in navigator) || !navigator.gpu) return null;
    try {
      return await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch {
      return null;
    }
  })();
  return adapterPromise;
}

/**
 * The device for running models, created once from {@link getGpuAdapter}.
 * Requests every feature the adapter offers (e.g. shader-f16) and its highest
 * limits, so the runtime can use the GPU fully.
 */
export function getGpuDevice(): Promise<GPUDevice> {
  devicePromise ??= (async () => {
    const adapter = await getGpuAdapter();
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    const requiredLimits: Record<string, number> = {};
    for (const key in adapter.limits) {
      const value = adapter.limits[key as keyof GPUSupportedLimits];
      if (typeof value === 'number') requiredLimits[key] = value;
    }
    return adapter.requestDevice({
      label: 'quick-inf',
      requiredFeatures: [...adapter.features] as GPUFeatureName[],
      requiredLimits,
    });
  })();
  // Let a failed request be retried rather than caching the failure. The
  // adapter may be spent by the attempt, so ask for a fresh one too.
  devicePromise.catch(() => {
    devicePromise = null;
    adapterPromise = null;
  });
  return devicePromise;
}
