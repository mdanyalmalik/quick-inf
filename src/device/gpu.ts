// The one WebGPU adapter and device this worker uses. The device probe reads
// the adapter, and the runtime is handed the device created from it, so the
// GPU whose features we check is the GPU that runs the model (on dual-GPU
// laptops, separate requests could otherwise land on different GPUs).
//
// A WebGPU adapter can create only one device, and a lost device can't be
// revived; after a loss the whole worker is replaced, which starts afresh.
//
// Strong references for the worker's lifetime: Chrome loses a device ("A valid
// external Instance reference no longer exists") when the GPU object or adapter
// it came from is garbage-collected while the device is in use. Mobile browsers
// collect aggressively, so this showed up on Android and not on desktops.
import { applyGpuBudget, MIN_LIMITS, simulation } from './simulation';
const held: { gpu?: GPU; adapter?: GPUAdapter; device?: GPUDevice } = {};

/**
 * Optional features to enable, when the GPU has them. Deliberately not "all of
 * them": with every feature on (notably subgroups), ONNX Runtime's multi-token
 * kernels reset an Android phone's GPU (Adreno) on the first prompt;
 * with only these, the same model runs correctly (see info/gameplan.md).
 */
const FEATURES: GPUFeatureName[] = ['shader-f16', 'timestamp-query'];

let adapterPromise: Promise<GPUAdapter | null> | null = null;
let devicePromise: Promise<GPUDevice> | null = null;

/** The adapter (physical GPU) chosen by the browser, preferring a discrete GPU. Null if none is usable. */
export function getGpuAdapter(): Promise<GPUAdapter | null> {
  adapterPromise ??= (async () => {
    if (!('gpu' in navigator) || !navigator.gpu || simulation()?.noWebgpu) return null;
    held.gpu = navigator.gpu;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) held.adapter = adapter;
      return adapter;
    } catch {
      return null;
    }
  })();
  return adapterPromise;
}

/**
 * The device for running models, created once from {@link getGpuAdapter}, with
 * the FEATURES the GPU has and its highest limits (the runtime needs more than
 * WebGPU's default storage buffers per shader).
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
    const sim = simulation();
    if (sim?.minLimits) Object.assign(requiredLimits, MIN_LIMITS);
    const device = await adapter.requestDevice({
      label: 'quick-inf',
      requiredFeatures: FEATURES.filter((f) => adapter.features.has(f) && !(f === 'shader-f16' && simulation()?.noF16)),
      requiredLimits,
    });
    if (sim?.gpuBudgetBytes) applyGpuBudget(device, sim.gpuBudgetBytes, sim.kill ? () => self.postMessage({ kind: 'simulated-kill' }) : undefined);
    held.device = device;
    return device;
  })();
  // Let a failed request be retried rather than caching the failure. The
  // adapter may be spent by the attempt, so ask for a fresh one too.
  devicePromise.catch(() => {
    devicePromise = null;
    adapterPromise = null;
  });
  return devicePromise;
}
