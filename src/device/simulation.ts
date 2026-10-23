// Development: run as a smaller device, to check the app still works (or fails
// cleanly) with less memory than the machine it's tested on.
//
// `?sim=<profile>` in the page URL, a preset name or comma-separated parts:
//   mobile        phone behaviour (phone catalogue, mobile memory caps)
//   gpu=<GB>      GPU memory budget: allocations past it fail like a real
//                 out-of-memory (an unusable buffer and a GPUOutOfMemoryError,
//                 or a RangeError for buffers mapped at creation)
//   min-limits    WebGPU's minimum buffer limits (256 MB per buffer, 128 MB per
//                 binding, as low-end phones have); the browser enforces these
//   ram=<GB>      the device memory reported to the app (navigator.deviceMemory)
//   no-webgpu     no WebGPU at all (as on browsers and devices without it): the CPU engine runs models
//   no-f16        a GPU without shader-f16 (16-bit maths)
//   kill          going past the GPU budget kills the page instead of failing the allocation, as iOS and
//                 Android do to a tab that uses too much memory (the page reloads at once, losing its state)
//   ort=<build>   force ONNX Runtime's build: jspi, asyncify or cpu (see runtime/onnx/ort-env.ts)
//   watchdog=<s>  how long a silent worker is waited for before it's replaced (runtime/client.ts; default 300)
//   no-bgfetch    no Background Fetch: downloads run in the page, as on Safari (runtime/background-download.ts)
//   bgfetch-stall Background Fetch accepts downloads and never starts them (seen on Samsung Internet)
//   stream=<GB>   stream every mixture-of-experts model's experts (runtime/onnx/moe-stream.ts), with this much GPU
//                 memory for cached ones, whatever was chosen in the load panel
// e.g. `?sim=phone` or `?sim=mobile,gpu=2,min-limits,ram=4`.
//
// The budget applies to the app's GPU device (the ONNX engine and the device
// check). The page passes the profile to the runtime worker in the worker's name.

export interface Simulation {
  text: string;
  mobile: boolean;
  gpuBudgetBytes?: number;
  minLimits: boolean;
  ramGb?: number;
  noWebgpu?: boolean;
  noF16?: boolean;
  /** The GPU device without subgroups (as on Qualcomm GPUs), to compare speeds. */
  noSubgroups?: boolean;
  kill?: boolean;
  ort?: 'jspi' | 'asyncify' | 'cpu';
  watchdogSeconds?: number;
  noBackgroundFetch?: boolean;
  stalledBackgroundFetch?: boolean;
  streamExperts?: number;
}

const PRESETS: Record<string, string> = {
  phone: 'mobile,gpu=3,min-limits,ram=8',
  'low-phone': 'mobile,gpu=1.5,min-limits,ram=4',
  'laptop-8gb': 'gpu=4,ram=8',
};

/** WebGPU's default (minimum guaranteed) limits. */
export const MIN_LIMITS = { maxBufferSize: 268_435_456, maxStorageBufferBindingSize: 134_217_728 } as const;

const WORKER_NAME_PREFIX = 'sim:';

export function parseSimulation(text: string): Simulation | null {
  const expanded = PRESETS[text] ?? text;
  if (!expanded) return null;
  const sim: Simulation = { text, mobile: false, minLimits: false };
  for (const part of expanded.split(',').map((p) => p.trim()).filter(Boolean)) {
    const [key, value] = part.split('=') as [string, string | undefined];
    if (key === 'mobile') sim.mobile = true;
    else if (key === 'min-limits') sim.minLimits = true;
    else if (key === 'gpu' && Number(value) > 0) sim.gpuBudgetBytes = Number(value) * 1e9;
    else if (key === 'ram' && Number(value) > 0) sim.ramGb = Number(value);
    else if (key === 'no-webgpu') sim.noWebgpu = true;
    else if (key === 'no-f16') sim.noF16 = true;
    else if (key === 'no-subgroups') sim.noSubgroups = true;
    else if (key === 'kill') sim.kill = true;
    else if (key === 'no-bgfetch') sim.noBackgroundFetch = true;
    else if (key === 'bgfetch-stall') sim.stalledBackgroundFetch = true;
    else if (key === 'stream' && Number(value) > 0) sim.streamExperts = Number(value);
    else if (key === 'ort' && (value === 'jspi' || value === 'asyncify' || value === 'cpu')) sim.ort = value;
    else if (key === 'watchdog' && Number(value) > 0) sim.watchdogSeconds = Number(value);
    else throw new Error(`Unknown ?sim part "${part}"`);
  }
  return sim;
}

let current: Simulation | null | undefined;

/** The simulated device, if any (development builds only). */
export function simulation(): Simulation | null {
  if (current !== undefined) return current;
  current = null;
  if (!import.meta.env.DEV) return current;
  const fromPage = new URLSearchParams(globalThis.location?.search ?? '').get('sim');
  const workerName = (globalThis as { name?: string }).name ?? '';
  const text = fromPage ?? (workerName.startsWith(WORKER_NAME_PREFIX) ? workerName.slice(WORKER_NAME_PREFIX.length) : null);
  if (text) {
    current = parseSimulation(text);
    console.info(`Simulating a smaller device (?sim=${text}):`, current);
  }
  return current;
}

/** The runtime worker's name, carrying the page's simulation to it. */
export function workerName(): string | undefined {
  const sim = simulation();
  return sim ? `${WORKER_NAME_PREFIX}${sim.text}` : undefined;
}

/** A limit as the simulated device has it. */
export function simulatedLimit(key: keyof typeof MIN_LIMITS, real: number): number {
  return simulation()?.minLimits ? Math.min(real, MIN_LIMITS[key]) : real;
}

/**
 * Makes `device` fail allocations past `budget` bytes the way a real out-of-memory does. Live bytes are
 * counted from createBuffer to destroy (the runtime destroys what it frees).
 */
export function applyGpuBudget(device: GPUDevice, budget: number, onKill?: () => void): void {
  let live = 0;
  // Out-of-memory error scopes that a simulated failure happened inside.
  const scopes: { filter: GPUErrorFilter; failed: GPUOutOfMemoryError | null }[] = [];
  const createBuffer = device.createBuffer.bind(device);
  const pushErrorScope = device.pushErrorScope.bind(device);
  const popErrorScope = device.popErrorScope.bind(device);

  device.pushErrorScope = (filter) => {
    scopes.push({ filter, failed: null });
    pushErrorScope(filter);
  };
  device.popErrorScope = async () => {
    const scope = scopes.pop();
    const real = await popErrorScope();
    return real ?? scope?.failed ?? null;
  };
  device.createBuffer = (descriptor) => {
    const size = descriptor.size;
    if (live + size > budget) {
      // A mobile OS doesn't report this: it ends the page.
      if (onKill) {
        onKill();
        // Nothing after this may run (no error to catch, no clean-up): wait here until the page is gone.
        for (;;);
      }
      const message = `Simulated out of memory: ${Math.round(size / 1e6)} MB requested with ${Math.round(live / 1e6)} of ${Math.round(budget / 1e6)} MB in use`;
      if (descriptor.mappedAtCreation) throw new RangeError(message);
      const error = new GPUOutOfMemoryError(message);
      const scope = [...scopes].reverse().find((s) => s.filter === 'out-of-memory');
      if (scope) scope.failed ??= error;
      else device.dispatchEvent(new GPUUncapturedErrorEvent('uncapturederror', { error }));
      // An unusable buffer, as a failed allocation gives: any use of it is an error.
      const buffer = createBuffer({ ...descriptor, size: 4, mappedAtCreation: false });
      buffer.destroy();
      return buffer;
    }
    const buffer = createBuffer(descriptor);
    live += size;
    const destroy = buffer.destroy.bind(buffer);
    let destroyed = false;
    buffer.destroy = () => {
      if (!destroyed) {
        destroyed = true;
        live -= size;
      }
      destroy();
    };
    return buffer;
  };
}
