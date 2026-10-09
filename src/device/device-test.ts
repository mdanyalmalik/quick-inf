// Quick device test: small WebGPU programs that measure what decides how fast a
// language model runs here, without downloading a model. Runs in the inference
// worker on the same GPU device the runtime uses (see gpu.ts).
//
//   bandwidth   Read a large buffer repeatedly: the raw memory speed.
//   matvec      Multiply a vector by a 4-bit quantized matrix, the operation that
//               reads the weights for every generated token. Phone GPUs reach a
//               much smaller share of their raw bandwidth here than desktop ones
//               (unpacking 4-bit values costs them more), so this, not raw
//               bandwidth, sets generation speed.
//   compute     A tiled matrix multiplication (fp16 when available). Sets how
//               fast prompts are processed.
//   dispatch    Hundreds of tiny dependent GPU jobs in one pass: GPU-side cost
//               per kernel.
//   step        The same jobs issued the way the runtime issues kernels (a bind
//               group and uniform write per job, a submit every 16 jobs), timed
//               on the wall clock: includes the CPU and driver cost per kernel,
//               which on phones is many times the GPU-side cost and sets the
//               fixed per-token time of small models.
//   roundTrip   Submit a job and read a few bytes back, as the runtime does for
//               every token to pick the next one.
//   memory      Try to hold N bytes of GPU memory (capped at what the largest
//               catalogue model needs), then free it. Never probes for the maximum.
//
// The whole set runs ROUNDS times and each result is the median across rounds
// (`spread` records how far apart the rounds were). Within a round, each
// measurement runs after warm-up passes (so the GPU reaches full clock) and is
// repeated: throughputs keep the fastest pass (what the hardware can sustain;
// slower passes are interference), latencies keep the median. GPU timestamps are
// used when the device supports them; otherwise wall-clock time.

import { getGpuAdapter, getGpuDevice } from './gpu';

export interface DeviceTestResult {
  /** 2 adds matvecGBps, stepUs, rounds and spread. */
  version: 1 | 2;
  ranAt: number;
  durationMs: number;
  gpu: { vendor: string; architecture: string; description: string };
  /** WebGPU features of the device (e.g. shader-f16), which decide which model files can run. */
  features: string[];
  timing: 'timestamp' | 'wall-clock';
  /** Sustained read bandwidth, GB/s (1e9 bytes). */
  bandwidthGBps: number;
  /** Matrix-multiply throughput, GFLOP/s, at the precision models use here. */
  computeGflops: number;
  computePrecision: 'f16' | 'f32';
  /** Cost of one small dependent GPU job, microseconds. */
  dispatchUs: number;
  /** Weight bytes per second through a 4-bit matrix-vector product, GB/s (version 2). */
  matvecGBps?: number;
  /** Wall-clock cost per kernel issued like the runtime does, microseconds (version 2). */
  stepUs?: number;
  /** Measurement rounds and, per metric, (max − min) ÷ median across them (version 2). */
  rounds?: number;
  spread?: Partial<Record<RoundMetric, number>>;
  /** Submit + read back a few bytes, milliseconds. */
  roundTripMs: number;
  memory:
    | { status: 'ok'; testedBytes: number }
    | { status: 'failed'; testedBytes: number; failedAtBytes: number }
    | { status: 'skipped'; reason: string };
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number };
  /** navigator.deviceMemory (GB, coarse, capped at 8 by browsers), if exposed. */
  deviceMemoryGb?: number;
}

export interface DeviceTestOptions {
  /** Bytes the memory check tries to hold; 0 or undefined skips it. */
  memoryBytes?: number;
  /** Why the memory check is skipped (e.g. a model is loaded), if it is. */
  skipMemoryReason?: string;
  onProgress?: (step: DeviceTestStep) => void;
}

export type DeviceTestStep = 'bandwidth' | 'compute' | 'dispatch' | 'memory';

type RoundMetric = 'bandwidthGBps' | 'matvecGBps' | 'computeGflops' | 'dispatchUs' | 'stepUs' | 'roundTripMs';

// WebGPU flag values (fixed by the spec; the DOM type library here lacks the constants).
const GPUBufferUsage = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80, QUERY_RESOLVE: 0x200 } as const;
const GPUMapMode = { READ: 0x1 } as const;

const ROUNDS = 3;
const WARMUP = 3;
const REPEATS = 5;
const BANDWIDTH_BYTES = 256 * 1024 * 1024;
const MATMUL_N = 1024;
const DISPATCH_JOBS = 400;
/** 4-bit weights for the matvec test: 64 MB, bigger than GPU caches. */
const MATVEC_K = 4096;
const MATVEC_ROWS = 32768;
/** Jobs per wall-clock step sample, and how often the runtime submits (ONNX Runtime WebGPU flushes every 16). */
const STEP_JOBS = 256;
const STEP_FLUSH = 16;
const MEMORY_CHUNK = 256 * 1024 * 1024;
/** Sustained GPU load before measuring, so the GPU reaches its working clock. */
const WARM_UP_MS = 500;

export async function runDeviceTest(options: DeviceTestOptions = {}): Promise<DeviceTestResult> {
  const start = performance.now();
  const adapter = await getGpuAdapter();
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await getGpuDevice();
  const timer = new GpuTimer(device);
  const info = adapter.info;

  const f16 = device.features.has('shader-f16');
  const rounds: Record<RoundMetric, number>[] = [];
  for (let round = 0; round < ROUNDS; round++) {
    options.onProgress?.('bandwidth');
    // GPUs idle at low clocks and only speed up under sustained load; without this,
    // results jump between clock states from one run to the next.
    await sustainLoad(device, round === 0 ? WARM_UP_MS : WARM_UP_MS / 4);
    const bandwidthGBps = await measureBandwidth(device, timer);
    const matvecGBps = await measureMatvec(device, timer);
    options.onProgress?.('compute');
    const computeGflops = await measureCompute(device, timer, f16);
    options.onProgress?.('dispatch');
    const dispatchUs = await measureDispatch(device, timer);
    const stepUs = await measureStep(device);
    const roundTripMs = await measureRoundTrip(device);
    rounds.push({ bandwidthGBps, matvecGBps, computeGflops, dispatchUs, stepUs, roundTripMs });
  }
  const median = (metric: RoundMetric) => medianValue(rounds.map((r) => r[metric]));
  const spread: Partial<Record<RoundMetric, number>> = {};
  for (const metric of Object.keys(rounds[0]!) as RoundMetric[]) {
    const values = rounds.map((r) => r[metric]);
    spread[metric] = Math.round(((Math.max(...values) - Math.min(...values)) / median(metric)) * 1000) / 1000;
  }

  let memory: DeviceTestResult['memory'];
  if (options.skipMemoryReason) memory = { status: 'skipped', reason: options.skipMemoryReason };
  else if (!options.memoryBytes) memory = { status: 'skipped', reason: 'nothing to test' };
  else {
    options.onProgress?.('memory');
    memory = await checkMemory(device, options.memoryBytes);
  }
  timer.destroy();

  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    version: 2,
    ranAt: Date.now(),
    durationMs: performance.now() - start,
    gpu: { vendor: info.vendor, architecture: info.architecture, description: info.description },
    features: [...device.features].sort(),
    timing: timer.supported ? 'timestamp' : 'wall-clock',
    bandwidthGBps: median('bandwidthGBps'),
    computeGflops: median('computeGflops'),
    computePrecision: f16 ? 'f16' : 'f32',
    dispatchUs: median('dispatchUs'),
    matvecGBps: median('matvecGBps'),
    stepUs: median('stepUs'),
    roundTripMs: median('roundTripMs'),
    rounds: ROUNDS,
    spread,
    memory,
    limits: {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    },
    deviceMemoryGb: nav.deviceMemory,
  };
}

/** Times compute passes with GPU timestamps when available, otherwise wall clock. */
class GpuTimer {
  readonly supported: boolean;
  private readonly querySet?: GPUQuerySet;
  private readonly resolve?: GPUBuffer;
  private readonly readback?: GPUBuffer;

  constructor(private readonly device: GPUDevice) {
    this.supported = device.features.has('timestamp-query');
    if (this.supported) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
  }

  /** Runs `encode` inside one compute pass and returns its duration in milliseconds. */
  async time(encode: (pass: GPUComputePassEncoder) => void): Promise<number> {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass(
      this.supported
        ? { timestampWrites: { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
        : undefined,
    );
    encode(pass);
    pass.end();
    if (this.supported) {
      encoder.resolveQuerySet(this.querySet!, 0, 2, this.resolve!, 0);
      encoder.copyBufferToBuffer(this.resolve!, 0, this.readback!, 0, 16);
    }
    const wallStart = performance.now();
    this.device.queue.submit([encoder.finish()]);
    if (!this.supported) {
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - wallStart;
    }
    await this.readback!.mapAsync(GPUMapMode.READ);
    const [begin, end] = new BigUint64Array(this.readback!.getMappedRange().slice(0));
    this.readback!.unmap();
    const ns = Number(end! - begin!);
    // Some drivers return zero or garbage for very short passes; fall back to wall clock then.
    if (ns <= 0) {
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - wallStart;
    }
    return ns / 1e6;
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolve?.destroy();
    this.readback?.destroy();
  }
}

async function samples(run: () => Promise<number>, warmup = WARMUP, repeats = REPEATS): Promise<number[]> {
  for (let i = 0; i < warmup; i++) await run();
  const out: number[] = [];
  for (let i = 0; i < repeats; i++) out.push(await run());
  return out.sort((a, b) => a - b);
}

/** Fastest pass, for throughputs. */
async function fastestOf(run: () => Promise<number>, warmup = WARMUP, repeats = REPEATS): Promise<number> {
  return (await samples(run, warmup, repeats))[0]!;
}

/** Median pass, for latencies. */
async function medianOf(run: () => Promise<number>): Promise<number> {
  return medianValue(await samples(run));
}

function medianValue(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function pipeline(device: GPUDevice, code: string): GPUComputePipeline {
  return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
}

async function measureBandwidth(device: GPUDevice, timer: GpuTimer): Promise<number> {
  const bytes = Math.min(BANDWIDTH_BYTES, device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const size = Math.floor(bytes / 16) * 16;
  const workgroups = 4096;
  const threads = workgroups * 256;
  const src = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  // Non-zero contents, so no hardware compression of zero pages can shortcut the reads.
  const chunk = new Uint32Array(4 * 1024 * 1024);
  for (let i = 0; i < chunk.length; i++) chunk[i] = (i * 2654435761) >>> 0;
  for (let offset = 0; offset < size; offset += chunk.byteLength) {
    device.queue.writeBuffer(src, offset, chunk, 0, Math.min(chunk.length, (size - offset) / 4));
  }
  const dst = device.createBuffer({ size: threads * 4, usage: GPUBufferUsage.STORAGE });
  const pipe = pipeline(
    device,
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = groups.x * 256u;
  let n = arrayLength(&src);
  var acc = vec4<u32>(0u);
  for (var i = gid.x; i < n; i = i + stride) {
    acc = acc ^ src[i];
  }
  // Writing the result keeps the compiler from skipping the reads.
  dst[gid.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}`,
  );
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
    ],
  });
  const ms = await fastestOf(() =>
    timer.time((pass) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(workgroups);
    }),
  );
  src.destroy();
  dst.destroy();
  return size / (ms / 1000) / 1e9;
}

async function measureCompute(
  device: GPUDevice,
  timer: GpuTimer,
  f16: boolean,
  warmup = WARMUP,
  repeats = REPEATS,
): Promise<number> {
  const n = MATMUL_N;
  const type = f16 ? 'f16' : 'f32';
  const elementBytes = f16 ? 2 : 4;
  const make = () => device.createBuffer({ size: n * n * elementBytes, usage: GPUBufferUsage.STORAGE });
  const [a, b, c] = [make(), make(), make()];
  const pipe = pipeline(
    device,
    /* wgsl */ `
${f16 ? 'enable f16;' : ''}
const N = ${n}u;
const TILE = 16u;
@group(0) @binding(0) var<storage, read> a: array<${type}>;
@group(0) @binding(1) var<storage, read> b: array<${type}>;
@group(0) @binding(2) var<storage, read_write> c: array<${type}>;
var<workgroup> ta: array<array<${type}, 16>, 16>;
var<workgroup> tb: array<array<${type}, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  var sum = ${type}(0);
  for (var t = 0u; t < N / TILE; t = t + 1u) {
    ta[l.y][l.x] = a[g.y * N + t * TILE + l.x];
    tb[l.y][l.x] = b[(t * TILE + l.y) * N + g.x];
    workgroupBarrier();
    for (var k = 0u; k < TILE; k = k + 1u) {
      sum = sum + ta[l.y][k] * tb[k][l.x];
    }
    workgroupBarrier();
  }
  c[g.y * N + g.x] = sum;
}`,
  );
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [a, b, c].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const ms = await fastestOf(
    () =>
      timer.time((pass) => {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(n / 16, n / 16);
      }),
    warmup,
    repeats,
  );
  for (const buffer of [a, b, c]) buffer.destroy();
  return (2 * n * n * n) / (ms / 1000) / 1e9;
}

/**
 * Keeps the GPU fully busy for about `ms` milliseconds, so it reaches its top clock
 * state before measuring. Work is queued back to back (two batches in flight):
 * waiting for each pass to finish leaves gaps, and some GPUs (Apple's) then stay
 * in a lower clock state.
 */
async function sustainLoad(device: GPUDevice, ms: number): Promise<void> {
  const out = device.createBuffer({ size: 1024 * 256 * 4, usage: GPUBufferUsage.STORAGE });
  const pipe = pipeline(
    device,
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  var x = f32(g.x) * 1e-6;
  for (var i = 0u; i < 2048u; i = i + 1u) { x = x * 0.9999 + 0.0001; }
  out[g.x] = x;
}`,
  );
  const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
  const submitBatch = () => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    for (let i = 0; i < 4; i++) pass.dispatchWorkgroups(1024);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return device.queue.onSubmittedWorkDone();
  };
  const end = performance.now() + ms;
  let previous = submitBatch();
  while (performance.now() < end) {
    const next = submitBatch();
    await previous;
    previous = next;
  }
  await previous;
  out.destroy();
}

async function measureDispatch(device: GPUDevice, timer: GpuTimer): Promise<number> {
  const buffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
  // Each job reads and writes the same buffer, so they run one after another,
  // like the dependent kernels of a model's forward pass.
  const pipe = pipeline(
    device,
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) {
  data[i] = data[i] + 1u;
}`,
  );
  const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  const ms = await medianOf(() =>
    timer.time((pass) => {
      pass.setPipeline(pipe);
      for (let i = 0; i < DISPATCH_JOBS; i++) {
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(1);
      }
    }),
  );
  buffer.destroy();
  return (ms * 1000) / DISPATCH_JOBS;
}

/**
 * y = W·x with W in 4-bit blocks of 32 (one f16 scale per block), the layout of
 * the runtime's MatMulNBits weights. One workgroup per output row. Returns weight
 * + scale bytes read per second, GB/s.
 */
async function measureMatvec(device: GPUDevice, timer: GpuTimer): Promise<number> {
  const k = MATVEC_K;
  const blocksPerRow = k / 32;
  const rows = Math.min(MATVEC_ROWS, Math.floor(device.limits.maxStorageBufferBindingSize / (k / 2)));
  const weightBytes = rows * blocksPerRow * 16;
  const scaleBytes = rows * blocksPerRow * 2;
  const weights = device.createBuffer({ size: weightBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const scales = device.createBuffer({ size: scaleBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const x = device.createBuffer({ size: k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const y = device.createBuffer({ size: rows * 4, usage: GPUBufferUsage.STORAGE });
  const chunk = new Uint32Array(4 * 1024 * 1024);
  for (let i = 0; i < chunk.length; i++) chunk[i] = (i * 2654435761) >>> 0;
  for (let offset = 0; offset < weightBytes; offset += chunk.byteLength) {
    device.queue.writeBuffer(weights, offset, chunk, 0, Math.min(chunk.length, (weightBytes - offset) / 4));
  }
  // Scales of about 0.01 (f16 0x211f), inputs of 1.
  device.queue.writeBuffer(scales, 0, new Uint32Array(scaleBytes / 4).fill(0x211f211f));
  device.queue.writeBuffer(x, 0, new Float32Array(k).fill(1));
  const pipe = pipeline(
    device,
    /* wgsl */ `
const BLOCKS = ${blocksPerRow}u;
@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> partial: array<f32, 64>;
fn dot8(word: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
  let lo = vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(15u)) - vec4<f32>(8.0);
  let hi = vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(15u)) - vec4<f32>(8.0);
  return dot(lo, a) + dot(hi, b);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x;
  var sum = 0.0;
  for (var b = t; b < BLOCKS; b = b + 64u) {
    let i = row * BLOCKS + b;
    let q = w[i];
    let pair = unpack2x16float(scales[i / 2u]);
    let scale = select(pair.x, pair.y, (i & 1u) == 1u);
    let xi = b * 8u;
    sum = sum + scale * (dot8(q.x, x[xi], x[xi + 1u]) + dot8(q.y, x[xi + 2u], x[xi + 3u])
                       + dot8(q.z, x[xi + 4u], x[xi + 5u]) + dot8(q.w, x[xi + 6u], x[xi + 7u]));
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[row] = partial[0]; }
}`,
  );
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [weights, scales, x, y].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const ms = await fastestOf(() =>
    timer.time((pass) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(rows);
    }),
  );
  for (const buffer of [weights, scales, x, y]) buffer.destroy();
  return (weightBytes + scaleBytes) / (ms / 1000) / 1e9;
}

/**
 * Wall-clock cost per kernel when issued like the runtime does: each job gets a
 * new bind group and a uniform write, and work is submitted every STEP_FLUSH
 * jobs. The fixed cost (one read-back) is removed by timing two batch sizes.
 */
async function measureStep(device: GPUDevice): Promise<number> {
  const data = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniforms = Array.from({ length: STEP_FLUSH }, () =>
    device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
  );
  const pipe = pipeline(
    device,
    /* wgsl */ `
struct Params { add: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) {
  data[i] = data[i] + params.add;
}`,
  );
  const layout = pipe.getBindGroupLayout(0);
  const param = new Uint32Array([1, 0, 0, 0]);
  const run = async (jobs: number) => {
    const start = performance.now();
    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    for (let i = 0; i < jobs; i++) {
      const uniform = uniforms[i % STEP_FLUSH]!;
      device.queue.writeBuffer(uniform, 0, param);
      pass.setPipeline(pipe);
      pass.setBindGroup(0, device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: data } }, { binding: 1, resource: { buffer: uniform } }] }));
      pass.dispatchWorkgroups(1);
      if ((i + 1) % STEP_FLUSH === 0 || i === jobs - 1) {
        pass.end();
        if (i === jobs - 1) encoder.copyBufferToBuffer(data, 0, readback, 0, 256);
        device.queue.submit([encoder.finish()]);
        encoder = device.createCommandEncoder();
        pass = encoder.beginComputePass();
      }
    }
    pass.end();
    await readback.mapAsync(GPUMapMode.READ);
    readback.unmap();
    return performance.now() - start;
  };
  const few = await medianOf(() => run(STEP_FLUSH));
  const many = await medianOf(() => run(STEP_JOBS));
  data.destroy();
  readback.destroy();
  for (const u of uniforms) u.destroy();
  return (Math.max(0, many - few) * 1000) / (STEP_JOBS - STEP_FLUSH);
}

async function measureRoundTrip(device: GPUDevice): Promise<number> {
  const buffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pipe = pipeline(
    device,
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) {
  data[i] = data[i] + 1u;
}`,
  );
  const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  const ms = await medianOf(async () => {
    const start = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, 256);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    readback.unmap();
    return performance.now() - start;
  });
  buffer.destroy();
  readback.destroy();
  return ms;
}

/** Tries to hold `target` bytes of GPU memory in chunks, clearing each so it's really committed. */
async function checkMemory(device: GPUDevice, target: number): Promise<DeviceTestResult['memory']> {
  const buffers: GPUBuffer[] = [];
  let held = 0;
  try {
    while (held < target) {
      const size = Math.min(MEMORY_CHUNK, device.limits.maxBufferSize, target - held);
      device.pushErrorScope('out-of-memory');
      const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(buffer);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const error = await device.popErrorScope();
      buffers.push(buffer);
      if (error) return { status: 'failed', testedBytes: target, failedAtBytes: held };
      held += size;
    }
    return { status: 'ok', testedBytes: target };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}
