// Runs a mixture-of-experts model whose experts don't all fit in memory: shared weights on the GPU, each layer's
// experts loaded as the router picks them (see moe-split.ts, moe-route.ts and info/expert-streaming-plan.md).
//
// The split model's segments are ONNX Runtime sessions of their own. `run` goes segment 0 → QMoE layer 0 → segment
// 1 → … → the last segment, keeping values on the GPU between them. For each layer the router's output is read back
// (tokens × experts, small), routed on the CPU, and the chosen experts copied from the GPU expert cache into slot
// buffers that one shared QMoE session reads. A cache miss reads the expert's bytes from the stored model file (they
// are contiguous in each stacked tensor) straight into its cache buffer.
//
// To the adapter this looks like one InferenceSession (inputMetadata, run, release), so its handling of tokens, states
// and prompts is unchanged.
//
// GPU ordering: ONNX Runtime submits a run's work before `run` returns, and work is queued in submission order, so a
// QMoE run has read its slots before later copies overwrite them. The router read-back waits for everything before it.

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import type { ByteSource } from './external-weights';
import { halfBits, route, slotGroups, type RoutingSettings } from './moe-route';
import { elemTypeOf, intAttributeOf, qmoeModel, type MoeLayer, type MoeSplit, type ValueType } from './moe-split';
import type { getOrt } from './ort-env';

type Ort = ReturnType<typeof getOrt>;

const BUFFER_MAP_READ = 0x1;
const MAP_MODE_READ = 0x1;
const BUFFER_COPY_SRC = 0x4;
const BUFFER_COPY_DST = 0x8;
const BUFFER_STORAGE = 0x80;
/** Each tensor's place in a cached expert's buffer starts at a multiple of this. */
const ALIGN = 256;
/** ONNX type numbers of the expert tensors, as ONNX Runtime tensor types. */
const TENSOR_TYPE: Record<number, 'uint8' | 'float16' | 'float32'> = { 1: 'float32', 2: 'uint8', 10: 'float16' };

export interface StreamOptions {
  ort: Ort;
  device: GPUDevice;
  split: MoeSplit;
  opset: Uint8Array[];
  /** Creates a session for a segment or QMoE model (mounting the model's weight files). */
  createSession(model: Uint8Array<ArrayBuffer>): Promise<InferenceSession>;
  /** The stored files, by the external-data location the rewritten graph gives (expert bytes are read from them). */
  sources: Record<string, ByteSource>;
  /** GPU memory for cached experts. */
  cacheBytes: number;
  /** Most experts fed to one QMoE run (the rest slot included). */
  maxSlots: number;
}

export interface StreamStats {
  hits: number;
  misses: number;
  readBytes: number;
  readMs: number;
  /** Experts the cache holds at most. */
  capacity: number;
  /** Time in each stage, milliseconds: segment runs, router read-backs, cache and slot copies, QMoE runs. */
  segmentMs: number;
  routerMs: number;
  fillMs: number;
  expertMs: number;
}

/** Creates the segments' sessions in order (each one's output types declare the next one's inputs). */
export async function createStreamedSession(options: StreamOptions): Promise<StreamedSession> {
  const { split } = options;
  const types = new Map<string, ValueType>();
  const segments: InferenceSession[] = [];
  try {
    for (const [s, segment] of split.segments.entries()) {
      const session = await options.createSession(segment.build(types));
      segments.push(session);
      for (const meta of session.outputMetadata) if (meta.isTensor) types.set(meta.name, { elem: elemTypeOf(meta.type), shape: meta.shape });
      const layer = split.layers[s];
      if (layer) types.set(layer.output, types.get(layer.x)!);
    }
  } catch (err) {
    for (const session of segments) await session.release().catch(() => {});
    throw err;
  }
  const session = new StreamedSession(options, segments, types);
  // Development: the cache's statistics, for tests (read in the worker).
  if (import.meta.env.DEV) (globalThis as { __quickInfStream?: StreamedSession }).__quickInfStream = session;
  return session;
}

export class StreamedSession {
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];
  readonly stats: StreamStats;
  private readonly cache: ExpertCache;
  private readonly slots: SlotBuffers;
  private readonly qmoe = new Map<string, Promise<InferenceSession>>();
  /** For each value passed between steps: the last step that reads it (segment s is step 2s, QMoE layer s is 2s + 1). */
  private readonly lastUse = new Map<string, number>();
  private readonly settings: RoutingSettings[];
  private staging: GPUBuffer | null = null;

  constructor(
    private readonly options: StreamOptions,
    private readonly segments: InferenceSession[],
    private readonly types: Map<string, ValueType>,
  ) {
    const { split } = options;
    const seen = new Set<string>();
    this.inputMetadata = segments.flatMap((s) => s.inputMetadata.filter((m) => split.modelInputs.includes(m.name) && !seen.has(m.name) && seen.add(m.name)));
    split.segments.forEach((segment, s) => segment.passedIn.forEach((n) => this.lastUse.set(n, Math.max(this.lastUse.get(n) ?? 0, 2 * s))));
    split.layers.forEach((layer, k) => {
      for (const n of [layer.x, layer.router]) this.lastUse.set(n, Math.max(this.lastUse.get(n) ?? 0, 2 * k + 1));
    });
    this.settings = split.layers.map((layer) => {
      if (intAttributeOf(layer, 'use_sparse_mixer', 0)) throw new Error('QMoE with use_sparse_mixer is not supported for streaming');
      return { experts: layer.experts, k: intAttributeOf(layer, 'k', 1), normalize: intAttributeOf(layer, 'normalize_routing_weights', 0) === 1 };
    });
    const largest = split.layers.reduce((a, l) => (expertBytes(l) > expertBytes(a) ? l : a));
    this.slots = new SlotBuffers(options.device, largest, options.maxSlots);
    this.cache = new ExpertCache(options.device, options.sources, Math.max(options.maxSlots + 1, Math.floor(options.cacheBytes / expertBytes(largest))));
    this.stats = this.cache.stats;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const { split, ort } = this.options;
    const values = new Map<string, Tensor>(Object.entries(feeds));
    const made = new Set<string>();
    const outputs: Record<string, Tensor> = {};
    const release = (step: number) => {
      for (const name of made) {
        if ((this.lastUse.get(name) ?? -1) <= step && !(name in outputs)) {
          values.get(name)?.dispose();
          values.delete(name);
          made.delete(name);
        }
      }
    };
    try {
      for (const [s, segment] of split.segments.entries()) {
        const start = performance.now();
        const result = await this.segments[s]!.run(Object.fromEntries(segment.inputs.map((n) => [n, values.get(n)!])));
        this.stats.segmentMs += performance.now() - start;
        for (const [name, value] of Object.entries(result)) {
          if (split.segments[s]!.outputs.includes(name) && !this.lastUse.has(name)) outputs[name] = value as Tensor;
          else {
            values.set(name, value as Tensor);
            made.add(name);
          }
        }
        release(2 * s);
        const layer = split.layers[s];
        if (!layer) break;
        const y = await this.runExperts(s, layer, values.get(layer.x)!, values.get(layer.router)!, ort);
        values.set(layer.output, y);
        made.add(layer.output);
        release(2 * s + 1);
      }
    } catch (err) {
      for (const name of made) values.get(name)?.dispose();
      for (const value of Object.values(outputs)) value.dispose();
      throw err;
    }
    return outputs;
  }

  async release(): Promise<void> {
    for (const session of this.segments) await session.release().catch(() => {});
    for (const session of this.qmoe.values()) await (await session).release().catch(() => {});
    this.cache.destroy();
    this.slots.destroy();
    this.staging?.destroy();
  }

  private async runExperts(index: number, layer: MoeLayer, x: Tensor, router: Tensor, ort: Ort): Promise<Tensor> {
    const settings = this.settings[index]!;
    let start = performance.now();
    const routing = route(await this.readRouter(router), settings);
    this.stats.routerMs += performance.now() - start;
    const groups = slotGroups(routing, this.options.maxSlots);
    const routerType = this.types.get(layer.router)!.elem;
    let sum: Tensor | null = null;
    for (const group of groups) {
      start = performance.now();
      const tensors = this.slots.fill(index, layer, group.experts, group.slots, this.cache, ort);
      this.stats.fillMs += performance.now() - start;
      start = performance.now();
      const routerTensor =
        routerType === 10
          ? new ort.Tensor('float16', halfBits(group.router), [routing.tokens, group.slots])
          : new ort.Tensor('float32', group.router, [routing.tokens, group.slots]);
      const session = await this.qmoeSession(layer, settings.k, sum !== null);
      const feeds: Record<string, Tensor> = { x, router: routerTensor, ...tensors };
      if (sum) feeds.acc = sum;
      const y = (await session.run(feeds)).y as Tensor;
      sum?.dispose();
      for (const t of Object.values(tensors)) t.dispose();
      sum = y;
      this.stats.expertMs += performance.now() - start;
    }
    return sum!;
  }

  /** The router's values on the CPU, through a reused staging buffer (ONNX Runtime's getData makes one each time). */
  private async readRouter(tensor: Tensor): Promise<Float32Array> {
    if (tensor.location !== 'gpu-buffer') return numbers(tensor);
    const bytes = tensor.size * (tensor.type === 'float16' ? 2 : 4);
    const size = Math.ceil(bytes / 4) * 4;
    const { device } = this.options;
    if (!this.staging || this.staging.size < size) {
      this.staging?.destroy();
      this.staging = device.createBuffer({ size: Math.max(size, 4096), usage: BUFFER_MAP_READ | BUFFER_COPY_DST });
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.gpuBuffer as GPUBuffer, 0, this.staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(MAP_MODE_READ, 0, size);
    const copy = this.staging.getMappedRange(0, size).slice(0, bytes);
    this.staging.unmap();
    return tensor.type === 'float16' ? halfToFloat(new Uint16Array(copy)) : new Float32Array(copy);
  }

  private qmoeSession(layer: MoeLayer, k: number, accumulate: boolean): Promise<InferenceSession> {
    const types = { x: this.types.get(layer.x)!.elem, router: this.types.get(layer.router)!.elem };
    const key = `${types.x}:${types.router}:${k}:${accumulate}:${layer.attributes.map((a) => a.join(',')).join('|')}`;
    let session = this.qmoe.get(key);
    if (!session) {
      session = this.options.createSession(qmoeModel(layer, this.options.opset, types, k + 1, accumulate));
      this.qmoe.set(key, session);
    }
    return session;
  }
}

/** Bytes of one expert (all its tensors). */
function expertBytes(layer: MoeLayer): number {
  return layer.tensors.reduce((n, t) => n + align(t.length / layer.experts), 0);
}

const align = (n: number) => Math.ceil(n / ALIGN) * ALIGN;

async function numbers(tensor: Tensor): Promise<Float32Array> {
  const data = (tensor.location === 'gpu-buffer' ? await tensor.getData() : tensor.data) as ArrayLike<number> | Uint16Array;
  if (tensor.type === 'float16' && data instanceof Uint16Array) return halfToFloat(data);
  return Float32Array.from(data as ArrayLike<number>);
}

function halfToFloat(half: Uint16Array): Float32Array {
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) {
    const h = half[i]!;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15);
    out[i] = h & 0x8000 ? -v : v;
  }
  return out;
}

/** Recently used experts on the GPU, least recently used dropped first; misses read from the stored file. */
class ExpertCache {
  readonly stats: StreamStats;
  private readonly entries = new Map<string, { buffer: GPUBuffer; offsets: number[] }>();
  private readonly pinned = new Set<string>();
  private readonly spare: GPUBuffer[] = [];
  private staging = new Uint8Array(0);

  constructor(
    private readonly device: GPUDevice,
    private readonly sources: Record<string, ByteSource>,
    capacity: number,
  ) {
    this.stats = { hits: 0, misses: 0, readBytes: 0, readMs: 0, capacity, segmentMs: 0, routerMs: 0, fillMs: 0, expertMs: 0 };
  }

  /** Experts of a group stay put until `unpin` (a group never needs more than the capacity). */
  get(layerIndex: number, layer: MoeLayer, expert: number): { buffer: GPUBuffer; offsets: number[] } {
    const key = `${layerIndex}:${expert}`;
    let entry = this.entries.get(key);
    if (entry) {
      this.entries.delete(key);
      this.entries.set(key, entry);
      this.stats.hits++;
      this.pinned.add(key);
      return entry;
    }
    this.stats.misses++;
    const size = expertBytes(layer);
    let buffer = this.spare.pop() ?? null;
    if (!buffer && this.entries.size >= this.stats.capacity) {
      for (const [k, e] of this.entries) {
        if (this.pinned.has(k)) continue;
        this.entries.delete(k);
        buffer = e.buffer;
        break;
      }
    }
    if (buffer && buffer.size < size) {
      buffer.destroy();
      buffer = null;
    }
    buffer ??= this.device.createBuffer({ size, usage: BUFFER_STORAGE | BUFFER_COPY_SRC | BUFFER_COPY_DST });
    const start = performance.now();
    const offsets: number[] = [];
    let offset = 0;
    for (const t of layer.tensors) {
      const per = t.length / layer.experts;
      const padded = Math.ceil(per / 4) * 4;
      if (this.staging.byteLength < padded) this.staging = new Uint8Array(padded);
      const target = this.staging.subarray(0, per);
      if (t.location === null) target.set(t.inline!.subarray(expert * per, (expert + 1) * per));
      else {
        const source = this.sources[t.location];
        if (!source) throw new Error(`Weights file ${t.location} isn't open`);
        if (source.readInto) source.readInto(t.offset + expert * per, target);
        else target.set(source.read(t.offset + expert * per, per));
      }
      this.staging.fill(0, per, padded);
      this.device.queue.writeBuffer(buffer, offset, this.staging, 0, padded);
      offsets.push(offset);
      offset += align(per);
    }
    this.stats.readMs += performance.now() - start;
    this.stats.readBytes += size;
    entry = { buffer, offsets };
    this.entries.set(key, entry);
    this.pinned.add(key);
    return entry;
  }

  unpin(): void {
    this.pinned.clear();
  }

  destroy(): void {
    for (const e of this.entries.values()) e.buffer.destroy();
    for (const b of this.spare) b.destroy();
    this.entries.clear();
  }
}

/** One buffer per expert tensor, holding a group's experts side by side as QMoE reads them. */
class SlotBuffers {
  private readonly buffers: GPUBuffer[];

  constructor(
    private readonly device: GPUDevice,
    layer: MoeLayer,
    private readonly maxSlots: number,
  ) {
    this.buffers = layer.tensors.map((t) =>
      device.createBuffer({ size: Math.ceil((t.length / layer.experts) * Math.max(maxSlots, 2)) + 4, usage: BUFFER_STORAGE | BUFFER_COPY_DST }),
    );
  }

  /** Copies `experts` into the first slots and zeroes the rest; returns QMoE's expert inputs over them. */
  fill(layerIndex: number, layer: MoeLayer, experts: number[], slots: number, cache: ExpertCache, ort: Ort): Record<string, Tensor> {
    if (slots > Math.max(this.maxSlots, 2)) throw new Error(`${slots} slots requested, ${this.maxSlots} available`);
    const encoder = this.device.createCommandEncoder();
    experts.forEach((e, i) => {
      const { buffer, offsets } = cache.get(layerIndex, layer, e);
      layer.tensors.forEach((t, j) => {
        const per = t.length / layer.experts;
        encoder.copyBufferToBuffer(buffer, offsets[j]!, this.buffers[j]!, i * per, per);
      });
    });
    layer.tensors.forEach((t, j) => {
      const per = t.length / layer.experts;
      const from = experts.length * per;
      const to = Math.ceil((slots * per) / 4) * 4;
      if (to > from) encoder.clearBuffer(this.buffers[j]!, from, to - from);
    });
    this.device.queue.submit([encoder.finish()]);
    cache.unpin();
    const tensors: Record<string, Tensor> = {};
    layer.tensors.forEach((t, j) => {
      const type = TENSOR_TYPE[t.dataType];
      if (!type) throw new Error(`Unsupported expert weight type ${t.dataType}`);
      tensors[`expert_${t.input}`] = ort.Tensor.fromGpuBuffer(this.buffers[j]!, { dataType: type, dims: [slots, ...t.dims.slice(1)] }) as Tensor;
    });
    return tensors;
  }

  destroy(): void {
    for (const b of this.buffers) b.destroy();
  }
}
