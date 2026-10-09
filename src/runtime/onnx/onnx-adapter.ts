// ONNX Runtime Web (WebGPU) adapter for decoder-only models exported in the
// Hugging Face / Optimum layout: inputs `input_ids`, `attention_mask`,
// `position_ids`, `past_key_values.{i}.{key,value}`; outputs `logits` and
// `present.{i}.{key,value}`.
//
// The KV cache stays on the GPU between steps (present -> past), only logits
// are read back. Generation re-runs the full prompt for each request; reusing
// the cache across turns is a possible later optimisation.

import type { InferenceSession, Tensor } from 'onnxruntime-web/webgpu';
import type { ModelArtifact, ModelSpec } from '../../catalogue/models';
import { getGpuAdapter, getGpuDevice } from '../../device/gpu';
import type { DeviceReport } from '../../device/probe';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { ChatMlFormatter } from '../../tokenizer/chatml';
import { StreamingDecoder } from '../../tokenizer/streaming-decoder';
import { fitTurns } from '../context';
import {
  RuntimeError,
  type AdapterCapabilities,
  type CompatibilityReport,
  type FinishReason,
  type GenerationRequest,
  type GenerationResult,
  type LoadedArtifacts,
  type LoadProgress,
  type RuntimeAdapter,
} from '../contract';
import { applyRepetitionPenalty, createRng, sampleToken } from '../sampling';
import { ATTENTION_BIAS_INPUT, bytesSource, externalizeWeights, lazyBytes, type ByteSource } from './external-weights';
import { prefillChunkTokens } from './limits';
import { getOrt } from './ort-env';


/** What the loaded graph itself expects, read from its input metadata. */
interface GraphShape {
  /** past_key_values.{i}.key / .value input names, in layer order. */
  pastNames: string[];
  kvType: 'float16' | 'float32';
  kvHeads: number;
  headDim: number;
  positionIds: boolean;
  /** Element type of the causal-mask input the rewrite added, if it did (see external-weights.ts). */
  attentionBias: 'float16' | 'float32' | null;
}

interface Loaded {
  model: ModelSpec;
  artifact: ModelArtifact;
  session: InferenceSession;
  graph: GraphShape;
  tokenizer: BpeTokenizer;
  formatter: ChatMlFormatter;
  stopIds: Set<number>;
  prefillChunk: number;
  device: GPUDevice;
  /** Reused buffer the last row of logits is copied into for reading. */
  staging: GPUBuffer | null;
}

/** Load options, for diagnostics experiments; the app uses the defaults. */
export interface LoadOptions {
  /** Extra ONNX Runtime session options. */
  sessionOverrides?: Partial<InferenceSession.SessionOptions>;
  /** Run on this device instead of the shared one. */
  device?: GPUDevice;
}

export interface AdapterEvents {
  /**
   * The GPU device was lost or the WebAssembly runtime crashed. ONNX Runtime
   * cannot recover inside the same worker, so the owner must restart it.
   */
  onFatal?: (error: RuntimeError) => void;
}

/** Failures that browsers can't trigger on demand, simulated for development testing. */
export type AdapterFault = 'device-lost' | 'wasm-crash' | 'hang';

export class OnnxWebGpuAdapter implements RuntimeAdapter {
  private loaded: Loaded | null = null;
  private generating = false;
  private cancelRequested = false;
  private watchedDevice: GPUDevice | null = null;
  private fatal: RuntimeError | null = null;
  private pendingFault: 'wasm-crash' | 'hang' | null = null;
  /**
   * The KV cache left by the last reply and the tokens it covers. The next
   * request in a chat starts with those same tokens (history + reply), so only
   * the new message has to be prefilled.
   */
  private kept: { ids: number[]; cache: KvCache } | null = null;
  private keptLength = 0;

  constructor(private readonly events: AdapterEvents = {}) {}

  describe(): AdapterCapabilities {
    return {
      id: 'onnxruntime-web-webgpu',
      formats: ['onnx'],
      precisions: ['q4f16', 'q4'],
      cancellation: 'next-step',
      streaming: true,
    };
  }

  checkCompatibility(model: ModelSpec, device: DeviceReport): CompatibilityReport {
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
      if (artifact.format !== 'onnx') continue;
      const missing = artifact.requiredGpuFeatures.filter((f) => !features.has(f));
      if (missing.length === 0) return { compatible: true, artifact, reasons };
      reasons.push(`${artifact.precision} needs GPU feature(s): ${missing.join(', ')}`);
    }
    return { compatible: false, reasons };
  }

  async load(
    model: ModelSpec,
    artifact: ModelArtifact,
    files: LoadedArtifacts,
    onProgress: (p: LoadProgress) => void,
    options: LoadOptions = {},
  ): Promise<{ sessionMs: number; validateMs: number }> {
    const { sessionOverrides = {} } = options;
    if (this.fatal) throw this.fatal;
    await this.unload();
    const ort = getOrt();

    let tokenizer: BpeTokenizer;
    try {
      tokenizer = BpeTokenizer.fromJSON(files.tokenizerJson);
    } catch (err) {
      throw new RuntimeError('load-failed', `Tokenizer could not be loaded: ${message(err)}`);
    }

    onProgress({ phase: 'loading' });
    const sessionStart = performance.now();
    // With logits left on the GPU, a prefill step copies back one row instead of the
    // whole [tokens × vocabulary] block (50+ MB for a 256-token chunk with Qwen's vocabulary).
    const outputLocations: Record<string, 'gpu-buffer' | 'cpu'> = { logits: 'gpu-buffer' };
    for (let i = 0; i < model.architecture.layers; i++) {
      outputLocations[`present.${i}.key`] = 'gpu-buffer';
      outputLocations[`present.${i}.value`] = 'gpu-buffer';
    }
    let device: GPUDevice;
    try {
      device = options.device ?? (await getGpuDevice());
      // ONNX Runtime requests (and then drops) an adapter of its own unless given one.
      const adapter = await getGpuAdapter();
      if (adapter) ort.env.webgpu.adapter = adapter;
    } catch (err) {
      throw new RuntimeError('load-failed', `The GPU could not be opened: ${message(err)}`, true);
    }
    this.watchDevice(device);
    let session: InferenceSession;
    try {
      const input = modelInput(files.graph);
      session = await ort.InferenceSession.create(input.bytes, {
        externalData: input.externalData,
        // Run on the device the probe checked, not one ONNX Runtime picks itself.
        executionProviders: [{ name: 'webgpu', device }],
        graphOptimizationLevel: 'all',
        preferredOutputLocation: outputLocations,
        enableCpuMemArena: false,
        ...sessionOverrides,
      });
    } catch (err) {
      if (isRuntimeFetchFailure(err)) {
        // ONNX Runtime remembers a failed start-up, so only a fresh worker can retry (the worker marks it).
        throw new RuntimeError('runtime-files-unavailable', `ONNX Runtime could not start: ${message(err)}`, true);
      }
      throw new RuntimeError('load-failed', `The model could not be loaded on the GPU: ${message(err)}`, true);
    }
    const sessionMs = performance.now() - sessionStart;

    let graph: GraphShape;
    let formatter: ChatMlFormatter;
    let stopIds: Set<number>;
    try {
      graph = readGraphShape(session, model);
      formatter = new ChatMlFormatter(tokenizer, {
        defaultSystemPrompt: model.promptFormat.defaultSystemPrompt,
        disableThinking: model.promptFormat.disableThinking,
      });
      stopIds = new Set(model.promptFormat.stopTokens.map((t) => tokenizer.specialTokenId(t)));
    } catch (err) {
      await session.release().catch(() => {});
      throw new RuntimeError('load-failed', `The model files don't match the catalogue: ${message(err)}`);
    }
    const prefillChunk = prefillChunkTokens(model.architecture.vocabSize);
    this.loaded = {
      model,
      artifact,
      session,
      graph,
      tokenizer,
      formatter,
      stopIds,
      prefillChunk,
      device,
      staging: null,
    };

    // Validate with one real forward pass before reporting ready.
    onProgress({ phase: 'validating' });
    const validateStart = performance.now();
    try {
      const cache = this.emptyCache();
      const { logits, cache: next } = await this.forward([formatter.endOfTurnId], cache, 0);
      disposeCache(next);
      if (logits.length !== model.architecture.vocabSize || !logits.every(Number.isFinite)) {
        throw new Error('the output contained invalid values');
      }
    } catch (err) {
      await this.unload();
      throw new RuntimeError('validation-failed', `The model loaded but a test run failed: ${message(err)}`);
    }
    return { sessionMs, validateMs: performance.now() - validateStart };
  }

  async generate(request: GenerationRequest, onDelta: (text: string) => void): Promise<GenerationResult> {
    const loaded = this.requireLoaded();
    if (this.generating) throw new RuntimeError('busy', 'A response is already being generated.');
    this.generating = true;
    this.cancelRequested = false;

    const { options } = request;
    const start = performance.now();
    let cache = this.emptyCache();
    // Tokens `cache` covers; null once it can't be kept (cancelled mid-prefill, failed).
    let fed: number[] | null = null;
    try {
      const { formatter, tokenizer, stopIds } = loaded;
      const promptBudget = options.contextTokens - Math.min(options.maxNewTokens, options.contextTokens / 2);
      const generationPrompt = formatter.generationPromptLength;
      const defaultSystem = loaded.model.promptFormat.defaultSystemPrompt;
      const implicitSystem =
        request.turns[0]?.role !== 'system' && defaultSystem !== undefined
          ? formatter.turnLength({ role: 'system', content: defaultSystem })
          : 0;
      const plan = fitTurns(request.turns, promptBudget, (t) => formatter.turnLength(t), generationPrompt + implicitSystem);
      const promptIds = formatter.format(plan.turns);
      const maxNew = Math.min(options.maxNewTokens, options.contextTokens - promptIds.length);

      const random = createRng(options.seed ?? (Math.random() * 2 ** 32) >>> 0);
      const decoder = new StreamingDecoder(tokenizer);
      const generated: number[] = [];
      let text = '';
      let finishReason: FinishReason = 'length';

      const reused = this.takeKept(promptIds);
      if (reused) {
        disposeCache(cache);
        cache = reused;
      }
      const reusedTokens = reused ? this.keptLength : 0;
      const prefilled = await this.prefill(promptIds.slice(reusedTokens), cache, reusedTokens);
      cache = prefilled.cache;
      fed = prefilled.cancelled ? null : [...promptIds];
      let logits = prefilled.logits;
      const prefillMs = performance.now() - start;
      let firstTokenAt = 0;

      if (prefilled.cancelled) finishReason = 'cancelled';
      else {
        for (;;) {
          if (this.cancelRequested) {
            finishReason = 'cancelled';
            break;
          }
          if (options.repetitionPenalty !== 1) {
            const window = options.repetitionWindow;
            const recent = generated.length >= window ? generated.slice(-window) : [...promptIds.slice(-(window - generated.length)), ...generated];
            applyRepetitionPenalty(logits, recent, options.repetitionPenalty);
          }
          const token = sampleToken(logits, options, random);
          if (generated.length === 0) firstTokenAt = performance.now();
          if (stopIds.has(token)) {
            finishReason = 'stop';
            break;
          }
          generated.push(token);
          const delta = decoder.push(token);
          if (delta) {
            text += delta;
            onDelta(delta);
          }
          if (generated.length >= maxNew) {
            finishReason = 'length';
            break;
          }
          const step = await this.forward([token], cache, promptIds.length + generated.length - 1);
          cache = step.cache;
          fed?.push(token);
          logits = step.logits;
        }
      }
      const tail = decoder.flush();
      if (tail) {
        text += tail;
        onDelta(tail);
      }

      const end = performance.now();
      const decodeMs = firstTokenAt ? end - firstTokenAt : 0;
      return {
        text,
        tokenIds: generated,
        finishReason,
        stats: {
          promptTokens: promptIds.length,
          reusedPromptTokens: reusedTokens,
          droppedTurns: plan.droppedTurns,
          generatedTokens: generated.length,
          prefillMs,
          timeToFirstTokenMs: firstTokenAt ? firstTokenAt - start : 0,
          decodeMs,
          decodeTokensPerSecond: generated.length > 1 && decodeMs > 0 ? ((generated.length - 1) * 1000) / decodeMs : 0,
          totalMs: end - start,
        },
      };
    } catch (err) {
      fed = null;
      if (this.fatal) throw this.fatal;
      if (err instanceof RuntimeError) throw err;
      if (isRuntimeAbort(err)) throw this.markFatal('runtime-crashed', `The inference runtime crashed: ${message(err)}`);
      throw new RuntimeError('generation-failed', `Generation failed: ${message(err)}`, true);
    } finally {
      if (fed && this.loaded === loaded) this.kept = { ids: fed, cache };
      else disposeCache(cache);
      this.generating = false;
      this.cancelRequested = false;
    }
  }

  cancel(): void {
    if (this.generating) this.cancelRequested = true;
  }

  reset(): void {
    this.dropKept();
  }

  async unload(): Promise<void> {
    this.dropKept();
    const loaded = this.loaded;
    this.loaded = null;
    if (loaded) {
      loaded.staging?.destroy();
      await loaded.session.release();
    }
  }

  get tokenizer(): BpeTokenizer {
    return this.requireLoaded().tokenizer;
  }

  /**
   * Development check: feeds `promptIds` then `continuation` one token at a time
   * and passes the logits before each continuation token to `inspect`, so
   * outputs can be compared step by step with a reference without sampling
   * divergence.
   */
  async scoreContinuation(
    promptIds: number[],
    continuation: number[],
    inspect: (step: number, logits: Float32Array) => void,
  ): Promise<void> {
    this.requireLoaded();
    let cache = this.emptyCache();
    try {
      const prefilled = await this.prefill(promptIds, cache);
      cache = prefilled.cache;
      let logits = prefilled.logits;
      for (let i = 0; i < continuation.length; i++) {
        inspect(i, logits);
        if (i === continuation.length - 1) break;
        const step = await this.forward([continuation[i]!], cache, promptIds.length + i);
        cache = step.cache;
        logits = step.logits;
      }
    } finally {
      disposeCache(cache);
    }
  }

  /** Subscribes once to loss of the GPU device the runtime uses. */
  private watchDevice(device: GPUDevice): void {
    if (device === this.watchedDevice) return;
    this.watchedDevice = device;
    void device.lost.then((info) => {
      if (info.reason === 'destroyed') return; // released on purpose
      this.onDeviceLost(info.message);
    });
  }

  private onDeviceLost(detail: string): void {
    this.markFatal('device-lost', `The GPU stopped responding${detail ? `: ${detail}` : '.'}`);
  }

  /**
   * Development only. `device-lost` reports a lost device and destroys it;
   * `wasm-crash` makes the next forward pass abort like a WebAssembly trap;
   * `hang` makes the next forward pass never finish.
   */
  simulateFault(fault: AdapterFault): void {
    if (fault === 'device-lost') {
      this.onDeviceLost('simulated device loss');
      this.watchedDevice?.destroy();
      return;
    }
    this.pendingFault = fault;
  }

  private markFatal(code: 'device-lost' | 'runtime-crashed', text: string): RuntimeError {
    if (this.fatal) return this.fatal;
    const error = new RuntimeError(code, text, true);
    this.fatal = error;
    this.cancelRequested = true;
    this.kept = null;
    const loaded = this.loaded;
    this.loaded = null;
    // Best effort: the session's GPU resources may already be gone.
    loaded?.session.release().catch(() => {});
    this.events.onFatal?.(error);
    return error;
  }

  /**
   * The kept KV cache, cut to the part `promptIds` starts with, leaving at least
   * one new token to prefill (which yields the next logits). The reply usually
   * re-tokenizes identically, so the whole cache is reused; where it doesn't (or
   * the template renders past replies differently, like Qwen3 without its empty
   * thinking block) the matching start is still reused. Sets `keptLength`.
   */
  private takeKept(promptIds: number[]): KvCache | null {
    const kept = this.kept;
    this.kept = null;
    this.keptLength = 0;
    if (!kept) return null;
    let common = 0;
    const limit = Math.min(kept.ids.length, promptIds.length - 1);
    while (common < limit && kept.ids[common] === promptIds[common]) common++;
    if (common === kept.ids.length) {
      this.keptLength = common;
      return kept.cache;
    }
    if (common >= MIN_REUSED_TOKENS && kept.cache.every((t) => t.location === 'gpu-buffer')) {
      const cut = this.truncateCache(kept.cache, common);
      disposeCache(kept.cache);
      this.keptLength = common;
      return cut;
    }
    disposeCache(kept.cache);
    return null;
  }

  /** Copies the first `length` positions of each [1, heads, positions, headDim] cache tensor into new GPU buffers. */
  private truncateCache(cache: KvCache, length: number): KvCache {
    const { device } = this.requireLoaded();
    const ort = getOrt();
    const encoder = device.createCommandEncoder();
    const cut = cache.map((tensor) => {
      const [batch, heads, positions, headDim] = tensor.dims as [number, number, number, number];
      const elementBytes = tensor.type === 'float16' ? 2 : 4;
      const from = positions * headDim * elementBytes;
      const to = length * headDim * elementBytes;
      const buffer = device.createBuffer({
        size: Math.max(4, Math.ceil((batch * heads * to) / 4) * 4),
        usage: BUFFER_STORAGE | BUFFER_COPY_SRC | BUFFER_COPY_DST,
      });
      for (let h = 0; h < batch * heads; h++) encoder.copyBufferToBuffer(tensor.gpuBuffer as GPUBuffer, h * from, buffer, h * to, to);
      return ort.Tensor.fromGpuBuffer(buffer, {
        dataType: tensor.type as 'float16' | 'float32',
        dims: [batch, heads, length, headDim],
        dispose: () => buffer.destroy(),
      });
    });
    device.queue.submit([encoder.finish()]);
    return cut;
  }

  private dropKept(): void {
    if (this.kept) disposeCache(this.kept.cache);
    this.kept = null;
  }

  private requireLoaded(): Loaded {
    if (!this.loaded) throw new RuntimeError('not-loaded', 'No model is loaded.');
    return this.loaded;
  }

  /** Feeds `ids` after the `pastLength` tokens already in `cache`. */
  private async prefill(ids: number[], cache: KvCache, pastLength = 0) {
    const chunk = this.requireLoaded().prefillChunk;
    let logits = new Float32Array(0);
    for (let offset = 0; offset < ids.length; offset += chunk) {
      if (this.cancelRequested) return { logits, cache, cancelled: true };
      // Only the last chunk's logits are used.
      const step = await this.forward(ids.slice(offset, offset + chunk), cache, pastLength + offset, offset + chunk >= ids.length);
      cache = step.cache;
      logits = step.logits;
    }
    return { logits, cache, cancelled: false };
  }

  /** One forward pass. Consumes `cache` (disposing its GPU buffers) and returns the next one. */
  private async forward(
    ids: number[],
    cache: KvCache,
    pastLength: number,
    wantLogits = true,
  ): Promise<{ logits: Float32Array<ArrayBuffer>; cache: KvCache }> {
    const { session, model, graph } = this.requireLoaded();
    const ort = getOrt();
    const n = ids.length;
    const total = pastLength + n;

    const feeds: Record<string, Tensor> = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]),
    };
    if (graph.positionIds) {
      feeds.position_ids = new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, i) => BigInt(pastLength + i)), [1, n]);
    }
    if (graph.attentionBias) feeds[ATTENTION_BIAS_INPUT] = causalMask(ort, graph.attentionBias, n, pastLength);
    graph.pastNames.forEach((name, i) => {
      feeds[name] = cache[i]!;
    });

    let outputs: InferenceSession.OnnxValueMapType;
    try {
      if (this.pendingFault) {
        const fault = this.pendingFault;
        this.pendingFault = null;
        if (fault === 'wasm-crash') throw new WebAssembly.RuntimeError('unreachable executed (simulated)');
        await new Promise<never>(() => {});
      }
      outputs = await session.run(feeds);
    } finally {
      disposeCache(cache);
    }

    const next: KvCache = graph.pastNames.map((name) => outputs[name.replace('past_key_values.', 'present.')] as Tensor);
    const logitsTensor = outputs.logits as Tensor;
    try {
      if (!wantLogits) return { logits: new Float32Array(0), cache: next };
      const vocab = model.architecture.vocabSize;
      const last =
        logitsTensor.location === 'gpu-buffer'
          ? await this.readLastRow(logitsTensor, n, vocab)
          : (logitsTensor.data as Float32Array | Uint16Array).subarray((n - 1) * vocab, n * vocab);
      const logits = logitsTensor.type === 'float16' ? halfToFloat(last as Uint16Array) : new Float32Array(last as Float32Array);
      return { logits, cache: next };
    } finally {
      logitsTensor.dispose();
    }
  }

  /** Copies the last row of a [1, n, vocab] logits tensor from the GPU. ONNX Runtime shares our device, so its work is queued ahead of this copy. */
  private async readLastRow(tensor: Tensor, n: number, vocab: number): Promise<Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>> {
    const loaded = this.requireLoaded();
    const half = tensor.type === 'float16';
    const rowBytes = vocab * (half ? 2 : 4);
    if (!loaded.staging || loaded.staging.size < rowBytes) {
      loaded.staging?.destroy();
      loaded.staging = loaded.device.createBuffer({ size: Math.ceil(rowBytes / 4) * 4, usage: BUFFER_MAP_READ | BUFFER_COPY_DST });
    }
    const staging = loaded.staging;
    const encoder = loaded.device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.gpuBuffer as GPUBuffer, (n - 1) * rowBytes, staging, 0, Math.ceil(rowBytes / 4) * 4);
    loaded.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE_READ);
    const bytes = staging.getMappedRange(0, Math.ceil(rowBytes / 4) * 4).slice(0, rowBytes);
    staging.unmap();
    return half ? new Uint16Array(bytes) : new Float32Array(bytes);
  }

  private emptyCache(): KvCache {
    const { graph } = this.requireLoaded();
    const ort = getOrt();
    const dims = [1, graph.kvHeads, 0, graph.headDim];
    const make = () =>
      graph.kvType === 'float16'
        ? new ort.Tensor('float16', new Uint16Array(0), dims)
        : new ort.Tensor('float32', new Float32Array(0), dims);
    return graph.pastNames.map(make);
  }
}

// WebGPU flag values (the DOM typings declare the types but not these constants).
const BUFFER_MAP_READ = 0x1;
const BUFFER_COPY_SRC = 0x4;
const BUFFER_COPY_DST = 0x8;
const BUFFER_STORAGE = 0x80;
const MAP_MODE_READ = 0x1;

/** Below this many matching tokens, cutting down the kept cache isn't worth it. */
const MIN_REUSED_TOKENS = 4;

/** Name the rewritten model uses for its external data; mounted as the original file. */
const WEIGHTS_FILE = 'weights.bin';

/**
 * What to hand ONNX Runtime: the model's structure with its weights pointing into
 * the original file, which ONNX Runtime then reads one tensor at a time (straight
 * to the GPU), instead of copying and parsing the whole file in WebAssembly
 * memory. Falls back to the plain file if it can't be rewritten.
 */
function modelInput(graph: Uint8Array<ArrayBuffer> | ByteSource): {
  bytes: Uint8Array<ArrayBuffer>;
  externalData?: { path: string; data: Uint8Array<ArrayBuffer> }[];
} {
  const source = graph instanceof Uint8Array ? bytesSource(graph) : graph;
  try {
    const rewritten = externalizeWeights(source, WEIGHTS_FILE);
    return { bytes: rewritten.graph, externalData: [{ path: WEIGHTS_FILE, data: lazyBytes(source) }] };
  } catch (err) {
    console.warn('Loading the whole model file instead of reading weights on demand:', err);
    return { bytes: graph instanceof Uint8Array ? graph : source.read(0, source.size) };
  }
}

type KvCache = Tensor[];

/**
 * Reads the KV-cache layout from the graph's inputs and checks it against the
 * catalogue entry (a mismatch means the wrong file or a wrong catalogue entry).
 */
function readGraphShape(session: InferenceSession, model: ModelSpec): GraphShape {
  const inputs = new Map(session.inputMetadata.map((m) => [m.name, m]));
  const pastNames = [...inputs.keys()]
    .filter((name) => name.startsWith('past_key_values.'))
    .sort((a, b) => layerOf(a) - layerOf(b) || Number(a.endsWith('.value')) - Number(b.endsWith('.value')));
  const first = inputs.get(pastNames[0] ?? '');
  if (!first || !first.isTensor) throw new Error('the graph has no KV-cache inputs');
  const kvHeads = Number(first.shape[1]);
  const headDim = Number(first.shape[3]);
  const kvType = first.type === 'float16' ? 'float16' : 'float32';
  const { layers, kvHeads: expectedHeads, headDim: expectedDim } = model.architecture;
  if (pastNames.length !== 2 * layers || kvHeads !== expectedHeads || headDim !== expectedDim) {
    throw new Error(
      `expected ${layers} layers × ${expectedHeads} KV heads × ${expectedDim}, found ${pastNames.length / 2} × ${kvHeads} × ${headDim}`,
    );
  }
  const bias = inputs.get(ATTENTION_BIAS_INPUT);
  const attentionBias = bias?.isTensor ? (bias.type === 'float16' ? 'float16' : 'float32') : null;
  return { pastNames, kvType, kvHeads, headDim, positionIds: inputs.has('position_ids'), attentionBias };
}

/**
 * [1, 1, n, past + n] attention bias: new token i sees every cached position and
 * new positions up to itself (0), and not later ones (the export's own fill value).
 */
function causalMask(ort: ReturnType<typeof getOrt>, type: 'float16' | 'float32', n: number, pastLength: number): Tensor {
  const total = pastLength + n;
  const dims = [1, 1, n, total];
  if (type === 'float16') {
    const data = new Uint16Array(n * total);
    // -65504, the most negative finite half; 0 is all-zero bits.
    for (let i = 0; i < n; i++) data.fill(0xfbff, i * total + pastLength + i + 1, (i + 1) * total);
    return new ort.Tensor('float16', data, dims);
  }
  const data = new Float32Array(n * total);
  for (let i = 0; i < n; i++) data.fill(-3.4028234663852886e38, i * total + pastLength + i + 1, (i + 1) * total);
  return new ort.Tensor('float32', data, dims);
}

function layerOf(name: string): number {
  return Number(name.split('.')[1]);
}

/** IEEE 754 half-precision to single-precision. */
function halfToFloat(half: Uint16Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) {
    const h = half[i]!;
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    out[i] =
      exponent === 0
        ? sign * 2 ** -14 * (fraction / 1024)
        : exponent === 31
          ? fraction
            ? NaN
            : sign * Infinity
          : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return out;
}


/**
 * Frees the cache's GPU buffers. Best effort: after a device loss or crash,
 * disposing throws, and that must not replace the error being reported.
 */
function disposeCache(cache: KvCache): void {
  for (const t of cache) {
    try {
      if (t.location === 'gpu-buffer') t.dispose();
    } catch {
      // The buffer is unusable either way.
    }
  }
  cache.length = 0;
}

/**
 * ONNX Runtime fetches its WebAssembly binary from the app's own server on first
 * use. If that fails (server down, offline), it reports "no available backend"
 * with the fetch error inside; that is not a GPU problem.
 */
function isRuntimeFetchFailure(err: unknown): boolean {
  const text = message(err);
  return /Failed to load|Failed to fetch|NetworkError/i.test(text) && /\.wasm|backend/i.test(text);
}

/**
 * WebAssembly traps and Emscripten aborts leave ONNX Runtime's memory in an
 * unknown state; nothing after them can be trusted. (Heuristic: matched on the
 * messages these failures produce.)
 */
function isRuntimeAbort(err: unknown): boolean {
  if (typeof WebAssembly !== 'undefined' && err instanceof WebAssembly.RuntimeError) return true;
  return /\bAborted\(|memory access out of bounds|unreachable executed/i.test(message(err));
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

