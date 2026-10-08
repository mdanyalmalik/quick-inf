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
import { getOrt } from './ort-env';

/** Prompt tokens per forward pass during prefill; bounds the logits buffer (chunk × vocab × 4 bytes). */
const PREFILL_CHUNK = 256;

interface Loaded {
  model: ModelSpec;
  artifact: ModelArtifact;
  session: InferenceSession;
  tokenizer: BpeTokenizer;
  formatter: ChatMlFormatter;
  stopIds: Set<number>;
}

export class OnnxWebGpuAdapter implements RuntimeAdapter {
  private loaded: Loaded | null = null;
  private generating = false;
  private cancelRequested = false;

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
  ): Promise<{ sessionMs: number; validateMs: number }> {
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
    const outputLocations: Record<string, 'gpu-buffer' | 'cpu'> = { logits: 'cpu' };
    for (let i = 0; i < model.architecture.layers; i++) {
      outputLocations[`present.${i}.key`] = 'gpu-buffer';
      outputLocations[`present.${i}.value`] = 'gpu-buffer';
    }
    let session: InferenceSession;
    try {
      session = await ort.InferenceSession.create(files.graph, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        preferredOutputLocation: outputLocations,
      });
    } catch (err) {
      throw new RuntimeError('load-failed', `The model could not be loaded on the GPU: ${message(err)}`, true);
    }
    const sessionMs = performance.now() - sessionStart;

    const formatter = new ChatMlFormatter(tokenizer, { defaultSystemPrompt: model.promptFormat.defaultSystemPrompt });
    const stopIds = new Set(model.promptFormat.stopTokens.map((t) => tokenizer.specialTokenId(t)));
    this.loaded = { model, artifact, session, tokenizer, formatter, stopIds };

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
    try {
      const { formatter, tokenizer, stopIds } = loaded;
      const promptBudget = options.contextTokens - Math.min(options.maxNewTokens, options.contextTokens / 2);
      const generationPrompt = 1 + tokenizer.encode('assistant\n').length;
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

      const prefilled = await this.prefill(promptIds, cache);
      cache = prefilled.cache;
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
      if (err instanceof RuntimeError) throw err;
      throw new RuntimeError('generation-failed', `Generation failed: ${message(err)}`, true);
    } finally {
      disposeCache(cache);
      this.generating = false;
      this.cancelRequested = false;
    }
  }

  cancel(): void {
    if (this.generating) this.cancelRequested = true;
  }

  reset(): void {
    // No conversation state is kept between requests yet.
  }

  async unload(): Promise<void> {
    const loaded = this.loaded;
    this.loaded = null;
    if (loaded) await loaded.session.release();
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

  private requireLoaded(): Loaded {
    if (!this.loaded) throw new RuntimeError('not-loaded', 'No model is loaded.');
    return this.loaded;
  }

  private async prefill(ids: number[], cache: KvCache) {
    let logits = new Float32Array(0);
    for (let offset = 0; offset < ids.length; offset += PREFILL_CHUNK) {
      if (this.cancelRequested) return { logits, cache, cancelled: true };
      const step = await this.forward(ids.slice(offset, offset + PREFILL_CHUNK), cache, offset);
      cache = step.cache;
      logits = step.logits;
    }
    return { logits, cache, cancelled: false };
  }

  /** One forward pass. Consumes `cache` (disposing its GPU buffers) and returns the next one. */
  private async forward(ids: number[], cache: KvCache, pastLength: number): Promise<{ logits: Float32Array<ArrayBuffer>; cache: KvCache }> {
    const { session, model } = this.requireLoaded();
    const ort = getOrt();
    const n = ids.length;
    const total = pastLength + n;

    const feeds: Record<string, Tensor> = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]),
      position_ids: new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, i) => BigInt(pastLength + i)), [1, n]),
    };
    for (let i = 0; i < model.architecture.layers; i++) {
      feeds[`past_key_values.${i}.key`] = cache[2 * i]!;
      feeds[`past_key_values.${i}.value`] = cache[2 * i + 1]!;
    }

    let outputs: InferenceSession.OnnxValueMapType;
    try {
      outputs = await session.run(feeds);
    } finally {
      disposeCache(cache);
    }

    const next: KvCache = [];
    for (let i = 0; i < model.architecture.layers; i++) {
      next.push(outputs[`present.${i}.key`] as Tensor, outputs[`present.${i}.value`] as Tensor);
    }
    const logitsTensor = outputs.logits as Tensor;
    const vocab = model.architecture.vocabSize;
    const all = logitsTensor.data as Float32Array;
    const logits = new Float32Array(all.subarray((n - 1) * vocab, n * vocab));
    logitsTensor.dispose();
    return { logits, cache: next };
  }

  private emptyCache(): KvCache {
    const { model, artifact } = this.requireLoaded();
    const ort = getOrt();
    const { layers, kvHeads, headDim } = model.architecture;
    const dims = [1, kvHeads, 0, headDim];
    const make = () =>
      artifact.kvCacheType === 'float16'
        ? new ort.Tensor('float16', new Uint16Array(0), dims)
        : new ort.Tensor('float32', new Float32Array(0), dims);
    return Array.from({ length: 2 * layers }, make);
  }
}

type KvCache = Tensor[];

function disposeCache(cache: KvCache): void {
  for (const t of cache) {
    if (t.location === 'gpu-buffer') t.dispose();
  }
  cache.length = 0;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

