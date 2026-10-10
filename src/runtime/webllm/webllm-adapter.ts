// WebLLM engine: MLC's compiled models (per-model WebGPU kernels), which on the
// Mac use 2–3× less memory than the ONNX engine and reply faster (see
// info/devlog.md). Lives in the runtime worker, imported only when a WebLLM
// model loads (the library is 6.6 MB).
//
// WebLLM downloads and saves its own files (OPFS backend). Each build is pinned
// in the catalogue: the weight repository commit, the compiled library by
// commit, and SRI hashes WebLLM checks for the config, tokenizer and library.
// The worker checks weight shards against their MD5s after a download
// (src/store/webllm-store.ts).
//
// Conversations: WebLLM keeps the cache of the previous reply and reuses it
// when a request's messages are exactly the previous ones plus a new message.
// So the adapter sends back each reply exactly as WebLLM produced it; text it
// cleans for display (Qwen3's empty thinking block) is mapped back.

import {
  MLCEngine,
  type ChatCompletionMessageParam,
  type InitProgressReport,
  type ModelRecord,
} from '@mlc-ai/web-llm';
import { artifactBytes, type ModelArtifact, type ModelSpec, type WebLlmArtifact } from '../../catalogue/models';
import { fitTurns } from '../context';
import {
  RuntimeError,
  type ChatTurn,
  type FinishReason,
  type GenerationRequest,
  type GenerationResult,
  type LoadedArtifacts,
  type LoadProgress,
  type RuntimeAdapter,
} from '../contract';
import type { AdapterEvents } from '../onnx/onnx-adapter';
import { extendsByOne, stripThinking } from './text';

/** Template tokens around each message, and around the whole prompt (system turn, reply header): generous upper bounds. */
const TOKENS_PER_TURN = 10;
const TOKENS_FIXED = 48;

interface Loaded {
  model: ModelSpec;
  artifact: WebLlmArtifact;
  engine: MLCEngine;
}

/** What WebLLM's cache holds after the last reply: the messages it saw and its reply, and their token count. */
interface Conversation {
  messages: ChatCompletionMessageParam[];
  tokens: number;
}

export class WebLlmAdapter implements RuntimeAdapter {
  private loaded: Loaded | null = null;
  /** The engine of a load in progress, so `unload()` can abort it. */
  private loading: MLCEngine | null = null;
  private generating = false;
  private cancelRequested = false;
  private conversation: Conversation | null = null;
  /** Replies as shown (cleaned) → as WebLLM produced them. */
  private readonly rawReplies = new Map<string, string>();

  constructor(private readonly events: AdapterEvents = {}) {}

  async load(
    model: ModelSpec,
    artifact: ModelArtifact,
    _files: LoadedArtifacts | null,
    onProgress: (p: LoadProgress) => void,
    options: { signal?: AbortSignal } = {},
  ): Promise<{ sessionMs: number; validateMs: number }> {
    if (artifact.engine !== 'webllm') throw new RuntimeError('load-failed', 'Not a WebLLM model file.');
    await this.unload();
    const { build } = artifact;
    const contextTokens = Math.min(model.generation.contextTokens, build.contextWindow);
    const record: ModelRecord = {
      model: `https://huggingface.co/${build.repo}/resolve/${build.revision}/`,
      model_id: artifact.modelId,
      model_lib: build.modelLib.url,
      vram_required_MB: build.vramMB,
      required_features: artifact.requiredGpuFeatures,
      // A fixed context window. WebLLM allows only one of the two, and its sliding window would
      // cap every layer's memory at the window (why Gemma 3 isn't in the catalogue).
      overrides: { context_window_size: contextTokens, sliding_window_size: -1 },
      integrity: { config: build.integrity.config, model_lib: build.modelLib.integrity, tokenizer: build.integrity.tokenizer },
    };
    const total = artifactBytes(model, artifact);
    const shardBytes = build.weightBytes;
    const engine = new MLCEngine({
      appConfig: { model_list: [record], cacheBackend: 'opfs', opfsAccessMode: 'auto' },
      logLevel: 'WARN',
      initProgressCallback: (report) => onProgress(progressOf(report, total, shardBytes)),
    });
    this.loading = engine;
    const abort = () => void engine.unload();
    options.signal?.addEventListener('abort', abort);

    const start = performance.now();
    try {
      await engine.reload(artifact.modelId, { context_window_size: contextTokens, sliding_window_size: -1 });
      // An aborted reload returns without an error.
      if (options.signal?.aborted || this.loading !== engine) throw new RuntimeError('cancelled', 'Loading was cancelled.');
    } catch (err) {
      await engine.unload().catch(() => {});
      // Cancelling unloads the engine mid-download, which surfaces as an abort or a fetch error.
      if (options.signal?.aborted) throw new RuntimeError('cancelled', 'Loading was cancelled.');
      throw err instanceof RuntimeError ? err : loadError(err);
    } finally {
      options.signal?.removeEventListener('abort', abort);
      if (this.loading === engine) this.loading = null;
    }
    const sessionMs = performance.now() - start;
    this.loaded = { model, artifact, engine };

    // One short reply before reporting ready.
    onProgress({ phase: 'validating' });
    const validateStart = performance.now();
    try {
      const reply = await engine.chat.completions.create({ messages: [{ role: 'user', content: 'Hi' }], max_tokens: 1, temperature: 0 });
      if (!reply.choices[0]) throw new Error('no reply');
      await engine.resetChat();
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
    const { engine, model } = loaded;
    const start = performance.now();
    try {
      const contextTokens = Math.min(options.contextTokens, loaded.artifact.build.contextWindow);
      const promptBudget = contextTokens - Math.min(options.maxNewTokens, contextTokens / 2);
      const count = (text: string) => this.countTokens(text);
      const plan = fitTurns(request.turns, promptBudget, (t) => count(t.content) + TOKENS_PER_TURN, TOKENS_FIXED);
      const messages = plan.turns.map((turn) => this.toMessage(turn));

      // WebLLM reuses its cache when these messages extend the previous ones exactly.
      const previous = this.conversation;
      const reused = previous && extendsByOne(messages, previous.messages) ? previous.tokens : 0;
      const estimated = plan.turns.reduce((n, t) => n + count(t.content) + TOKENS_PER_TURN, TOKENS_FIXED);
      const maxNew = Math.max(1, Math.min(options.maxNewTokens, contextTokens - estimated));

      const stream = await engine.chat.completions.create({
        messages,
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: maxNew,
        temperature: options.temperature,
        top_p: options.topP,
        repetition_penalty: options.repetitionPenalty,
        seed: options.seed,
        extra_body: model.promptFormat.disableThinking ? { enable_thinking: false } : undefined,
      });

      let raw = '';
      let shown = '';
      let firstAt = 0;
      let finish: string | null = null;
      let usage: { prompt_tokens: number; completion_tokens: number } | null = null;
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? '';
        if (delta) {
          if (!firstAt) firstAt = performance.now();
          raw += delta;
          // Hold back text until it can't be the start of the empty thinking block.
          const visible = model.promptFormat.disableThinking ? stripThinking(raw) : raw;
          if (visible !== null && visible.length > shown.length) {
            onDelta(visible.slice(shown.length));
            shown = visible;
          }
        }
        finish = chunk.choices[0]?.finish_reason ?? finish;
        if (chunk.usage) usage = chunk.usage;
      }
      if (model.promptFormat.disableThinking) {
        const visible = stripThinking(raw, true) ?? raw;
        if (visible.length > shown.length) onDelta(visible.slice(shown.length));
        shown = visible;
      }
      const end = performance.now();
      if (shown !== raw) this.rawReplies.set(shown, raw);
      const cancelled = this.cancelRequested || finish === 'abort';
      const prefilled = usage?.prompt_tokens ?? 0;
      const generated = usage?.completion_tokens ?? 0;
      const promptTokens = reused + prefilled;
      this.conversation = cancelled ? null : { messages: [...messages, { role: 'assistant', content: raw }], tokens: promptTokens + generated };
      if (cancelled) await engine.resetChat().catch(() => {});
      const finishReason: FinishReason = cancelled ? 'cancelled' : finish === 'length' ? 'length' : 'stop';
      const decodeMs = firstAt ? end - firstAt : 0;
      return {
        text: shown,
        tokenIds: [],
        finishReason,
        stats: {
          promptTokens,
          reusedPromptTokens: reused,
          droppedTurns: plan.droppedTurns,
          generatedTokens: generated,
          prefillMs: firstAt ? firstAt - start : end - start,
          timeToFirstTokenMs: firstAt ? firstAt - start : 0,
          decodeMs,
          decodeTokensPerSecond: generated > 1 && decodeMs > 0 ? ((generated - 1) * 1000) / decodeMs : 0,
          totalMs: end - start,
        },
      };
    } catch (err) {
      this.conversation = null;
      throw this.generationError(err);
    } finally {
      this.generating = false;
      this.cancelRequested = false;
    }
  }

  cancel(): void {
    if (!this.generating) return;
    this.cancelRequested = true;
    void this.loaded?.engine.interruptGenerate();
  }

  reset(): void {
    this.conversation = null;
    void this.loaded?.engine.resetChat().catch(() => {});
  }

  async unload(): Promise<void> {
    this.conversation = null;
    this.rawReplies.clear();
    const loading = this.loading;
    this.loading = null;
    const loaded = this.loaded;
    this.loaded = null;
    await Promise.all([loading?.unload(), loaded?.engine.unload()].map((p) => p?.catch(() => {})));
  }

  private toMessage(turn: ChatTurn): ChatCompletionMessageParam {
    if (turn.role === 'assistant') return { role: 'assistant', content: this.rawReplies.get(turn.content) ?? turn.content };
    return { role: turn.role, content: turn.content };
  }

  /**
   * Token count with the model's own tokenizer. WebLLM doesn't expose it, so
   * this reaches into the loaded pipeline (pinned WebLLM version); if that ever
   * changes, it falls back to a cautious estimate.
   */
  private countTokens(text: string): number {
    const { engine, artifact } = this.requireLoaded();
    try {
      const pipelines = (engine as unknown as { loadedModelIdToPipeline: Map<string, { tokenizer: { encode(t: string): Int32Array } }> })
        .loadedModelIdToPipeline;
      const tokenizer = pipelines.get(artifact.modelId)?.tokenizer;
      if (tokenizer) return tokenizer.encode(text).length;
    } catch {
      // Fall through to the estimate.
    }
    return Math.ceil(text.length / 2.5);
  }

  private generationError(err: unknown): RuntimeError {
    if (err instanceof RuntimeError) return err;
    const name = err instanceof Error ? err.name || err.constructor.name : '';
    if (/ContextWindowSizeExceeded/.test(name)) {
      return new RuntimeError('context-overflow', 'The conversation is too long for the model’s context window.');
    }
    if (isDeviceLoss(err)) {
      const error = new RuntimeError('device-lost', `The GPU stopped responding: ${message(err)}`, true);
      this.loaded = null;
      this.events.onFatal?.(error);
      return error;
    }
    return new RuntimeError('generation-failed', `Generation failed: ${message(err)}`, true);
  }

  private requireLoaded(): Loaded {
    if (!this.loaded) throw new RuntimeError('not-loaded', 'No model is loaded.');
    return this.loaded;
  }
}

/**
 * The GPU device went away. A buffer unmapped while being read back is the same thing seen from
 * WebLLM's side (an Adreno phone during a long prompt).
 */
function isDeviceLoss(err: unknown): boolean {
  const name = err instanceof Error ? err.name || err.constructor.name : '';
  return /DeviceLost|ModelNotLoaded/.test(name) || /device (was )?lost|unmapped before mapping was resolved/i.test(message(err));
}

/** Maps WebLLM's progress reports (a fraction plus text) onto our load phases. */
function progressOf(report: InitProgressReport, totalBytes: number, shardBytes: number): LoadProgress {
  if (report.text.startsWith('Fetching param cache')) {
    return {
      phase: 'downloading',
      file: 'weights',
      // Config, tokenizer and library come first; the shards make up the rest.
      loadedBytes: Math.round(totalBytes - shardBytes + report.progress * shardBytes),
      totalBytes,
    };
  }
  if (/^Loading (model from cache|GPU shader)/.test(report.text) || report.text.startsWith('Finish loading')) {
    return { phase: 'loading' };
  }
  return { phase: 'checking' };
}

/** WebLLM's load errors in the app's terms. */
function loadError(err: unknown): RuntimeError {
  const name = err instanceof Error ? err.name || err.constructor.name : '';
  const text = message(err);
  if (/ShaderF16Support|FeatureSupport/.test(name)) return new RuntimeError('incompatible-device', text);
  if (/WebGPUNotAvailable|WebGPUNotFound/.test(name)) return new RuntimeError('webgpu-unavailable', text);
  if (/Integrity/.test(name)) return new RuntimeError('integrity-mismatch', `A downloaded file didn’t match its checksum: ${text}`, true);
  if (/QuotaExceeded/.test(name) || /quota/i.test(text)) return new RuntimeError('storage-full', text, true);
  if (/DeviceLost/.test(name) || /device (was )?lost/i.test(text)) return new RuntimeError('device-lost', `The GPU stopped responding while loading: ${text}`, true);
  if (/Failed to fetch|NetworkError|Cannot fetch|status/i.test(text)) return new RuntimeError('download-failed', text, true);
  return new RuntimeError('load-failed', `WebLLM couldn’t load the model: ${text}`, true);
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
