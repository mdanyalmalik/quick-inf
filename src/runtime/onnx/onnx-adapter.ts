// ONNX Runtime Web (WebGPU) adapter for decoder-only models exported in the
// Hugging Face / Optimum layout: inputs `input_ids`, `attention_mask`,
// `position_ids`, `past_key_values.{i}.{key,value}`; outputs `logits` and
// `present.{i}.{key,value}`.
//
// Also Qwen3.5's layout (transformers.js exports): token embeddings come from a
// separate graph and enter as `inputs_embeds`; `position_ids` has three rows
// (multimodal rotary positions, equal for text); and most layers use linear
// attention, carrying `past_conv.{i}` and `past_recurrent.{i}` states of fixed
// size instead of a growing KV cache (`present_conv.{i}`, `present_recurrent.{i}`).
// ONNX Runtime GenAI's Qwen3.5 builds name those states `past_key_values.{i}.conv_state`
// and `.recurrent_state`, and their embeddings graph also takes `image_features` (empty
// for text). LFM2's exports carry only `past_conv.{i}` states besides the KV cache, and take
// `num_logits_to_keep`; Gemma 3's and Granite 4's take no `position_ids`.
//
// Weights stream from the saved file straight to the GPU (external-weights.ts).
// The KV cache stays on the GPU between steps (present -> past), and only the
// last row of logits is read back. The cache is kept between replies: the next
// request reuses its longest shared start, cut down on the GPU when needed.
//
// Static caches and graph capture: where every attention layer is GroupQueryAttention reading the graph's KV inputs
// directly (most exports), the KV cache is instead one fixed buffer per layer as long as the conversation may get,
// passed as both the past and the present (GroupQueryAttention then writes the new positions in place, told how
// much is filled by inputs of its own: external-weights.ts, staticAttentionInputs). Fixed-size states (linear attention's)
// are updated in place where their op allows it (CausalConvWithState, LinearAttention), else alternate between two
// buffers. The small inputs live in fixed buffers too, so a decoding step has the same shapes and buffers every time,
// and ONNX Runtime records its GPU work once and replays it ("graph capture", one captured graph per buffer
// alternation): no per-kernel CPU work per token, which was most of a small model's time (ONNX Runtime spends some
// 10-20 µs of CPU per op it dispatches). Prompts run uncaptured (gpu_graph_id -1) on the same buffers. Cutting the
// kept cache back is then just a shorter mask.

import type { InferenceSession, Tensor } from 'onnxruntime-web/webgpu';
import type { ModelArtifact, ModelSpec, OnnxArtifact } from '../../catalogue/models';
import { getGpuAdapter, getGpuDevice } from '../../device/gpu';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { createPromptFormatter, type PromptFormatOptions, type PromptFormatter } from '../../tokenizer/prompt-format';
import { HarmonyStream, MuseGlimmerStream } from '../../tokenizer/harmony';
import { StreamingDecoder } from '../../tokenizer/streaming-decoder';
import { fitTurns } from '../context';
import {
  RuntimeError,
  type FinishReason,
  type GenerationRequest,
  type GenerationResult,
  type LoadedArtifacts,
  type LoadProgress,
  type RuntimeAdapter,
} from '../contract';
import { applyRepetitionPenalty, createRng, sampleToken } from '../sampling';
import {
  ATTENTION_BIAS_INPUT,
  bytesSource,
  externalizeWeights,
  graphStates,
  leanGraph,
  halfToFloat32,
  ROTARY_COS_INPUT,
  ROTARY_ROWS_INPUT,
  ROTARY_SIN_INPUT,
  SEQLENS_INPUT,
  staticAttentionInputs,
  stateWindows,
  exposeHidden,
  HIDDEN_OUTPUT,
  unpruneLogits,
  textOnlyEmbeddings,
  TOTAL_LENGTH_INPUT,
  lazyBytes,
  releaseReadBuffer,
  type ByteSource,
  type GraphStates,
  type RewriteOptions,
  LOGIT_ROWS,
} from './external-weights';
import { draftSessionOptions, EagleDrafter } from './eagle-drafter';
import { MtpDrafter } from './mtp-drafter';
import { prefillChunkTokens } from './limits';
import { opsetImports, splitMoe } from './moe-split';
import { createStreamedSession, type StreamedSession } from './moe-stream';
import { getOrt, loadOrt } from './ort-env';

/** What the adapter needs of a model session: an ONNX Runtime session, or streamed experts' (moe-stream.ts). */
type ModelSession = Pick<InferenceSession, 'inputMetadata' | 'run' | 'release'> | StreamedSession;

/** A state input carried from one step to the next, and the output that updates it. */
interface StateInput {
  name: string;
  output: string;
  /** 'kv': grows by the new tokens; 'conv' and 'recurrent': fixed size (linear attention). */
  kind: 'kv' | 'conv' | 'recurrent';
  type: 'float16' | 'float32';
  /** Shape of the empty state (a KV cache with no positions). */
  emptyDims: number[];
}

/** What the loaded graph itself expects, read from its input metadata. */
interface GraphShape {
  /** State inputs in layer order (keys before values). */
  states: StateInput[];
  kvType: 'float16' | 'float32';
  kvHeads: number;
  headDim: number;
  positionIds: boolean;
  /** Rows of position_ids: 3 for Qwen3.5's multimodal rotary positions. */
  positionRows: number;
  /** Takes `num_logits_to_keep` (LFM2): how many final positions get logits. */
  logitsToKeep: boolean;
  /** Takes `inputs_embeds` from the separate embeddings graph instead of `input_ids`. */
  embedsInput: boolean;
  /** Also takes `input_ids` beside `inputs_embeds` (Gemma 4 26B's GenAI build). */
  idsWithEmbeds: boolean;
  /** Also takes `per_layer_inputs` from the embeddings graph (Gemma 3n/4). */
  perLayerInputs: boolean;
  /** The embeddings graph's image input (GenAI's multimodal builds), fed empty: [0, hidden size]. */
  imageFeatures: { type: 'float16' | 'float32'; width: number } | null;
  /** Has linear-attention state, which can't be cut back to a shorter prefix. */
  recurrent: boolean;
  /** Element type of the causal-mask input the rewrite added, if it did (see external-weights.ts). */
  attentionBias: 'float16' | 'float32' | null;
}

interface Loaded {
  model: ModelSpec;
  artifact: OnnxArtifact;
  session: ModelSession;
  /** The separate token-embeddings graph, if the model has one. */
  embeddings: InferenceSession | null;
  graph: GraphShape;
  tokenizer: BpeTokenizer;
  formatter: PromptFormatter;
  stopIds: Set<number>;
  prefillChunk: number;
  /** Null on the CPU engine. */
  device: GPUDevice | null;
  /** Reused buffer the last row of logits is copied into for reading. */
  staging: GPUBuffer | null;
  /** Static caches (see the top of this file): which state outputs their op updates in place, and whether decoding is graph-captured. */
  staticPlan: {
    inPlace: Set<string>;
    capture: boolean;
    rotaryFrequencies: Float32Array | null;
    /** States keeping per-token slots (external-weights.ts: stateWindows), so checking passes can be cut back. */
    windows: { size: number; pasts: Set<string> } | null;
  } | null;
  /** Logits are [1, 1, vocabulary] (the output layer computes the last position only). */
  lastTokenLogits: boolean;
  /** LoadOptions.speculate (by default the draft head's guesses, else NGRAM_GUESSES). */
  speculate: number;
  /** The artifact's draft head's session (OnnxArtifact.draft), when the graph outputs taps for it. */
  draft: InferenceSession | null;
  /** The graph output the draft layer reads: eagle_taps (EAGLE-3) or HIDDEN_OUTPUT (MTP). */
  draftFeatures: string;
}

/** The fixed buffers of a static cache (see the top of this file). */
interface StaticStore {
  /** Positions the KV buffers hold. */
  maxLength: number;
  /** Per state, in graph.states order: one tensor (KV, or updated in place) or two used alternately. */
  sets: Tensor[][];
  /** Which of two alternating tensors holds each such state now. */
  parity: 0 | 1;
  /** Captured graphs are numbered 2 × epoch + parity; a new store (other buffers) starts a new epoch. */
  epoch: number;
  /** The attention mask, as long as the KV buffers and all ones (see allocateStatic). */
  mask: Tensor;
  maskData: BigInt64Array<ArrayBuffer>;
  positions: Tensor | null;
  ids: Tensor;
  embeds: Tensor | null;
  logitsToKeep: Tensor | null;
  logits: Tensor | null;
  /** GroupQueryAttention's lengths (see external-weights.ts: staticAttentionInputs), on the GPU when captured. */
  seqlens: Tensor;
  totalLength: Tensor;
  /** Rotary inputs (see external-weights.ts: staticAttentionInputs), for one new token. */
  rotary: { rows: Tensor; cos: Tensor; sin: Tensor } | null;
  /**
   * Captured passes of a fixed number of tokens (checking guesses: the last token plus 1 to `speculate` guesses), by
   * token count: their ids, logits, taps and positions buffers.
   */
  passes: Map<number, FixedPass>;
  /** Holds one window slot while it's copied within its state's buffer (WebGPU copies need two buffers). */
  windowScratch: GPUBuffer | null;
  /** Fixed passes' graph ids already recorded (see forwardStatic). */
  recorded: Set<number>;
}

interface FixedPass {
  ids: Tensor;
  logits: Tensor;
  taps: Tensor | null;
  positions: Tensor | null;
  rotary: { rows: Tensor; cos: Tensor; sin: Tensor } | null;
  embeds: Tensor | null;
}

/** Tensors that belong to a static store: never disposed with a cache. */
const pinned = new WeakSet<Tensor>();

/** Disposes a tensor unless a static store owns it. */
function release(tensor: Tensor): void {
  if (!pinned.has(tensor)) tensor.dispose();
}

/** Captured graph ids from here on are fixed-size passes' (StaticStore.passes); below, single steps'. */
const FIXED_GRAPHS = 1 << 20;

/** Load options, for diagnostics experiments; the app uses the defaults. */
export interface LoadOptions {
  /** Not used: an ONNX session can't be stopped part-way; the worker cancels between steps. */
  signal?: AbortSignal;
  /** Extra ONNX Runtime session options. */
  sessionOverrides?: Partial<InferenceSession.SessionOptions>;
  /** Bits for a full-precision output layer (see external-weights.ts); 0 keeps it as it is. Default 8. */
  quantizeOutput?: 0 | 4 | 8;
  /** Run on this device instead of the shared one. */
  device?: GPUDevice;
  /** 'cpu': ONNX Runtime's WebAssembly engine instead of WebGPU (devices without a usable GPU). */
  processor?: 'gpu' | 'cpu';
  /** Prompt format overrides (diagnostics: Llama's fixed reference date). */
  formatOptions?: PromptFormatOptions;
  /**
   * Mixture-of-experts models: keep only the shared weights on the GPU and load experts as the router picks them,
   * with this much GPU memory for recently used ones (see moe-stream.ts). Ignored for other models and on the CPU.
   */
  streamExperts?: { cacheBytes: number; maxSlots?: number };
  /** Static caches where the graph allows them (see the top of this file). Default true. */
  staticCache?: boolean;
  /** Graph-captured decoding with a static cache. Default true. */
  graphCapture?: boolean;
  /** Fewer kernels per token on WebGPU (external-weights.ts: leanGraph). Default true. */
  leanGraph?: boolean;
  /**
   * Speculative decoding: up to this many guessed tokens checked per pass (see generate). Default: the artifact's draft
   * head's `guesses`, else NGRAM_GUESSES (guesses from earlier text); 0 turns it off.
   */
  speculate?: number;
  /** MatMulNBits' accuracy_level for every 4/8-bit matrix product (4: int8 activations, which unlocks ONNX Runtime's DP4A and subgroup-matrix prompt kernels). */
  matMulNBitsAccuracy?: number;
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
  /** The EAGLE-3 draft head's session and cache (LoadOptions.draft), made at the first generation. */
  private drafter: EagleDrafter | MtpDrafter | null = null;
  /** Static passes also return the graph's eagle_taps (drafting), never as captured single steps. */
  private tapsWanted = false;
  /** Positions whose draft rows are written with the right next token (the drafter's cache is good up to here). */
  private draftValid = 0;
  private cancelRequested = false;
  private watchedDevice: GPUDevice | null = null;
  private fatal: RuntimeError | null = null;
  /** The GPU reported running out of memory since the current load began (ONNX Runtime doesn't check). */
  private outOfMemory = false;
  private pendingFault: 'wasm-crash' | 'hang' | null = null;
  /**
   * The KV cache left by the last reply and the tokens it covers. The next
   * request in a chat starts with those same tokens (history + reply), so only
   * the new message has to be prefilled.
   */
  private kept: { ids: number[]; cache: KvCache } | null = null;
  /**
   * Linear-attention models only: a copy of the state at the end of the last
   * prompt's turns, before its generation prompt. Their state can't be cut back,
   * and templates render past replies differently from how they were generated
   * (Qwen's without the empty thinking block), so the next prompt usually matches
   * `kept` only up to there.
   */
  private checkpoint: { ids: number[]; cache: KvCache } | null = null;
  private keptLength = 0;
  /** Lets go of weight files kept open for streamed experts. */
  private releaseFiles: (() => void) | null = null;
  /** The static cache's buffers, when the loaded model uses one. */
  private static: StaticStore | null = null;
  private staticEpoch = 0;

  constructor(private readonly events: AdapterEvents = {}) {}

  async load(
    model: ModelSpec,
    artifact: ModelArtifact,
    files: LoadedArtifacts | null,
    onProgress: (p: LoadProgress) => void,
    options: LoadOptions = {},
  ): Promise<{ sessionMs: number; validateMs: number }> {
    const { sessionOverrides = {} } = options;
    if (this.fatal) throw this.fatal;
    if (artifact.engine !== 'onnx-webgpu' || !files) throw new RuntimeError('load-failed', 'Not an ONNX model file.');
    const formatKind = model.promptFormat.kind;
    await this.unload();
    const ort = await loadOrt();
    const onCpu = options.processor === 'cpu';

    let tokenizer: BpeTokenizer;
    try {
      tokenizer = BpeTokenizer.fromJSON(files.tokenizerJson ?? JSON.parse(new TextDecoder().decode(files.files?.[model.tokenizer.path])));
    } catch (err) {
      throw new RuntimeError('load-failed', `Tokenizer could not be loaded: ${message(err)}`);
    }

    onProgress({ phase: 'loading' });
    const sessionStart = performance.now();
    // Every output stays on the GPU: the states go straight back in, and a prefill step copies
    // back one row of logits instead of the whole [tokens × vocabulary] block (50+ MB for a
    // 256-token chunk with Qwen's vocabulary).
    let device: GPUDevice | null = null;
    if (!onCpu) {
      try {
        device = options.device ?? (await getGpuDevice());
        // ONNX Runtime requests (and then drops) an adapter of its own unless given one.
        const adapter = await getGpuAdapter();
        if (adapter) ort.env.webgpu.adapter = adapter;
      } catch (err) {
        throw new RuntimeError('load-failed', `The GPU could not be opened: ${message(err)}`, true);
      }
      this.watchDevice(device);
    }
    this.outOfMemory = false;
    let session: ModelSession;
    let embeddings: InferenceSession | null = null;
    let lastTokenLogits = true;
    let staticPlan: Loaded['staticPlan'] = null;
    const sessionOptions = (externalData: ReturnType<typeof modelInput>['externalData'], capture = false): InferenceSession.SessionOptions => ({
      externalData,
      // Run on the device the probe checked, not one ONNX Runtime picks itself.
      // Buffers freed by a run go back at its end instead of into ONNX Runtime's cache, which
      // never shrinks: a long prompt's temporary buffers otherwise stay held (300+ MB). No speed cost measured.
      ...(device
        ? {
            // Native options passed through (tools/patch-ort.mjs). Graph capture is switched on in the WebGPU backend
            // only: ONNX Runtime Web's own session option also makes every run reuse the first run's inputs and outputs,
            // which prompts (other shapes, uncaptured) can't. int64 kernels on the GPU keep the remaining shape
            // arithmetic there, where it's recorded. More dispatches per compute pass and per submitted command buffer
            // (16 by default): each costs GPU time between kernels, and a model runs several hundred per token.
            executionProviders: [
              {
                name: 'webgpu',
                device,
                storageBufferCacheMode: 'lazyRelease',
                qiOptions: { maxNumPendingDispatches: String(PENDING_DISPATCHES), ...(capture && { enableGraphCapture: '1', enableInt64: '1' }) },
              } as InferenceSession.WebGpuExecutionProviderOption,
            ],
            preferredOutputLocation: 'gpu-buffer' as const,
          }
        : { executionProviders: ['wasm'] }),
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      ...sessionOverrides,
    });
    const create = async (graph: LoadedArtifacts['graph'], externalData: LoadedArtifacts['externalData'], main = false): Promise<ModelSession> => {
      const input = modelInput(graph, externalData, {
        quantizeOutput: options.quantizeOutput ?? 8,
        quantizeWeights: artifact.quantizeWeights,
        matMulNBitsAccuracy: options.matMulNBitsAccuracy,
      });
      // The main graph is created last (an embeddings graph has no logits).
      lastTokenLogits = input.lastTokenLogits;
      // GenAI exports keep only the last position before the output layer: all of them, for checking guesses.
      if (main && input.lastTokenLogits) input.bytes = unpruneLogits(input.bytes) ?? input.bytes;
      const split = main && device && options.streamExperts ? splitMoe(input.bytes) : null;
      if (split && device && options.streamExperts) {
        // Expert bytes are read from the stored files while the model runs: keep them open until it's unloaded.
        this.releaseFiles = files.keepOpen?.() ?? null;
        return createStreamedSession({
          ort,
          device,
          split,
          opset: opsetImports(input.bytes),
          createSession: (bytes) => ort.InferenceSession.create(bytes, sessionOptions(input.externalData)),
          sources: input.sources,
          cacheBytes: options.streamExperts.cacheBytes,
          maxSlots: options.streamExperts.maxSlots ?? 32,
        });
      }
      staticPlan = main && device && options.staticCache !== false ? planStatic(input.states) : null;
      // Text only: image features never come in, so their scatter (through CPU-only ops) goes.
      if (!main) input.bytes = textOnlyEmbeddings(input.bytes) ?? input.bytes;
      const read = (location: string, offset: number, length: number) => input.sources[location]?.read(offset, length) ?? null;
      const rewired = staticPlan && staticAttentionInputs(input.bytes, read, { lastRows: lastTokenLogits });
      if (staticPlan && !rewired) staticPlan = null;
      if (staticPlan && rewired) {
        input.bytes = rewired.graph;
        staticPlan.rotaryFrequencies = rewired.rotaryFrequencies;
      }
      // Fewer kernels per token (float32 detours in float16 graphs); WebGPU only, as the CPU backend lacks float16
      // kernels for some of the ops.
      if (main && device && options.leanGraph !== false) input.bytes = leanGraph(input.bytes, read, { islands: !artifact.keepsFloat32 }) ?? input.bytes;
      // Speculation with recurrent states: their ops keep a slot per checked token (see stateWindows).
      const guesses = Math.min(options.speculate ?? NGRAM_GUESSES, LOGIT_ROWS - 1);
      const windowed = staticPlan && guesses > 0 ? stateWindows(input.bytes, guesses + 1) : null;
      if (staticPlan && windowed) {
        input.bytes = windowed.graph;
        staticPlan.windows = { size: guesses + 1, pasts: windowed.pasts };
      }
      // An MTP draft layer reads the model's final hidden states.
      if (staticPlan && artifact.draft?.kind === 'mtp' && files.draft && options.speculate !== 0) input.bytes = exposeHidden(input.bytes) ?? input.bytes;
      if (staticPlan && options.graphCapture !== false) {
        try {
          const captured = await ort.InferenceSession.create(input.bytes, sessionOptions(input.externalData, true));
          staticPlan.capture = true;
          return captured;
        } catch (err) {
          // Some op the capture can't record: run uncaptured, still on static buffers.
          console.warn('Graph capture unavailable for this model:', err);
        }
      }
      return ort.InferenceSession.create(input.bytes, sessionOptions(input.externalData));
    };
    try {
      if (files.embeddings) embeddings = (await create(files.embeddings.graph, files.embeddings.externalData)) as InferenceSession;
      session = await create(files.graph, files.externalData, true);
    } catch (err) {
      this.releaseFiles?.();
      this.releaseFiles = null;
      releaseReadBuffer();
      await embeddings?.release().catch(() => {});
      if (isRuntimeFetchFailure(err)) {
        // ONNX Runtime remembers a failed start-up, so only a fresh worker can retry (the worker marks it).
        throw new RuntimeError('runtime-files-unavailable', `ONNX Runtime could not start: ${message(err)}`, true);
      }
      if (this.outOfMemory || OUT_OF_MEMORY.test(message(err))) {
        throw new RuntimeError('out-of-memory', `${outOfMemoryText(model)} (${message(err)})`, true);
      }
      throw new RuntimeError('load-failed', `The model could not be loaded on the ${onCpu ? 'CPU' : 'GPU'}: ${message(err)}`, true);
    }
    releaseReadBuffer();
    const sessionMs = performance.now() - sessionStart;

    let graph: GraphShape;
    let formatter: PromptFormatter;
    let stopIds: Set<number>;
    try {
      graph = readGraphShape(session, embeddings, model);
      if (graph.embedsInput !== (embeddings !== null)) throw new Error('the graph and the embeddings file don’t go together');
      formatter = createPromptFormatter(formatKind, tokenizer, { ...model.promptFormat, ...options.formatOptions });
      stopIds = new Set(model.promptFormat.stopTokens.map((t) => tokenizer.specialTokenId(t)));
    } catch (err) {
      await session.release().catch(() => {});
      await embeddings?.release().catch(() => {});
      throw new RuntimeError('load-failed', `The model files don't match the catalogue: ${message(err)}`);
    }
    const prefillChunk = prefillChunkTokens(model.architecture.vocabSize, lastTokenLogits);
    // The draft layer (EAGLE-3 heads: development only; static caches only: its rows are cut back by overwriting, as
    // the model's).
    let draft: InferenceSession | null = null;
    const mtp = artifact.draft?.kind === 'mtp';
    const draftFeatures = mtp ? HIDDEN_OUTPUT : 'eagle_taps';
    if (
      (import.meta.env.DEV || mtp) &&
      files.draft &&
      artifact.draft &&
      device &&
      staticPlan &&
      options.speculate !== 0 &&
      (session as InferenceSession).outputNames?.includes(draftFeatures)
    ) {
      try {
        const input = modelInput(files.draft.graph, files.draft.externalData, { externalizeOnly: true });
        draft = await ort.InferenceSession.create(input.bytes, draftSessionOptions(sessionOptions(input.externalData)));
      } catch (err) {
        await session.release().catch(() => {});
        await embeddings?.release().catch(() => {});
        throw new RuntimeError('load-failed', `The model's draft head could not be loaded: ${message(err)}`, true);
      }
    }
    this.loaded = {
      model,
      artifact,
      session,
      embeddings,
      graph,
      tokenizer,
      formatter,
      stopIds,
      prefillChunk,
      device,
      staging: null,
      speculate: options.speculate ?? (draft ? artifact.draft!.guesses : NGRAM_GUESSES),
      draft,
      draftFeatures,
      staticPlan,
      lastTokenLogits,
    };
    // A small store for the check below; generate() sizes it to each conversation's length.
    if (staticPlan) this.allocateStatic(VALIDATION_LENGTH);

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
      if (this.fatal) throw this.fatal;
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
    let cache: KvCache = [];
    // Tokens `cache` covers; null once it can't be kept (cancelled mid-prefill, failed).
    let fed: number[] | null = null;
    try {
      if (this.static && this.static.maxLength < options.contextTokens) this.allocateStatic(options.contextTokens);
      const drafting = loaded.draft !== null && this.static !== null && (!loaded.graph.recurrent || this.windowedStates()) && loaded.speculate > 0;
      if (drafting) {
        this.drafter ??=
          loaded.draftFeatures === HIDDEN_OUTPUT
            ? new MtpDrafter(getOrt(), loaded.device!, loaded.draft!, loaded.artifact.draft!, loaded.speculate)
            : new EagleDrafter(getOrt(), loaded.device!, loaded.draft!, loaded.artifact.draft!, loaded.speculate);
        if (this.drafter.maxLength < this.static!.maxLength) {
          this.drafter.allocate(this.static!.maxLength);
          this.draftValid = 0;
        }
      }
      this.tapsWanted = drafting;
      const { formatter, tokenizer, stopIds } = loaded;
      const promptBudget = options.contextTokens - Math.min(options.maxNewTokens, options.contextTokens / 2);
      const fixed = formatter.fixedLength(request.turns[0]?.role === 'system');
      const plan = fitTurns(request.turns, promptBudget, (t) => formatter.turnLength(t), fixed);
      const promptIds = formatter.format(plan.turns);
      const maxNew = Math.min(options.maxNewTokens, options.contextTokens - promptIds.length);

      const random = createRng(options.seed ?? (Math.random() * 2 ** 32) >>> 0);
      const kind = loaded.model.promptFormat.kind;
      const decoder = kind === 'harmony' ? new HarmonyStream(tokenizer) : kind === 'muse-glimmer' ? new MuseGlimmerStream(tokenizer) : new StreamingDecoder(tokenizer);
      const generated: number[] = [];
      let text = '';
      let finishReason: FinishReason = 'length';

      const reused = this.takeKept(promptIds);
      cache = reused ?? this.emptyCache();
      // Drafting: the target feeds again from the first position whose draft row needs writing (the row before the
      // first new token pairs with that token), for its taps.
      const reusedTokens = reused ? (drafting ? Math.max(0, Math.min(this.keptLength - 1, this.draftValid)) : this.keptLength) : 0;
      const turnsEnd = promptIds.length - formatter.generationLength;
      let prefilled: { logits: Float32Array<ArrayBuffer>; cache: KvCache; cancelled: boolean; taps?: { tensor: Tensor; start: number } };
      if (loaded.graph.recurrent && turnsEnd > reusedTokens) {
        prefilled = await this.prefill(promptIds.slice(reusedTokens, turnsEnd), cache, reusedTokens, false);
        cache = prefilled.cache;
        if (!prefilled.cancelled) {
          this.dropCheckpoint();
          this.checkpoint = { ids: promptIds.slice(0, turnsEnd), cache: this.cloneCache(cache) };
          prefilled = await this.prefill(promptIds.slice(turnsEnd), cache, turnsEnd);
        }
      } else {
        prefilled = await this.prefill(promptIds.slice(reusedTokens), cache, reusedTokens);
      }
      cache = prefilled.cache;
      fed = prefilled.cancelled ? null : [...promptIds];
      let logits = prefilled.logits;
      // The last pass's taps (drafting), for the drafter's next guesses.
      let lastTaps = prefilled.taps;
      const prefillMs = performance.now() - start;
      let firstTokenAt = 0;

      if (prefilled.cancelled) finishReason = 'cancelled';
      else {
        // Speculative decoding (static caches without linear attention, where a pass's rejected guesses are simply
        // overwritten later; a recurrent state would keep them): each pass
        // feeds the last sampled token plus tokens guessed to follow it, and `rows` holds the logits of each of those
        // positions. Tokens are sampled from the rows in turn; a guess is kept while the model's own sample equals it
        // (exactly the model's distribution: the guesses carry no probabilities). The first sample that differs is
        // fed by the next pass. With no guesses, a pass is one decoding step.
        const speculate = this.static && (!loaded.graph.recurrent || this.windowedStates()) ? Math.min(loaded.speculate, LOGIT_ROWS - 1) : 0;
        const vocab = loaded.model.architecture.vocabSize;
        let rows: Float32Array<ArrayBuffer>[] = [logits];
        let guesses: number[] = [];
        // Generated tokens whose positions the cache holds.
        let cached = 0;
        // The last checking pass: generated tokens before it, and its tokens (windowed states are cut back to the kept
        // ones before the next pass, see settleWindows).
        let pass: { base: number; size: number } | null = null;
        // The share of recent guesses kept (moving average): text that doesn't repeat itself gets guesses only from
        // 4-token matches (a missed guess still costs a row of the pass).
        let keptShare = 0.5;
        const settle = () => {
          if (pass && pass.size > 1) this.settleWindows(cached - pass.base, pass.size);
          pass = null;
        };
        decoding: for (;;) {
          let token = -1;
          for (let i = 0; i < rows.length; i++) {
            if (this.cancelRequested) {
              finishReason = 'cancelled';
              break decoding;
            }
            const row = rows[i]!;
            if (options.repetitionPenalty !== 1) {
              const window = options.repetitionWindow;
              const recent =
                generated.length >= window ? generated.slice(-window) : [...promptIds.slice(-(window - generated.length)), ...generated];
              applyRepetitionPenalty(row, recent, options.repetitionPenalty);
            }
            const sampleStart = performance.now();
            // Development: each sampled row's top-1 minus top-2 logit, to tell near ties from errors (bench &full=1).
            const margins = import.meta.env.DEV ? (globalThis as { __quickInfMargins?: number[] }).__quickInfMargins : undefined;
            if (margins) {
              let first = -Infinity;
              let second = -Infinity;
              for (const v of row) {
                if (v > first) [first, second] = [v, first];
                else if (v > second) second = v;
              }
              margins.push(first - second);
            }
            token = sampleToken(row, options, random);
            profilePhase('sample', sampleStart);
            if (generated.length === 0) firstTokenAt = performance.now();
            if (stopIds.has(token)) {
              finishReason = 'stop';
              break decoding;
            }
            generated.push(token);
            const kept = i < guesses.length && token === guesses[i];
            if (kept) cached++;
            speculation.accepted += kept ? 1 : 0;
            const delta = decoder.push(token);
            if (delta) {
              text += delta;
              onDelta(delta);
            }
            if (generated.length >= maxNew) {
              finishReason = 'length';
              break decoding;
            }
            if (!kept) break;
          }
          if (lastTaps && this.drafter) {
            // Draft rows for the last pass's positions: row i pairs its taps with the token after it.
            const length = promptIds.length + generated.length;
            const at = (i: number) => (i < promptIds.length ? promptIds[i]! : generated[i - promptIds.length]!);
            const { tensor, start } = lastTaps;
            const ids = Array.from({ length: tensor.dims[1]! }, (_, i) => (start + i + 1 < length ? at(start + i + 1) : token));
            const last = length - 2 - start;
            // Where the text repeats earlier text (a 4-token match), its continuation is the better guess, and free:
            // the draft layer then only takes the pass's rows into its cache.
            const repeated = ngramGuesses(promptIds, generated, speculate, 4);
            const draftStart = performance.now();
            guesses = await this.drafter.guess(tensor, ids, start, last, repeated.length ? 0 : speculate);
            if (repeated.length) guesses = repeated;
            profilePhase('draft', draftStart);
            this.draftValid = start + last + 1;
            release(tensor);
            lastTaps = undefined;
          } else {
            if (pass && pass.size > 1) keptShare = 0.8 * keptShare + 0.2 * ((cached - pass.base) / (pass.size - 1));
            guesses = speculate ? ngramGuesses(promptIds, generated, speculate, keptShare < 0.2 ? 4 : 2) : [];
          }
          speculation.passes++;
          speculation.guessed += guesses.length;
          settle();
          pass = { base: generated.length, size: guesses.length + 1 };
          const step = await this.forward([token, ...guesses], cache, promptIds.length + generated.length - 1, true, guesses.length + 1);
          cache = step.cache;
          if (step.taps) lastTaps = { tensor: step.taps, start: promptIds.length + generated.length - 1 };
          cached = generated.length;
          rows = Array.from({ length: guesses.length + 1 }, (_, i) => step.logits.subarray(i * vocab, (i + 1) * vocab));
        }
        settle();
        if (lastTaps) release(lastTaps.tensor);
        if (fed) fed = [...promptIds, ...generated.slice(0, cached)];
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
      this.tapsWanted = false;
      this.generating = false;
      this.cancelRequested = false;
    }
  }

  cancel(): void {
    if (this.generating) this.cancelRequested = true;
  }

  reset(): void {
    this.dropKept();
    this.dropCheckpoint();
  }

  async unload(): Promise<void> {
    this.dropKept();
    this.dropCheckpoint();
    const loaded = this.loaded;
    this.loaded = null;
    this.destroyStatic();
    this.drafter?.destroy();
    this.drafter = null;
    if (loaded) {
      loaded.staging?.destroy();
      await loaded.draft?.release();
      await loaded.session.release();
      await loaded.embeddings?.release();
    }
    this.releaseFiles?.();
    this.releaseFiles = null;
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
    this.dropKept();
    this.dropCheckpoint();
    const needed = promptIds.length + continuation.length;
    if (this.static && this.static.maxLength < needed) this.allocateStatic(Math.max(needed, this.requireLoaded().model.generation.contextTokens));
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
    device.addEventListener('uncapturederror', (event) => {
      if (typeof GPUOutOfMemoryError !== 'undefined' && event.error instanceof GPUOutOfMemoryError) this.outOfMemory = true;
    });
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

  private markFatal(code: 'device-lost' | 'runtime-crashed' | 'out-of-memory', text: string): RuntimeError {
    if (this.fatal) return this.fatal;
    const error = new RuntimeError(code, text, true);
    this.fatal = error;
    this.cancelRequested = true;
    this.kept = null;
    this.checkpoint = null;
    this.static = null;
    const loaded = this.loaded;
    this.loaded = null;
    // Best effort: the session's GPU resources may already be gone.
    loaded?.session.release().catch(() => {});
    loaded?.embeddings?.release().catch(() => {});
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
    const startsWith = (ids: number[]) => ids.length < promptIds.length && ids.every((id, i) => promptIds[i] === id);
    // A linear-attention model reuses whichever saved state the prompt continues exactly.
    if (this.loaded?.graph.recurrent && this.checkpoint && !(this.kept && startsWith(this.kept.ids)) && startsWith(this.checkpoint.ids)) {
      const { ids, cache } = this.checkpoint;
      this.checkpoint = null;
      this.dropKept();
      this.keptLength = ids.length;
      return this.static ? this.restoreStatic(cache) : cache;
    }
    this.dropCheckpoint();
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
    // Linear-attention state covers every kept token and can't be cut back.
    const cuttable = !this.requireLoaded().graph.recurrent;
    if (cuttable && common >= MIN_REUSED_TOKENS && this.loaded?.device && kept.cache.every((t) => t.location === 'gpu-buffer')) {
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
    // A static cache holds every position already: the shorter mask is the cut.
    if (this.static) return [...cache];
    const device = this.requireLoaded().device!;
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

  private dropCheckpoint(): void {
    if (this.checkpoint) disposeCache(this.checkpoint.cache);
    this.checkpoint = null;
  }

  /** A copy of `cache` (in new GPU buffers, or new arrays on the CPU engine). */
  private cloneCache(cache: KvCache): KvCache {
    const { device } = this.requireLoaded();
    const ort = getOrt();
    if (!device) {
      return cache.map((tensor) =>
        tensor.type === 'float16'
          ? new ort.Tensor('float16', (tensor.data as Uint16Array).slice(), tensor.dims)
          : new ort.Tensor('float32', (tensor.data as Float32Array).slice(), tensor.dims),
      );
    }
    const encoder = device.createCommandEncoder();
    const copy = cache.map((tensor, i) => {
      // A static KV buffer keeps the copied positions until they're overwritten, which only happens past them.
      if (this.static && this.requireLoaded().graph.states[i]!.kind === 'kv') return tensor;
      const source = tensor.gpuBuffer as GPUBuffer;
      const buffer = device.createBuffer({ size: source.size, usage: BUFFER_STORAGE | BUFFER_COPY_SRC | BUFFER_COPY_DST });
      encoder.copyBufferToBuffer(source, 0, buffer, 0, source.size);
      return ort.Tensor.fromGpuBuffer(buffer, {
        dataType: tensor.type as 'float16' | 'float32',
        dims: tensor.dims,
        dispose: () => buffer.destroy(),
      });
    });
    device.queue.submit([encoder.finish()]);
    return copy;
  }

  private requireLoaded(): Loaded {
    if (!this.loaded) throw new RuntimeError('not-loaded', 'No model is loaded.');
    return this.loaded;
  }

  /** Feeds `ids` after the `pastLength` tokens already in `cache`. */
  private async prefill(ids: number[], cache: KvCache, pastLength = 0, wantLogits = true) {
    const chunk = this.requireLoaded().prefillChunk;
    let logits = new Float32Array(0);
    // Drafting: earlier chunks' taps go to the drafter's cache; the last chunk's are returned (its last row pairs with
    // the token sampled next).
    let taps: { tensor: Tensor; start: number } | undefined;
    for (let offset = 0; offset < ids.length; offset += chunk) {
      if (this.cancelRequested) return { logits, cache, cancelled: true };
      // Only the last chunk's logits are used.
      const step = await this.forward(
        ids.slice(offset, offset + chunk),
        cache,
        pastLength + offset,
        wantLogits && offset + chunk >= ids.length,
      );
      cache = step.cache;
      logits = step.logits;
      if (step.taps) {
        const end = offset + step.taps.dims[1]!;
        if (end < ids.length && this.drafter) {
          await this.drafter.guess(step.taps, ids.slice(offset + 1, end + 1), pastLength + offset, end - offset - 1, 0);
          this.draftValid = pastLength + end;
          release(step.taps);
        } else taps = { tensor: step.taps, start: pastLength + offset };
      }
    }
    return { logits, cache, cancelled: false, taps };
  }

  /**
   * One forward pass. Consumes `cache` (disposing its GPU buffers) and returns the next one. `logits` are the last
   * `keep` positions' rows, one after another (more than one: checking guessed tokens, at most LOGIT_ROWS).
   */
  private async forward(
    ids: number[],
    cache: KvCache,
    pastLength: number,
    wantLogits = true,
    keep = 1,
  ): Promise<{ logits: Float32Array<ArrayBuffer>; cache: KvCache; taps?: Tensor }> {
    if (this.static) return this.forwardStatic(ids, cache, pastLength, wantLogits, keep);
    const { session, embeddings, model, graph, artifact } = this.requireLoaded();
    const ort = getOrt();
    const n = ids.length;
    const total = pastLength + n;

    const idsTensor = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]);
    const feeds: Record<string, Tensor> = {
      attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]),
    };
    if (graph.positionIds) {
      const rows = graph.positionRows;
      const positions = BigInt64Array.from({ length: rows * n }, (_, i) => BigInt(pastLength + (i % n)));
      feeds.position_ids = new ort.Tensor('int64', positions, rows === 1 ? [1, n] : [rows, 1, n]);
    }
    if (graph.attentionBias) feeds[ATTENTION_BIAS_INPUT] = causalMask(ort, graph.attentionBias, n, pastLength);
    // Only the last position's logits are read.
    if (graph.logitsToKeep) feeds[LOGITS_TO_KEEP_INPUT] = new ort.Tensor('int64', BigInt64Array.of(1n), []);
    graph.states.forEach((state, i) => {
      feeds[state.name] = cache[i]!;
    });

    let outputs: InferenceSession.OnnxValueMapType;
    let embedded: Tensor | null = null;
    let perLayer: Tensor | null = null;
    try {
      if (this.pendingFault) {
        const fault = this.pendingFault;
        this.pendingFault = null;
        if (fault === 'wasm-crash') throw new WebAssembly.RuntimeError('unreachable executed (simulated)');
        await new Promise<never>(() => {});
      }
      if (embeddings) {
        const embeddingFeeds: Record<string, Tensor> = { input_ids: idsTensor };
        if (graph.imageFeatures) {
          const { type, width } = graph.imageFeatures;
          embeddingFeeds[IMAGE_FEATURES_INPUT] =
            type === 'float16'
              ? new ort.Tensor('float16', new Uint16Array(0), [0, width])
              : new ort.Tensor('float32', new Float32Array(0), [0, width]);
        }
        const embedStart = performance.now();
        const embeddingOutputs = await embeddings.run(embeddingFeeds);
        profilePhase('embed', embedStart);
        if (graph.idsWithEmbeds) feeds.input_ids = idsTensor;
        embedded = embeddingOutputs.inputs_embeds as Tensor;
        feeds.inputs_embeds = embedded;
        // Gemma 3n/4: per-layer embeddings, from the same graph.
        if (graph.perLayerInputs) {
          perLayer = embeddingOutputs.per_layer_inputs as Tensor;
          feeds.per_layer_inputs = perLayer;
        }
      } else {
        feeds.input_ids = idsTensor;
      }
      const runStart = performance.now();
      outputs = await session.run(feeds);
      profilePhase(n === 1 ? 'run' : 'prefillRun', runStart);
    } finally {
      disposeCache(cache);
      if (embedded?.location === 'gpu-buffer') embedded.dispose();
      if (perLayer?.location === 'gpu-buffer') perLayer.dispose();
    }

    const next: KvCache = graph.states.map((state) => outputs[state.output] as Tensor);
    const logitsTensor = outputs.logits as Tensor;
    try {
      if (!wantLogits) return { logits: new Float32Array(0), cache: next };
      const vocab = model.architecture.vocabSize;
      // One row per token, or only the last one's (see external-weights.ts).
      const rows = logitsTensor.dims[1]!;
      const readStart = performance.now();
      const last =
        logitsTensor.location === 'gpu-buffer'
          ? await this.readLastRows(logitsTensor, rows, vocab, keep)
          : (logitsTensor.data as Float32Array | Uint16Array).subarray((rows - keep) * vocab, rows * vocab);
      profilePhase(n === 1 ? 'read' : 'prefillRead', readStart);
      const convertStart = performance.now();
      const logits = capLogits(logitsTensor.type === 'float16' ? halfToFloat32(last as Uint16Array) : new Float32Array(last as Float32Array), artifact);
      profilePhase('convert', convertStart);
      // A failed allocation leaves an unusable buffer and the results are garbage, not an error.
      if (this.outOfMemory) {
        disposeCache(next);
        throw this.markFatal('out-of-memory', outOfMemoryText(model));
      }
      return { logits, cache: next };
    } finally {
      logitsTensor.dispose();
    }
  }

  /** Copies the last row of a [1, n, vocab] logits tensor from the GPU. ONNX Runtime shares our device, so its work is queued ahead of this copy. */
  /** Copies the last `keep` rows of a [1, n, vocab] logits tensor from the GPU. ONNX Runtime shares our device, so its work is queued ahead of this copy. */
  private async readLastRows(tensor: Tensor, n: number, vocab: number, keep = 1): Promise<Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>> {
    const loaded = this.requireLoaded();
    const device = loaded.device!;
    const half = tensor.type === 'float16';
    const bytesLength = keep * vocab * (half ? 2 : 4);
    const padded = Math.ceil(bytesLength / 4) * 4;
    if (!loaded.staging || loaded.staging.size < padded) {
      loaded.staging?.destroy();
      loaded.staging = device.createBuffer({ size: padded, usage: BUFFER_MAP_READ | BUFFER_COPY_DST });
    }
    const staging = loaded.staging;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.gpuBuffer as GPUBuffer, (n - keep) * vocab * (half ? 2 : 4), staging, 0, padded);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE_READ);
    const bytes = staging.getMappedRange(0, padded).slice(0, bytesLength);
    staging.unmap();
    return half ? new Uint16Array(bytes) : new Float32Array(bytes);
  }

  /** Empty KV caches, and zeroed linear-attention states. */
  private emptyCache(): KvCache {
    const { graph } = this.requireLoaded();
    if (this.static) {
      // Nothing kept can still be in these buffers after this.
      const st = this.static;
      const encoder = this.requireLoaded().device!.createCommandEncoder();
      st.sets.forEach((set, i) => {
        if (graph.states[i]!.kind !== 'kv') for (const t of set) encoder.clearBuffer(t.gpuBuffer as GPUBuffer);
      });
      this.requireLoaded().device!.queue.submit([encoder.finish()]);
      st.parity = 0;
      return this.staticCurrent();
    }
    const ort = getOrt();
    return graph.states.map(({ type, emptyDims }) => {
      const size = emptyDims.reduce((a, b) => a * b, 1);
      return type === 'float16'
        ? new ort.Tensor('float16', new Uint16Array(size), emptyDims)
        : new ort.Tensor('float32', new Float32Array(size), emptyDims);
    });
  }

  /** (Re)creates the static store with room for `maxLength` positions; what was kept in the old one is dropped. */
  private allocateStatic(maxLength: number): void {
    const loaded = this.requireLoaded();
    const { device, graph, staticPlan, session, model } = loaded;
    if (!device || !staticPlan) return;
    this.dropKept();
    this.dropCheckpoint();
    this.destroyStatic();
    const ort = getOrt();
    const buffer = (dataType: GpuTensorType, dims: number[]) => {
      const bytes = dims.reduce((a, b) => a * b, 1) * BYTES_PER_ELEMENT[dataType];
      const gpuBuffer = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: BUFFER_STORAGE | BUFFER_COPY_SRC | BUFFER_COPY_DST });
      const tensor = ort.Tensor.fromGpuBuffer(gpuBuffer, { dataType, dims, dispose: () => gpuBuffer.destroy() });
      pinned.add(tensor);
      return tensor;
    };
    const sets = graph.states.map((state) => {
      if (state.kind === 'kv') {
        const [batch, heads, , dim] = state.emptyDims as [number, number, number, number];
        return [buffer(state.type, [batch, heads, maxLength, dim])];
      }
      const one = () => buffer(state.type, state.emptyDims);
      return staticPlan.inPlace.has(state.output) ? [one()] : [one(), one()];
    });
    const embedsMeta = session.inputMetadata.find((m) => m.name === 'inputs_embeds');
    const embedsType = embedsMeta?.isTensor && embedsMeta.type === 'float32' ? 'float32' : 'float16';
    // Static stores are only made for plain sessions (not streamed experts).
    const logitsMeta = (session as InferenceSession).outputMetadata.find((m) => m.name === 'logits');
    const logitsType = logitsMeta?.isTensor && logitsMeta.type === 'float16' ? 'float16' : 'float32';
    this.static = {
      maxLength,
      sets,
      parity: 0,
      epoch: this.staticEpoch++,
      mask: buffer('int64', [1, maxLength]),
      maskData: new BigInt64Array(maxLength),
      positions: graph.positionIds ? buffer('int64', graph.positionRows === 1 ? [1, 1] : [graph.positionRows, 1, 1]) : null,
      ids: buffer('int64', [1, 1]),
      embeds: graph.embedsInput ? buffer(embedsType, [1, 1, model.architecture.hiddenSize]) : null,
      logitsToKeep: graph.logitsToKeep ? buffer('int64', []) : null,
      logits: loaded.lastTokenLogits ? buffer(logitsType, [1, 1, model.architecture.vocabSize]) : null,
      seqlens: buffer('int32', [1]),
      totalLength: buffer('int32', []),
      rotary: staticPlan.rotaryFrequencies
        ? {
            rows: buffer('int64', [1, 1]),
            cos: buffer('float32', [1, staticPlan.rotaryFrequencies.length]),
            sin: buffer('float32', [1, staticPlan.rotaryFrequencies.length]),
          }
        : null,
      passes: new Map(),
      windowScratch: null,
      recorded: new Set(),
    };
    const slots = this.windowSlots();
    if (slots.length) this.static.windowScratch = device.createBuffer({ size: Math.max(16, ...slots.map((s) => s.slotBytes)), usage: BUFFER_COPY_SRC | BUFFER_COPY_DST });
    const rows = Math.min(loaded.speculate + 1, LOGIT_ROWS);
    if (staticPlan.capture && loaded.speculate > 0 && loaded.lastTokenLogits && !this.static.logitsToKeep) {
      const tapsMeta = (session as InferenceSession).outputMetadata.find((m) => m.name === loaded.draftFeatures);
      const tapsWidth = loaded.draftFeatures === HIDDEN_OUTPUT ? model.architecture.hiddenSize : tapsMeta?.isTensor ? Number(tapsMeta.shape[2]) : 0;
      const half = staticPlan.rotaryFrequencies?.length ?? 0;
      // A pass per size (guesses vary).
      for (let n = 2; n <= rows; n++) {
        const pass: FixedPass = {
          ids: buffer('int64', [1, n]),
          logits: buffer(logitsType, [1, n, model.architecture.vocabSize]),
          taps: loaded.draft && tapsWidth > 0 ? buffer('float16', [1, n, tapsWidth]) : null,
          positions: graph.positionIds ? buffer('int64', graph.positionRows === 1 ? [1, n] : [graph.positionRows, 1, n]) : null,
          rotary: half ? { rows: buffer('int64', [1, n]), cos: buffer('float32', [n, half]), sin: buffer('float32', [n, half]) } : null,
          embeds: loaded.embeddings ? buffer(embedsType, [1, n, model.architecture.hiddenSize]) : null,
        };
        if (pass.rotary) device.queue.writeBuffer(pass.rotary.rows.gpuBuffer as GPUBuffer, 0, BigInt64Array.from({ length: n }, (_, i) => BigInt(i)));
        this.static.passes.set(n, pass);
      }
    }
    if (this.static.logitsToKeep) device.queue.writeBuffer(this.static.logitsToKeep.gpuBuffer as GPUBuffer, 0, BigInt64Array.of(1n));
    // All ones: attention reads its lengths from their own inputs here (see external-weights.ts: staticAttentionInputs),
    // and what else reads the mask treats it as "these positions are real tokens" (LFM2's convolutions take its last
    // entries for the new tokens), which a zero past the conversation would contradict.
    device.queue.writeBuffer(this.static.mask.gpuBuffer as GPUBuffer, 0, this.static.maskData.fill(1n));
  }

  /** Whether every fixed-size state keeps a window of per-token slots (so checking passes can be cut back). */
  private windowedStates(): boolean {
    const { graph, staticPlan } = this.requireLoaded();
    const pasts = staticPlan?.windows?.pasts;
    return !!pasts && graph.states.every((s) => s.kind === 'kv' || pasts.has(s.name)) && this.windowSlots().every((s) => s.slotBytes % 4 === 0);
  }

  /** The windowed states' buffers (updated in place) and the bytes of one slot. */
  private windowSlots(): { buffer: GPUBuffer; slotBytes: number }[] {
    const { graph, staticPlan } = this.requireLoaded();
    const windows = staticPlan?.windows;
    const st = this.static;
    if (!windows || !st) return [];
    return graph.states.flatMap((state, i) => {
      const tensor = st.sets[i]?.[0];
      if (!windows.pasts.has(state.name) || !tensor || st.sets[i]!.length !== 1) return [];
      const bytes = state.emptyDims.reduce((a, b) => a * b, 1) * BYTES_PER_ELEMENT[state.type];
      return [{ buffer: tensor.gpuBuffer as GPUBuffer, slotBytes: bytes / windows.size }];
    });
  }

  /**
   * After a checking pass of `size` tokens of which the first `kept` + 1 stay: the windowed states after token `kept`
   * (slot kept + W - size) become the current ones (slot W - 1). Queued after the pass's own work.
   */
  private settleWindows(kept: number, size: number): void {
    const { device, staticPlan } = this.requireLoaded();
    const scratch = this.static?.windowScratch;
    const W = staticPlan?.windows?.size;
    if (!device || !scratch || !W || kept >= size - 1) return;
    const encoder = device.createCommandEncoder();
    for (const { buffer, slotBytes } of this.windowSlots()) {
      encoder.copyBufferToBuffer(buffer, (kept + W - size) * slotBytes, scratch, 0, slotBytes);
      encoder.copyBufferToBuffer(scratch, 0, buffer, (W - 1) * slotBytes, slotBytes);
    }
    device.queue.submit([encoder.finish()]);
  }

  private destroyStatic(): void {
    const st = this.static;
    this.static = null;
    if (!st) return;
    st.windowScratch?.destroy();
    const passes = [...st.passes.values()].flatMap((p) => [p.ids, p.logits, p.taps, p.positions, p.embeds, p.rotary?.rows, p.rotary?.cos, p.rotary?.sin]);
    for (const t of [...st.sets.flat(), st.mask, st.positions, st.ids, st.embeds, st.logitsToKeep, st.logits, st.seqlens, st.totalLength, st.rotary?.rows, st.rotary?.cos, st.rotary?.sin, ...passes]) {
      try {
        t?.dispose();
      } catch {
        // Gone with a lost device.
      }
    }
  }

  /** The static store's current state tensors, in graph.states order. */
  private staticCurrent(): KvCache {
    const st = this.static!;
    return st.sets.map((set) => set[set.length === 1 ? 0 : st.parity]!);
  }

  /** Copies a checkpoint's fixed-size states back into the static buffers, and returns them. */
  private restoreStatic(saved: KvCache): KvCache {
    const { device } = this.requireLoaded();
    const current = this.staticCurrent();
    const encoder = device!.createCommandEncoder();
    saved.forEach((tensor, i) => {
      if (tensor === current[i]) return;
      const target = current[i]!.gpuBuffer as GPUBuffer;
      encoder.copyBufferToBuffer(tensor.gpuBuffer as GPUBuffer, 0, target, 0, Math.min(target.size, (tensor.gpuBuffer as GPUBuffer).size));
    });
    device!.queue.submit([encoder.finish()]);
    disposeCache(saved);
    return current;
  }

  /** One forward pass on the static store (see the top of this file); `cache` is its current tensors. */
  private async forwardStatic(
    ids: number[],
    cache: KvCache,
    pastLength: number,
    wantLogits: boolean,
    keep = 1,
  ): Promise<{ logits: Float32Array<ArrayBuffer>; cache: KvCache; taps?: Tensor }> {
    const loaded = this.requireLoaded();
    const { session, embeddings, model, graph, device, staticPlan } = loaded;
    const st = this.static!;
    const ort = getOrt();
    const n = ids.length;
    const total = pastLength + n;
    if (total > st.maxLength) throw new RuntimeError('generation-failed', `The conversation is longer than the ${st.maxLength} positions reserved for it.`);
    // Passes that return taps are never captured single steps; a pass of a fixed size (checking guesses) may be.
    const decode = n === 1 && !this.tapsWanted;
    const fixed = staticPlan!.capture && !decode && wantLogits ? st.passes.get(n) : undefined;
    if (fixed && this.tapsWanted && !fixed.taps) throw new Error('no taps buffer for a captured pass');
    const queue = device!.queue;
    const write = (tensor: Tensor, data: BigInt64Array) => queue.writeBuffer(tensor.gpuBuffer as GPUBuffer, 0, data);

    const feeds: Record<string, Tensor> = { attention_mask: st.mask };
    // On the GPU when captured (ONNX Runtime then reads them there); otherwise the total is read on the CPU.
    queue.writeBuffer(st.seqlens.gpuBuffer as GPUBuffer, 0, Int32Array.of(total - 1));
    feeds[SEQLENS_INPUT] = st.seqlens;
    if (staticPlan!.capture) {
      queue.writeBuffer(st.totalLength.gpuBuffer as GPUBuffer, 0, Int32Array.of(total));
      feeds[TOTAL_LENGTH_INPUT] = st.totalLength;
    } else {
      feeds[TOTAL_LENGTH_INPUT] = new ort.Tensor('int32', Int32Array.of(total), []);
    }
    if (graph.positionIds) {
      const rows = graph.positionRows;
      if (decode) {
        write(st.positions!, new BigInt64Array(rows).fill(BigInt(pastLength)));
        feeds.position_ids = st.positions!;
      } else {
        const positions = BigInt64Array.from({ length: rows * n }, (_, i) => BigInt(pastLength + (i % n)));
        if (fixed?.positions) write(fixed.positions, positions);
        feeds.position_ids = fixed?.positions ?? new ort.Tensor('int64', positions, rows === 1 ? [1, n] : [rows, 1, n]);
      }
    }
    if (st.logitsToKeep) feeds[LOGITS_TO_KEEP_INPUT] = st.logitsToKeep;
    const frequencies = staticPlan!.rotaryFrequencies;
    if (frequencies && st.rotary) {
      const half = frequencies.length;
      const cos = new Float32Array(n * half);
      const sin = new Float32Array(n * half);
      for (let t = 0; t < n; t++)
        for (let j = 0; j < half; j++) {
          const angle = (pastLength + t) * frequencies[j]!;
          cos[t * half + j] = Math.cos(angle);
          sin[t * half + j] = Math.sin(angle);
        }
      if (decode) {
        queue.writeBuffer(st.rotary.cos.gpuBuffer as GPUBuffer, 0, cos);
        queue.writeBuffer(st.rotary.sin.gpuBuffer as GPUBuffer, 0, sin);
        feeds[ROTARY_ROWS_INPUT] = st.rotary.rows;
        feeds[ROTARY_COS_INPUT] = st.rotary.cos;
        feeds[ROTARY_SIN_INPUT] = st.rotary.sin;
      } else if (fixed?.rotary) {
        queue.writeBuffer(fixed.rotary.cos.gpuBuffer as GPUBuffer, 0, cos);
        queue.writeBuffer(fixed.rotary.sin.gpuBuffer as GPUBuffer, 0, sin);
        feeds[ROTARY_ROWS_INPUT] = fixed.rotary.rows;
        feeds[ROTARY_COS_INPUT] = fixed.rotary.cos;
        feeds[ROTARY_SIN_INPUT] = fixed.rotary.sin;
      } else {
        feeds[ROTARY_ROWS_INPUT] = new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, i) => BigInt(i)), [1, n]);
        feeds[ROTARY_COS_INPUT] = new ort.Tensor('float32', cos, [n, half]);
        feeds[ROTARY_SIN_INPUT] = new ort.Tensor('float32', sin, [n, half]);
      }
    }
    const current = this.staticCurrent();
    const fetches: Record<string, Tensor | null> = {};
    const next: KvCache = [];
    graph.states.forEach((state, i) => {
      const set = st.sets[i]!;
      feeds[state.name] = current[i]!;
      const out = set.length === 1 ? set[0]! : set[1 - st.parity]!;
      fetches[state.output] = out;
      next.push(out);
    });
    // With fetches given, ONNX Runtime returns only those outputs: the logits too, in ONNX Runtime's own buffer when the
    // store has none for them (graphs whose output layer reads every position, so their shape varies).
    // Passes of several tokens get up to LOGIT_ROWS rows, in ONNX Runtime's own buffer.
    fetches.logits = decode ? (st.logits ?? null) : (fixed?.logits ?? null);
    if (this.tapsWanted) fetches[loaded.draftFeatures] = fixed?.taps ?? null;

    let idsTensor: Tensor | null = null;
    let embedded: Tensor | null = null;
    if (decode) {
      write(st.ids, BigInt64Array.of(BigInt(ids[0]!)));
      idsTensor = st.ids;
    } else if (fixed) {
      write(fixed.ids, BigInt64Array.from(ids, BigInt));
      idsTensor = fixed.ids;
    } else {
      idsTensor = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]);
    }
    let outputs: InferenceSession.OnnxValueMapType;
    try {
      if (embeddings) {
        // On the CPU when the store's ids are on the GPU: the embeddings session has no int64 kernels of its own.
        const embeddingFeeds: Record<string, Tensor> = { input_ids: decode || fixed ? new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]) : idsTensor };
        if (graph.imageFeatures) {
          const { type, width } = graph.imageFeatures;
          embeddingFeeds[IMAGE_FEATURES_INPUT] =
            type === 'float16' ? new ort.Tensor('float16', new Uint16Array(0), [0, width]) : new ort.Tensor('float32', new Float32Array(0), [0, width]);
        }
        const embedStart = performance.now();
        if (decode) {
          await embeddings.run(embeddingFeeds, { inputs_embeds: st.embeds! });
          feeds.inputs_embeds = st.embeds!;
        } else if (fixed?.embeds) {
          await embeddings.run(embeddingFeeds, { inputs_embeds: fixed.embeds });
          feeds.inputs_embeds = fixed.embeds;
        } else {
          embedded = (await embeddings.run(embeddingFeeds)).inputs_embeds as Tensor;
          feeds.inputs_embeds = embedded;
        }
        profilePhase('embed', embedStart);
        if (graph.idsWithEmbeds) feeds.input_ids = idsTensor;
      } else {
        feeds.input_ids = idsTensor;
      }
      // Captured graphs: one per buffer parity for single steps, and per size for fixed passes.
      const graphId = decode && staticPlan!.capture ? 2 * st.epoch + st.parity : fixed ? FIXED_GRAPHS + (2 * st.epoch + st.parity) * 16 + n : -1;
      const runStart = performance.now();
      if (fixed && !st.recorded.has(graphId)) {
        // The run that records a fixed pass's graph can compute wrong values (seen with Qwen3.5: its replays were
        // right; cause not found), which a recurrent state would keep: record from a saved copy of the fixed-size
        // states, put them back, then run the recorded graph.
        const saved = this.cloneCache(cache);
        await (session as InferenceSession).run(feeds, fetches, { extra: { gpu_graph_id: String(graphId) } });
        st.recorded.add(graphId);
        this.restoreStatic(saved);
      }
      outputs = await (session as InferenceSession).run(feeds, fetches, { extra: { gpu_graph_id: String(graphId) } });
      profilePhase(decode ? 'run' : 'prefillRun', runStart);
    } finally {
      if (embedded?.location === 'gpu-buffer') embedded.dispose();
    }
    if (st.sets.some((set) => set.length === 2)) st.parity = st.parity === 0 ? 1 : 0;
    // The caller's array is spent (as with the other path); `next` is the same buffers.
    cache.length = 0;

    const ownLogits = decode && st.logits ? st.logits : (fixed?.logits ?? null);
    const logitsTensor = (ownLogits ?? outputs.logits) as Tensor;
    const taps = this.tapsWanted ? (outputs[loaded.draftFeatures] as Tensor) : undefined;
    try {
      if (!wantLogits) return { logits: new Float32Array(0), cache: next, taps };
      const vocab = model.architecture.vocabSize;
      const rows = logitsTensor.dims[1]!;
      const readStart = performance.now();
      const last = await this.readLastRows(logitsTensor, rows, vocab, keep);
      profilePhase(decode ? 'read' : 'prefillRead', readStart);
      const convertStart = performance.now();
      const logits = capLogits(logitsTensor.type === 'float16' ? halfToFloat32(last as Uint16Array) : new Float32Array(last as Float32Array), loaded.artifact);
      profilePhase('convert', convertStart);
      if (this.outOfMemory) throw this.markFatal('out-of-memory', outOfMemoryText(model));
      return { logits, cache: next, taps };
    } finally {
      if (!ownLogits) logitsTensor.dispose();
    }
  }
}

/** Speculative decoding's counts (development: read by the bench page). */
export const speculation = { passes: 0, guessed: 0, accepted: 0 };
(globalThis as { __quickInfSpeculation?: typeof speculation }).__quickInfSpeculation = speculation;

/**
 * Guesses per checking pass by default (prompt lookup, see ngramGuesses): lossless (a guess is kept only while the
 * model's own sample equals it), much faster where text repeats earlier text (rewriting, editing, code), about even
 * elsewhere.
 */
const NGRAM_GUESSES = 3;

/** Most recent tokens searched for an earlier occurrence of the last few. */
const NGRAM_WINDOW = 4096;

/**
 * Tokens guessed to follow the conversation (prompt lookup): after the latest earlier occurrence of its last 4 tokens
 * (else 3, else 2), whatever followed it. The longer the match, the likelier the guesses hold, and a guess that doesn't
 * still costs a row of the checking pass: a 2-token match gives 1 guess, 3 tokens 2, 4 tokens up to `count`. Matches
 * shorter than `shortest` tokens are ignored.
 */
export function ngramGuesses(prompt: readonly number[], generated: readonly number[], count: number, shortest = 2): number[] {
  const at = (i: number) => (i < prompt.length ? prompt[i]! : generated[i - prompt.length]!);
  const length = prompt.length + generated.length;
  const from = Math.max(0, length - NGRAM_WINDOW);
  for (const n of [4, 3, 2].filter((m) => m >= shortest)) {
    if (length < n + 1) continue;
    const most = n >= 4 ? count : Math.min(count, n - 1);
    for (let j = length - n - 1; j >= from; j--) {
      let match = true;
      for (let t = 0; t < n && match; t++) match = at(j + t) === at(length - n + t);
      if (!match) continue;
      const guesses: number[] = [];
      for (let t = j + n; t < length && guesses.length < most; t++) guesses.push(at(t));
      return guesses;
    }
  }
  return [];
}

/** The model's final soft cap (OnnxArtifact.logitSoftcap), applied in place to logits read from a graph that leaves it out. */
function capLogits(logits: Float32Array<ArrayBuffer>, artifact: OnnxArtifact): Float32Array<ArrayBuffer> {
  const softcap = artifact.logitSoftcap;
  if (softcap) for (let i = 0; i < logits.length; i++) logits[i] = softcap.cap * Math.tanh((logits[i]! * softcap.multiplier) / softcap.cap);
  return logits;
}

// WebGPU flag values (the DOM typings declare the types but not these constants).
const BUFFER_MAP_READ = 0x1;
const BUFFER_COPY_SRC = 0x4;
const BUFFER_COPY_DST = 0x8;
const BUFFER_STORAGE = 0x80;
const MAP_MODE_READ = 0x1;

/** Dispatches ONNX Runtime encodes before ending a compute pass and submitting (overridable with ?dispatches= in development). */
const PENDING_DISPATCHES = (import.meta.env.DEV && Number(new URLSearchParams(globalThis.location?.search ?? '').get('dispatches'))) || 256;

/** Positions of the static store made at load, for the check run there. */
const VALIDATION_LENGTH = 64;

type GpuTensorType = 'float16' | 'float32' | 'int64' | 'int32';
const BYTES_PER_ELEMENT: Record<GpuTensorType, number> = { float16: 2, float32: 4, int64: 8, int32: 4 };

/**
 * Whether a graph can use a static cache (see the top of this file): every KV output written by GroupQueryAttention
 * (which then shares the buffer with its input), no control flow; and which fixed-size states are updated in place.
 */
function planStatic(states: GraphStates | null): NonNullable<Loaded['staticPlan']> | null {
  if (!states || states.controlFlow) return null;
  const inPlace = new Set<string>();
  let kv = 0;
  for (const [output, op] of states.writers) {
    if (/^present\.\d+\.(key|value)$/.test(output)) {
      if (op !== 'GroupQueryAttention') return null;
      kv++;
    } else if (output.startsWith('present') && IN_PLACE_OPS.has(op)) {
      inPlace.add(output);
    }
  }
  return kv > 0 ? { inPlace, capture: false, rotaryFrequencies: null, windows: null } : null;
}

/** Ops that may write a state into the buffer they read it from (ONNX Runtime checks for the shared buffer). */
const IN_PLACE_OPS = new Set(['CausalConvWithState', 'LinearAttention']);

/** Below this many matching tokens, cutting down the kept cache isn't worth it. */
const MIN_REUSED_TOKENS = 4;

/** Name the rewritten model uses for its external data; mounted as the original file. */
const WEIGHTS_FILE = 'weights.bin';

/**
 * What to hand ONNX Runtime: the model's structure with its weights pointing into
 * the original file, which ONNX Runtime then reads one tensor at a time (straight
 * to the GPU), instead of copying and parsing the whole file in WebAssembly
 * memory. Falls back to the plain file if it can't be rewritten. Weight files the
 * export already keeps outside the graph are mounted under their own names, and
 * read the same way.
 */
function modelInput(
  graph: Uint8Array<ArrayBuffer> | ByteSource,
  externalData: Record<string, Uint8Array<ArrayBuffer> | ByteSource> = {},
  options: Omit<RewriteOptions, 'externalData' | 'quantizeOutput'> & { quantizeOutput?: 0 | 4 | 8 } = {},
): {
  bytes: Uint8Array<ArrayBuffer>;
  externalData?: { path: string; data: Uint8Array<ArrayBuffer> }[];
  /** The output layer computes the last position only (see external-weights.ts). */
  lastTokenLogits: boolean;
  /** Every file the rewritten graph names, for reading weights directly (streamed experts). */
  sources: Record<string, ByteSource>;
  /** What writes the graph's outputs (for static caches); null if it couldn't be rewritten. */
  states: GraphStates | null;
} {
  const source = graph instanceof Uint8Array ? bytesSource(graph) : graph;
  // Files over 2 GB are mounted as windows instead (see external-weights.ts).
  const exported = Object.entries(externalData)
    .filter(([, data]) => (data instanceof Uint8Array ? data.byteLength : data.size) < 2 ** 31)
    .map(([path, data]) => ({ path, data: lazyBytes(data instanceof Uint8Array ? bytesSource(data) : data) }));
  try {
    const sources = Object.fromEntries(
      Object.entries(externalData).map(([path, data]) => [path, data instanceof Uint8Array ? bytesSource(data) : data]),
    );
    const rewritten = externalizeWeights(source, WEIGHTS_FILE, { ...options, quantizeOutput: options.quantizeOutput || undefined, externalData: sources });
    const extra = rewritten.extraFiles.map((f) => ({ path: f.path, data: lazyBytes(f.source) }));
    return {
      bytes: rewritten.graph,
      externalData: [{ path: WEIGHTS_FILE, data: lazyBytes(source) }, ...exported, ...extra],
      lastTokenLogits: rewritten.lastTokenLogits,
      sources: { [WEIGHTS_FILE]: source, ...sources, ...Object.fromEntries(rewritten.extraFiles.map((f) => [f.path, f.source])) },
      states: graphStates(rewritten.graph),
    };
  } catch (err) {
    console.warn('Loading the whole model file instead of reading weights on demand:', err);
    return { bytes: graph instanceof Uint8Array ? graph : source.read(0, source.size), externalData: exported, lastTokenLogits: false, sources: {}, states: null };
  }
}

/**
 * An ONNX Runtime WebGPU session for a graph that isn't a language model (image models' parts): weights read
 * from `graph` tensor by tensor as for language models, without their rewrites; outputs come back to the CPU.
 */
export async function createGraphSession(
  graph: ByteSource,
  /** Null: ONNX Runtime's CPU (WebAssembly) engine. */
  device: GPUDevice | null,
  externalData: Record<string, ByteSource> = {},
  /** Outputs to leave on the GPU (e.g. a KV cache fed back in). */
  gpuOutputs: string[] = [],
  options: {
    /** Sets the graph's MatMulNBits accuracy_level (see external-weights.ts). */
    matMulNBitsAccuracy?: number;
    /** ONNX Runtime's graph optimizations ('all' by default). */
    optimization?: 'disabled' | 'basic' | 'all';
    /** Sizes for symbolic dimensions, fixed at creation. */
    freeDimensions?: Record<string, number>;
    /** Quantize float MatMul/Gemm weights to this many bits (see external-weights.ts). */
    quantizeWeights?: 4 | 8;
    /** Compute a float16 graph in float32 (see external-weights.ts). */
    computeFloat32?: boolean;
  } = {},
): Promise<InferenceSession> {
  const { matMulNBitsAccuracy, optimization = 'all', freeDimensions, quantizeWeights, computeFloat32 } = options;
  const ort = await loadOrt();
  if (device) {
    const adapter = await getGpuAdapter();
    if (adapter) ort.env.webgpu.adapter = adapter;
  }
  const input = modelInput(graph, externalData, { externalizeOnly: true, matMulNBitsAccuracy, quantizeWeights, computeFloat32 });
  try {
    return await ort.InferenceSession.create(input.bytes, {
      externalData: input.externalData,
      ...(device &&
        gpuOutputs.length > 0 && {
          preferredOutputLocation: Object.fromEntries(gpuOutputs.map((name) => [name, 'gpu-buffer' as const])),
        }),
      // Buckets let a run reuse the buffers its earlier layers freed. "lazyRelease" (the chat models' mode) keeps
      // every intermediate of a run alive until it ends: for an image decoder at 512², several GB more at peak
      // (SD Turbo 9.7 → 5.2 GB, Z-Image 12.3 → 7.0 GB), at the cost of about 1–2 GB staying cached between runs.
      executionProviders: device ? [{ name: 'webgpu', device, storageBufferCacheMode: 'bucket' }] : ['wasm'],
      graphOptimizationLevel: optimization,
      ...(freeDimensions && { freeDimensionOverrides: freeDimensions }),
      // Development: ONNX Runtime's per-node log (set in the worker to find a failing kernel).
      ...((globalThis as { __quickInfOrtVerbose?: boolean }).__quickInfOrtVerbose && { logSeverityLevel: 0, logVerbosityLevel: 4 }),
      enableCpuMemArena: false,
    });
  } finally {
    releaseReadBuffer();
  }
}

type KvCache = Tensor[];

/**
 * Development: per-phase time totals of the forward passes, when the worker sets `__quickInfProfile` to an object
 * (phase name → [count, total ms]).
 */
function profilePhase(phase: string, start: number): void {
  if (!import.meta.env.DEV) return;
  const profile = (globalThis as { __quickInfProfile?: Record<string, [number, number]> }).__quickInfProfile;
  if (!profile) return;
  const entry = (profile[phase] ??= [0, 0]);
  entry[0]++;
  entry[1] += performance.now() - start;
}

const LOGITS_TO_KEEP_INPUT = 'num_logits_to_keep';
const IMAGE_FEATURES_INPUT = 'image_features';

/** Failure messages that mean an allocation failed (including the simulated budget's, see device/simulation.ts). */
const OUT_OF_MEMORY = /out of memory|\bOOM\b|memory copy|allocat|GPUOutOfMemoryError/i;

function outOfMemoryText(model: ModelSpec): string {
  return `There isn’t enough free GPU memory to run ${model.name} on this device`;
}

/**
 * Reads the state layout from the graph's inputs and checks it against the
 * catalogue entry (a mismatch means the wrong file or a wrong catalogue entry).
 */
function readGraphShape(session: ModelSession, embeddings: InferenceSession | null, model: ModelSpec): GraphShape {
  const inputs = new Map(session.inputMetadata.map((m) => [m.name, m]));
  const kindOf = (name: string): StateInput['kind'] | null =>
    name.startsWith('past_key_values.')
      ? name.endsWith('.conv_state')
        ? 'conv'
        : name.endsWith('.recurrent_state')
          ? 'recurrent'
          : 'kv'
      : name.startsWith('past_conv.')
        ? 'conv'
        : name.startsWith('past_recurrent.')
          ? 'recurrent'
          : null;
  const states: StateInput[] = [];
  for (const [name, meta] of inputs) {
    const kind = kindOf(name);
    if (!kind) continue;
    if (!meta.isTensor) throw new Error(`${name} is not a tensor`);
    // Batch 1; a KV cache starts with no positions (its third dimension). Some exports leave the KV heads or
    // head size symbolic (Ministral 3 8B's "kv_cache_dim"): those come from the catalogue.
    const { kvHeads: catalogueHeads, headDim: catalogueDim } = model.architecture;
    const emptyDims = meta.shape.map((d, i) =>
      typeof d === 'number' ? d : kind !== 'kv' ? 1 : i === 2 ? 0 : i === 1 ? catalogueHeads : i === 3 ? catalogueDim : 1,
    );
    states.push({
      name,
      output: name.replace('past_key_values.', 'present.').replace('past_', 'present_'),
      kind,
      type: meta.type === 'float16' ? 'float16' : 'float32',
      emptyDims,
    });
  }
  states.sort((a, b) => layerOf(a.name) - layerOf(b.name) || a.name.localeCompare(b.name));
  const kv = states.filter((s) => s.kind === 'kv');
  const linear = states.filter((s) => s.kind === 'conv').length;
  const first = kv[0];
  if (!first) throw new Error('the graph has no KV-cache inputs');
  const kvHeads = first.emptyDims[1]!;
  const headDim = first.emptyDims[3]!;
  const { kvHeads: expectedHeads, headDim: expectedDim } = model.architecture;
  const layers = model.architecture.cacheLayers ?? model.architecture.layers;
  if (kv.length / 2 + linear !== layers || kvHeads !== expectedHeads || headDim !== expectedDim) {
    throw new Error(
      `expected ${layers} layers × ${expectedHeads} KV heads × ${expectedDim}, found ${kv.length / 2} + ${linear} × ${kvHeads} × ${headDim}`,
    );
  }
  const bias = inputs.get(ATTENTION_BIAS_INPUT);
  const attentionBias = bias?.isTensor ? (bias.type === 'float16' ? 'float16' : 'float32') : null;
  const position = inputs.get('position_ids');
  const image = embeddings?.inputMetadata.find((m) => m.name === IMAGE_FEATURES_INPUT);
  return {
    states,
    kvType: first.type,
    kvHeads,
    headDim,
    positionIds: position !== undefined,
    positionRows: position?.isTensor && position.shape.length === 3 ? Number(position.shape[0]) : 1,
    logitsToKeep: inputs.has(LOGITS_TO_KEEP_INPUT),
    embedsInput: inputs.has('inputs_embeds'),
    idsWithEmbeds: inputs.has('inputs_embeds') && inputs.has('input_ids'),
    perLayerInputs: inputs.has('per_layer_inputs'),
    imageFeatures: image?.isTensor
      ? { type: image.type === 'float16' ? 'float16' : 'float32', width: Number(image.shape[1] ?? model.architecture.hiddenSize) }
      : null,
    recurrent: linear > 0,
    attentionBias,
  };
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

/**
 * Frees the cache's GPU buffers. Best effort: after a device loss or crash,
 * disposing throws, and that must not replace the error being reported.
 */
function disposeCache(cache: KvCache): void {
  for (const t of cache) {
    try {
      if (t.location === 'gpu-buffer' && !pinned.has(t)) t.dispose();
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
