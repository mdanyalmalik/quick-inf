// Runtime contract between the worker and its execution backend (this app's own
// ONNX Runtime adapter, onnx/). Backend-specific objects (sessions, tensors, GPU
// buffers) never cross this boundary.

import type { EngineId, ModelArtifact, ModelSpec } from '../catalogue/models';
import type { ChatTurn } from '../tokenizer/prompt-format';

export type { ChatTurn };
import type { ByteSource } from './onnx/external-weights';

export type LifecycleState =
  | 'idle'
  | 'checking'
  | 'waiting'
  | 'downloading'
  | 'reading'
  | 'verifying'
  | 'loading'
  | 'validating'
  | 'ready'
  | 'generating'
  | 'stopping'
  | 'failed';

export interface CompatibilityReport {
  compatible: boolean;
  /** Best artifact this device can run, if any. */
  artifact?: ModelArtifact;
  reasons: string[];
}

export type RuntimeErrorCode =
  | 'webgpu-unavailable'
  | 'incompatible-device'
  | 'download-failed'
  | 'integrity-mismatch'
  | 'load-failed'
  | 'validation-failed'
  | 'context-overflow'
  | 'generation-failed'
  | 'busy'
  | 'not-loaded'
  /** The user cancelled a load. */
  | 'cancelled'
  /** The GPU device was lost (driver reset, sleep, too much memory); the runtime must restart. */
  | 'device-lost'
  /** The GPU ran out of memory loading or running the model; the runtime restarts. */
  | 'out-of-memory'
  /** The worker or its WebAssembly runtime crashed; the runtime must restart. */
  | 'runtime-crashed'
  /** Not enough browser storage for the download. */
  | 'storage-full'
  /** The model store (OPFS / IndexedDB) can't be used, e.g. in some private windows. */
  | 'storage-unavailable'
  /** Another tab is using these model files (e.g. deleting them). */
  | 'in-use-elsewhere'
  /** The runtime's own files (e.g. ONNX Runtime's WebAssembly) couldn't be fetched from the app's server. */
  | 'runtime-files-unavailable';

export interface RuntimeErrorInfo {
  code: RuntimeErrorCode;
  message: string;
  /** Whether trying the same action again might succeed (e.g. a network failure). */
  retryable: boolean;
}

export class RuntimeError extends Error {
  constructor(
    readonly code: RuntimeErrorCode,
    message: string,
    readonly retryable = false,
  ) {
    super(message);
  }

  toInfo(): RuntimeErrorInfo {
    return { code: this.code, message: this.message, retryable: this.retryable };
  }
}

export type LoadProgress =
  | { phase: 'checking' }
  /** Another tab is downloading the same files; this load continues once it finishes. */
  | { phase: 'waiting' }
  | {
      phase: 'downloading';
      file: string;
      /** Bytes on disk so far, including any saved by an earlier, interrupted download. */
      loadedBytes: number;
      totalBytes: number;
      /** Average for this session's transfer; absent until enough data has arrived. */
      bytesPerSecond?: number;
      /** True when continuing an interrupted download. */
      resumed?: boolean;
    }
  /** Reading files already saved on this device. */
  | { phase: 'reading' }
  | { phase: 'verifying' | 'loading' | 'validating' };

export interface LoadTimings {
  /** 0 when every file was already saved on this device. */
  downloadMs: number;
  /** Reading saved files from device storage. */
  readMs: number;
  verifyMs: number;
  sessionMs: number;
  validateMs: number;
}

export interface LoadResult {
  modelId: string;
  artifactId: string;
  engine: EngineId;
  /** Development: loaded a particular artifact (not recorded as a measurement). */
  engineOverride?: boolean;
  precision: ModelArtifact['precision'];
  /** Bytes downloaded by this load (0 when the files were already saved). */
  downloadedBytes: number;
  timings: LoadTimings;
}

export interface GenerationOptions {
  maxNewTokens: number;
  /** 0 selects greedy decoding. */
  temperature: number;
  topP: number;
  /** 1 disables it. Applied to the last `repetitionWindow` tokens (prompt and reply). */
  repetitionPenalty: number;
  repetitionWindow: number;
  /** Fixed seed for reproducible sampling; random when omitted. */
  seed?: number;
  /** Context budget (prompt + new tokens). */
  contextTokens: number;
}

export interface GenerationRequest {
  turns: ChatTurn[];
  options: GenerationOptions;
}

export type FinishReason = 'stop' | 'length' | 'cancelled';

export interface GenerationStats {
  promptTokens: number;
  /** Of those, how many were already in the KV cache from the previous reply (not prefilled again). */
  reusedPromptTokens: number;
  /** Older turns left out to fit the context budget. */
  droppedTurns: number;
  generatedTokens: number;
  prefillMs: number;
  timeToFirstTokenMs: number;
  decodeMs: number;
  /** Generated tokens per second after the first token. */
  decodeTokensPerSecond: number;
  totalMs: number;
}

export interface GenerationResult {
  text: string;
  tokenIds: number[];
  finishReason: FinishReason;
  stats: GenerationStats;
}

/** Artifact bytes handed to an adapter; where they came from is the store's concern. */
export interface LoadedArtifacts {
  /** The model file: whole in memory (when this browser can't save models), or read on demand from disk. */
  graph: Uint8Array<ArrayBuffer> | ByteSource;
  /** The graph's external weight files, by file name (as the graph refers to them). */
  externalData?: Record<string, Uint8Array<ArrayBuffer> | ByteSource>;
  /** A separate token-embedding graph and its weight files (Qwen3.5). */
  embeddings?: { graph: Uint8Array<ArrayBuffer> | ByteSource; externalData: Record<string, Uint8Array<ArrayBuffer> | ByteSource> };
  /** Parsed tokenizer.json, for the engine that uses this app's own tokenizer (onnx-webgpu). */
  tokenizerJson?: unknown;
  /** Every other file the artifact lists (tokenizer and config files), by repository path. */
  files?: Record<string, Uint8Array<ArrayBuffer>>;
}

export interface RuntimeAdapter {
  /**
   * `files`: the artifact's files from the app's store. `signal` cancels the load.
   */
  load(
    model: ModelSpec,
    artifact: ModelArtifact,
    files: LoadedArtifacts | null,
    onProgress: (p: LoadProgress) => void,
    options?: { signal?: AbortSignal },
  ): Promise<{ sessionMs: number; validateMs: number }>;
  generate(request: GenerationRequest, onDelta: (text: string) => void): Promise<GenerationResult>;
  /** Requests cancellation of the current generation; it takes effect after the current forward pass. */
  cancel(): void;
  /** Drops per-conversation state (the adapter keeps none between requests today). */
  reset(): void;
  unload(): Promise<void>;
}
