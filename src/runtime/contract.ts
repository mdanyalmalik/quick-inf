// Runtime contract shared by every execution backend. The ONNX Runtime adapter
// is the first implementation; a custom WebGPU engine would implement the same
// shape. Backend-specific objects (sessions, tensors, GPU buffers) never cross
// this boundary.

import type { ModelArtifact, ModelSpec } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import type { ChatTurn } from '../tokenizer/chatml';

export type { ChatTurn };

export type LifecycleState =
  | 'idle'
  | 'checking'
  | 'downloading'
  | 'verifying'
  | 'loading'
  | 'validating'
  | 'ready'
  | 'generating'
  | 'stopping'
  | 'failed';

export interface AdapterCapabilities {
  id: string;
  formats: ModelArtifact['format'][];
  precisions: ModelArtifact['precision'][];
  /** 'next-step': a cancel takes effect after the in-flight forward pass finishes. */
  cancellation: 'immediate' | 'next-step';
  streaming: boolean;
}

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
  | 'not-loaded';

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
  | { phase: 'downloading'; file: string; loadedBytes: number; totalBytes: number }
  | { phase: 'verifying' | 'loading' | 'validating' };

export interface LoadTimings {
  downloadMs: number;
  verifyMs: number;
  sessionMs: number;
  validateMs: number;
}

export interface LoadResult {
  modelId: string;
  artifactId: string;
  precision: ModelArtifact['precision'];
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
  graph: Uint8Array<ArrayBuffer>;
  tokenizerJson: unknown;
}

export interface RuntimeAdapter {
  describe(): AdapterCapabilities;
  checkCompatibility(model: ModelSpec, device: DeviceReport): CompatibilityReport;
  load(
    model: ModelSpec,
    artifact: ModelArtifact,
    files: LoadedArtifacts,
    onProgress: (p: LoadProgress) => void,
  ): Promise<{ sessionMs: number; validateMs: number }>;
  generate(request: GenerationRequest, onDelta: (text: string) => void): Promise<GenerationResult>;
  /** Requests cancellation of the current generation; see `describe().cancellation`. */
  cancel(): void;
  /** Drops per-conversation state (the adapter keeps none between requests today). */
  reset(): void;
  unload(): Promise<void>;
}
