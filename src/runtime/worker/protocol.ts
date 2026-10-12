import type { ImageGenerationProgress, ImageGenerationRequest, ImageGenerationResult } from '../image/image-engine';
// Messages between the main thread (RuntimeClient) and the inference worker.

import type { DeviceTestResult, DeviceTestStep } from '../../device/device-test';
import type { DeviceReport } from '../../device/probe';
import type { AdapterFault } from '../onnx/onnx-adapter';
import type {
  CompatibilityReport,
  GenerationRequest,
  GenerationResult,
  LifecycleState,
  LoadProgress,
  LoadResult,
  RuntimeErrorInfo,
} from '../contract';

export interface RuntimeStatus {
  state: LifecycleState;
  /** Model being loaded, loaded, or that failed; null when idle. */
  modelId: string | null;
  progress?: LoadProgress;
  error?: RuntimeErrorInfo;
  load?: LoadResult;
  /** The worker can no longer run models (GPU lost, crash); the client must restart it before loading. */
  needsRestart?: boolean;
}

export interface ModelStorage {
  /** Bytes saved on this device: complete files plus partial downloads. */
  savedBytes: number;
  /** Of those, what a resumed download keeps. */
  resumableBytes: number;
  /** Artifacts whose files (weights and tokenizer) are all saved and verified. */
  completeArtifacts: string[];
  /** An interrupted download can be resumed. */
  partial: boolean;
}

export interface StorageReport {
  /** False when this browser or window can't save models; every load downloads again. */
  available: boolean;
  /** By model id. Models with nothing saved are absent. */
  models: Record<string, ModelStorage>;
  /** Saved files no catalogue model uses (e.g. an older revision). */
  otherBytes: number;
  /** Exact size of all model files on disk, including partial downloads. */
  modelFileBytes: number;
  /** Size of the download records (IndexedDB `artifacts`), measured as data, not on-disk bytes. */
  recordBytes: number;
  /** Whole-origin estimate from the browser (all storage, not just models). Not shown: Chrome reports a fixed quota and lags after deletions. */
  usageBytes?: number;
  quotaBytes?: number;
  /** Whether the browser promised not to clear this site's storage under pressure. */
  persisted?: boolean;
}

export interface CheckResult {
  device: DeviceReport;
  compatibility: CompatibilityReport;
}

export type WorkerRequest =
  | { kind: 'check'; requestId: number; modelId: string }
  /** `artifact`: development only, see RuntimeClient.load. */
  | { kind: 'load'; requestId: number; modelId: string; artifact?: string }
  | { kind: 'generate'; requestId: number; request: GenerationRequest }
  /** An image from the loaded image model (see src/runtime/image). */
  | { kind: 'generate-image'; requestId: number; request: ImageGenerationRequest }
  | { kind: 'cancel'; targetRequestId: number }
  | { kind: 'cancel-load' }
  | { kind: 'reset' }
  | { kind: 'unload'; requestId: number }
  | { kind: 'storage'; requestId: number }
  | { kind: 'delete-model'; requestId: number; modelId: string }
  /** Quick GPU test; `memoryBytes` is how much GPU memory the memory check tries to hold. */
  | { kind: 'device-test'; requestId: number; memoryBytes: number }
  /** Development: teacher-forced logits (see DebugScore). */
  | { kind: 'debug-score'; requestId: number; promptIds: number[]; continuation: number[]; ids: number[][] }
  /** Development builds only; ignored in production. */
  | { kind: 'debug-fault'; fault: DebugFault };

/** Simulated failures for testing recovery (see OnnxWebGpuAdapter.simulateFault). */
export type DebugFault = AdapterFault | 'worker-crash';

export interface WorkerResults {
  check: CheckResult;
  load: LoadResult;
  generate: GenerationResult;
  'generate-image': ImageGenerationResult;
  unload: null;
  storage: StorageReport;
  'delete-model': null;
  'device-test': DeviceTestResult;
  'debug-score': DebugScore;
}

/**
 * Development: logits when feeding `promptIds` then `continuation` one token at a
 * time. Per step: the logits at the requested ids, and the model's top token.
 */
export interface DebugScore {
  steps: { logits: number[]; top: [id: number, logit: number] }[];
}

export type WorkerEvent =
  | { kind: 'status'; status: RuntimeStatus }
  /** Sent when saved models change (and once at startup). */
  | { kind: 'storage'; report: StorageReport }
  | { kind: 'delta'; requestId: number; text: string }
  | { kind: 'image-progress'; requestId: number; progress: ImageGenerationProgress }
  | { kind: 'device-test-progress'; requestId: number; step: DeviceTestStep }
  | { kind: 'result'; requestId: number; value: WorkerResults[keyof WorkerResults] }
  | { kind: 'error'; requestId: number; error: RuntimeErrorInfo };
