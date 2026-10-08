// Messages between the main thread (RuntimeClient) and the inference worker.

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
  /** Bytes saved on this device: complete files plus resumable partial downloads. */
  savedBytes: number;
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
  /** Whole-origin estimate from the browser (all storage, not just models). */
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
  | { kind: 'load'; requestId: number; modelId: string }
  | { kind: 'generate'; requestId: number; request: GenerationRequest }
  | { kind: 'cancel'; targetRequestId: number }
  | { kind: 'cancel-load' }
  | { kind: 'reset' }
  | { kind: 'unload'; requestId: number }
  | { kind: 'storage'; requestId: number }
  | { kind: 'delete-model'; requestId: number; modelId: string }
  /** Development builds only; ignored in production. */
  | { kind: 'debug-fault'; fault: DebugFault };

/** Simulated failures for testing recovery (see OnnxWebGpuAdapter.simulateFault). */
export type DebugFault = AdapterFault | 'worker-crash';

export interface WorkerResults {
  check: CheckResult;
  load: LoadResult;
  generate: GenerationResult;
  unload: null;
  storage: StorageReport;
  'delete-model': null;
}

export type WorkerEvent =
  | { kind: 'status'; status: RuntimeStatus }
  /** Sent when saved models change (and once at startup). */
  | { kind: 'storage'; report: StorageReport }
  | { kind: 'delta'; requestId: number; text: string }
  | { kind: 'result'; requestId: number; value: WorkerResults[keyof WorkerResults] }
  | { kind: 'error'; requestId: number; error: RuntimeErrorInfo };
