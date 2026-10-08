// Messages between the main thread (RuntimeClient) and the inference worker.

import type { DeviceReport } from '../../device/probe';
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
  | { kind: 'reset' }
  | { kind: 'unload'; requestId: number };

export interface WorkerResults {
  check: CheckResult;
  load: LoadResult;
  generate: GenerationResult;
  unload: null;
}

export type WorkerEvent =
  | { kind: 'status'; status: RuntimeStatus }
  | { kind: 'delta'; requestId: number; text: string }
  | { kind: 'result'; requestId: number; value: WorkerResults[keyof WorkerResults] }
  | { kind: 'error'; requestId: number; error: RuntimeErrorInfo };
