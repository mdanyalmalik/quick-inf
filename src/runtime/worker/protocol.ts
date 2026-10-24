import type { ImageGenerationProgress, ImageGenerationRequest, ImageGenerationResult } from '../image/image-engine';
import type { SpeechGenerationProgress, SpeechGenerationRequest, SpeechGenerationResult } from '../speech/speech-engine';
// Messages between the main thread (RuntimeClient) and the inference worker.

import type { DeviceTestResult, DeviceTestSoFar, DeviceTestStep, MemoryCheck } from '../../device/device-test';
import type { DeviceReport } from '../../device/probe';
import type { AdapterFault } from '../onnx/onnx-adapter';
import type { PeerState } from '../../distributed/peer-stage';
import type { DeviceInfo, PipelineMessage } from '../../distributed/protocol';
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
  /** Parts of it kept from running them for another device (sharing a model: src/distributed), not counted above. */
  sharedPartBytes: number;
}

export interface StorageReport {
  /** False when this browser or window can't save models; every load downloads again. */
  available: boolean;
  /** By model id. Models with nothing saved are absent. */
  models: Record<string, ModelStorage>;
  /** Saved files no catalogue model uses (e.g. an older revision). */
  otherBytes: number;
  /** Exact size of all model files on disk, including partial downloads and parts kept from sharing. */
  modelFileBytes: number;
  /** Size of the download records (IndexedDB `artifacts`), measured as data, not on-disk bytes. */
  recordBytes: number;
  /** Whole-origin estimate from the browser (all storage, not just models). Not shown: Chrome reports a fixed quota and lags after deletions. */
  usageBytes?: number;
  quotaBytes?: number;
  /** Whether the browser promised not to clear this site's storage under pressure. */
  persisted?: boolean;
}

export interface DownloadPlan {
  /** Files not yet saved (a partly saved file is listed whole). */
  files: { url: string; path: string; sizeBytes: number }[];
  /** False when this browser can't save models (nothing to download ahead of a load). */
  canSave: boolean;
}

export interface CheckResult {
  device: DeviceReport;
  compatibility: CompatibilityReport;
}

export type WorkerRequest =
  | { kind: 'check'; requestId: number; modelId: string }
  /** `artifact`: development only, see RuntimeClient.load. */
  /** `streamExperts`: run a mixture-of-experts model with streamed experts and this much memory for them. */
  /** `share`: run it across these devices (src/distributed): this one's layers here, theirs on them. */
  | { kind: 'load'; requestId: number; modelId: string; artifact?: string; streamExperts?: { cacheBytes: number }; share?: ShareRequest }
  /** A pipeline message from another device (the page carries them: workers have no WebRTC). */
  | { kind: 'pipe'; message: PipelineMessage }
  /** Run part of another device's model: say hello to it as device `id`, then load and run what it assigns. */
  | { kind: 'join'; requestId: number; id: string; device: DeviceInfo }
  | { kind: 'generate'; requestId: number; request: GenerationRequest }
  /** An image from the loaded image model (see src/runtime/image). */
  | { kind: 'generate-image'; requestId: number; request: ImageGenerationRequest }
  /** Speech from the loaded speech model (see src/runtime/speech); audio arrives as `speech-progress` events. */
  | { kind: 'generate-speech'; requestId: number; request: SpeechGenerationRequest }
  | { kind: 'cancel'; targetRequestId: number }
  | { kind: 'cancel-load' }
  | { kind: 'reset' }
  | { kind: 'unload'; requestId: number }
  | { kind: 'storage'; requestId: number }
  /** What loading a model here would still download (for downloading in the background; see client.ts). */
  | { kind: 'download-plan'; requestId: number; modelId: string }
  | { kind: 'delete-model'; requestId: number; modelId: string }
  /** Quick GPU test; `memoryBytes` is how much GPU memory the memory check tries to hold. */
  | { kind: 'device-test'; requestId: number }
  | { kind: 'memory-check'; requestId: number; targetBytes: number }
  /** Development: teacher-forced logits (see DebugScore). */
  | { kind: 'debug-score'; requestId: number; promptIds: number[]; continuation: number[]; ids: number[][] }
  /** Development builds only; ignored in production. */
  | { kind: 'debug-fault'; fault: DebugFault };

/** Simulated failures for testing recovery (see OnnxWebGpuAdapter.simulateFault). */
export type DebugFault = AdapterFault | 'worker-crash';

/** The devices to share a load with: this one, and those connected (by their pipeline id). */
export interface ShareRequest {
  host: DeviceInfo;
  devices: { id: string; device: DeviceInfo }[];
}

export interface WorkerResults {
  join: null;
  check: CheckResult;
  load: LoadResult;
  generate: GenerationResult;
  'generate-image': ImageGenerationResult;
  'generate-speech': SpeechGenerationResult;
  unload: null;
  storage: StorageReport;
  'download-plan': DownloadPlan;
  'delete-model': null;
  'device-test': DeviceTestResult;
  'memory-check': MemoryCheck;
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
  | { kind: 'speech-progress'; requestId: number; progress: SpeechGenerationProgress }
  | { kind: 'device-test-progress'; requestId: number; step: DeviceTestStep; sofar: DeviceTestSoFar }
  /** The memory check holds `heldBytes` and is about to hold `nextBytes`. */
  | { kind: 'memory-check-progress'; requestId: number; heldBytes: number; nextBytes: number }
  | { kind: 'result'; requestId: number; value: WorkerResults[keyof WorkerResults] }
  /** A pipeline message for another device. */
  | { kind: 'pipe'; message: PipelineMessage }
  /** What this device is doing for the model it joined. */
  | { kind: 'peer-state'; state: PeerState }
  | { kind: 'error'; requestId: number; error: RuntimeErrorInfo }
  /** Development (`?sim=…,kill`): the simulated device ran out of memory, so the page must die as a mobile browser's would. */
  | { kind: 'simulated-kill' };
