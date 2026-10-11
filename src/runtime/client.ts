// Main-thread handle to the inference worker. This is the only runtime API the
// UI uses; it never touches ONNX Runtime directly.
//
// The client owns the worker's life: if the worker crashes or reports that it
// can't continue (GPU lost), the next load replaces it with a fresh one.
// `restart()` is also the last-resort way to stop a reply that won't stop.
//
// Engines keep memory after a model unloads: WebAssembly memory never shrinks,
// and ONNX Runtime also keeps its GPU buffer cache (about 1.4 GB left after
// Qwen3 0.6B on the Mac). So once a worker has loaded
// a model, unloading it or switching to another replaces the worker, which frees
// all of it.

import type { DeviceTestResult, DeviceTestStep } from '../device/device-test';
import { workerName } from '../device/simulation';
import { RuntimeError, type GenerationRequest, type GenerationResult, type LoadResult } from './contract';
import type { CheckResult, DebugFault, DebugScore, RuntimeStatus, StorageReport, WorkerEvent, WorkerRequest, WorkerResults } from './worker/protocol';

type Pending = {
  resolve: (value: unknown) => void;
  reject: (err: RuntimeError) => void;
  onDelta?: (text: string) => void;
  onProgress?: (step: DeviceTestStep) => void;
};

/** A finished load or reply, reported so performance can be recorded (see src/recommend/). */
export type Measurement =
  | { kind: 'load'; modelId: string; ok: true; result: LoadResult }
  | { kind: 'load'; modelId: string; ok: false; errorCode: string }
  | { kind: 'reply'; modelId: string; artifactId: string; result: GenerationResult };

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export interface GenerationHandle {
  result: Promise<GenerationResult>;
  cancel(): void;
}

const IDLE: RuntimeStatus = { state: 'idle', modelId: null };
/** Until the worker reports, assume nothing is saved. */
const NO_STORAGE: StorageReport = { available: true, models: {}, otherBytes: 0, modelFileBytes: 0, recordBytes: 0 };

export class RuntimeClient {
  private worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private readonly listeners = new Set<() => void>();
  private nextId = 1;
  private status: RuntimeStatus = IDLE;
  private storage: StorageReport = NO_STORAGE;
  private persistenceRequested = false;
  private readonly measurementListeners = new Set<(m: Measurement) => void>();
  /** The current worker has loaded a model, so it holds memory only a fresh worker frees (see the top of this file). */
  private ranModel = false;

  constructor() {
    this.worker = this.spawn();
  }

  getStatus = (): RuntimeStatus => this.status;

  /** Which models are saved on this device; updated by the worker whenever that changes. */
  getStorage = (): StorageReport => this.storage;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  check(modelId: string): Promise<CheckResult> {
    return this.request<'check'>({ kind: 'check', modelId });
  }

  /**
   * `options.artifact` (development builds only) forces a particular build, for
   * comparing and testing; such loads and their replies aren't recorded as
   * measurements.
   */
  load(modelId: string, options: { artifact?: string } = {}): Promise<LoadResult> {
    const loaded = this.status.load;
    const alreadyLoaded =
      this.status.state === 'ready' && loaded?.modelId === modelId && (!options.artifact || loaded.artifactId === options.artifact);
    if (this.status.needsRestart || (this.ranModel && !alreadyLoaded && this.idle())) this.restart();
    this.requestPersistence();
    const attempt = () => this.request<'load'>({ kind: 'load', modelId, artifact: options.artifact });
    // The runtime's own files failing to download (a flaky connection) is retried once, in a fresh
    // worker: ONNX Runtime won't start again in the worker where it failed.
    const promise = attempt().catch((err: unknown) => {
      if (!(err instanceof RuntimeError && err.code === 'runtime-files-unavailable')) throw err;
      this.restart();
      return attempt();
    });
    promise.then(
      (result) => {
        if (!result.engineOverride) this.measure({ kind: 'load', modelId, ok: true, result });
      },
      (err: unknown) => {
        const code = err instanceof RuntimeError ? err.code : 'load-failed';
        if (code !== 'cancelled' && !options.artifact) this.measure({ kind: 'load', modelId, ok: false, errorCode: code });
      },
    );
    return promise;
  }

  /** Called after every finished load or reply. */
  onMeasurement(listener: (m: Measurement) => void): () => void {
    this.measurementListeners.add(listener);
    return () => this.measurementListeners.delete(listener);
  }

  /** Runs the quick GPU test in the worker. */
  runDeviceTest(memoryBytes: number, onProgress?: (step: DeviceTestStep) => void): Promise<DeviceTestResult> {
    return this.request<'device-test'>({ kind: 'device-test', memoryBytes }, undefined, onProgress);
  }

  /** Development: teacher-forced logits of the loaded model (see DebugScore). */
  debugScore(promptIds: number[], continuation: number[], ids: number[][]): Promise<DebugScore> {
    // The worker only answers this in development builds.
    if (!import.meta.env.DEV) return Promise.reject(new Error('debugScore is only available in development'));
    return this.request<'debug-score'>({ kind: 'debug-score', promptIds, continuation, ids });
  }

  /** Deletes a model's saved files from this device. */
  deleteModel(modelId: string): Promise<null> {
    return this.request<'delete-model'>({ kind: 'delete-model', modelId });
  }

  /**
   * Asks the browser not to clear saved models when space runs low. Only the
   * page (not a worker) can ask. Refusal is fine: models still save, they're
   * just evictable. Asked once, when the user first loads a model.
   */
  private requestPersistence(): void {
    if (this.persistenceRequested) return;
    this.persistenceRequested = true;
    void navigator.storage
      ?.persist?.()
      .then(() => this.request<'storage'>({ kind: 'storage' }))
      .then((report) => this.setStorage(report))
      .catch(() => {});
  }

  /** Stops an in-progress load. Downloads stop at once; loading onto the GPU stops when that step finishes. */
  cancelLoad(): void {
    this.post({ kind: 'cancel-load' });
  }

  generate(request: GenerationRequest, onDelta: (text: string) => void): GenerationHandle {
    const requestId = this.nextId;
    const loaded = this.status.load;
    const result = this.request<'generate'>({ kind: 'generate', request }, onDelta);
    if (loaded && !loaded.engineOverride) {
      result.then(
        (r) => this.measure({ kind: 'reply', modelId: loaded.modelId, artifactId: loaded.artifactId, result: r }),
        () => {},
      );
    }
    return { result, cancel: () => this.post({ kind: 'cancel', targetRequestId: requestId }) };
  }

  /** Drops the cached state of the last reply, so the next request starts from scratch. */
  forgetLastReply(): void {
    this.post({ kind: 'reset' });
  }

  async unload(): Promise<null> {
    await this.request<'unload'>({ kind: 'unload' });
    if (this.ranModel && this.idle()) this.restart();
    return null;
  }

  /** Nothing is loading, replying or testing (a restart would interrupt nothing). */
  private idle(): boolean {
    return this.pending.size === 0 && ['idle', 'ready', 'failed'].includes(this.status.state);
  }

  /**
   * Terminates the worker and starts a fresh one: stops anything in progress
   * immediately and frees all GPU memory. The model has to be loaded again.
   */
  restart(): void {
    this.ranModel = false;
    this.worker.terminate();
    this.rejectAll(new RuntimeError('cancelled', 'The runtime was restarted.'));
    this.worker = this.spawn();
    this.setStatus(IDLE);
  }

  /** Development only: simulates a failure in the worker to exercise recovery. */
  debugFault(fault: DebugFault): void {
    if (import.meta.env.DEV) this.post({ kind: 'debug-fault', fault });
  }

  private spawn(): Worker {
    const worker = new Worker(new URL('./worker/runtime.worker.ts', import.meta.url), { type: 'module', name: workerName() });
    worker.onmessage = (e: MessageEvent<WorkerEvent>) => {
      if (worker === this.worker) this.onEvent(e.data);
    };
    worker.onerror = (e) => {
      if (worker !== this.worker) return;
      e.preventDefault();
      this.crashed(`The inference worker crashed: ${e.message || 'unknown error'}`);
    };
    worker.onmessageerror = () => {
      if (worker === this.worker) this.crashed('The inference worker sent a message that could not be read.');
    };
    return worker;
  }

  private crashed(message: string): void {
    // Don't leave a half-broken worker running (it may still hold file locks or be downloading).
    this.worker.terminate();
    const error = new RuntimeError('runtime-crashed', message, true);
    this.rejectAll(error);
    this.setStatus({ state: 'failed', modelId: this.status.modelId, error: error.toInfo(), needsRestart: true });
  }

  private rejectAll(error: RuntimeError): void {
    for (const p of this.pending.values()) p.reject(error);
    this.pending.clear();
  }

  private request<K extends keyof WorkerResults>(
    msg: DistributiveOmit<Extract<WorkerRequest, { kind: K }>, 'requestId'>,
    onDelta?: (text: string) => void,
    onProgress?: (step: DeviceTestStep) => void,
  ): Promise<WorkerResults[K]> {
    const requestId = this.nextId++;
    return new Promise<WorkerResults[K]>((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject, onDelta, onProgress });
      this.post({ ...msg, requestId } as WorkerRequest);
    });
  }

  private post(msg: WorkerRequest): void {
    this.worker.postMessage(msg);
  }

  private onEvent(event: WorkerEvent): void {
    switch (event.kind) {
      case 'status':
        this.setStatus(event.status);
        return;
      case 'storage':
        this.setStorage(event.report);
        return;
      case 'delta':
        this.pending.get(event.requestId)?.onDelta?.(event.text);
        return;
      case 'device-test-progress':
        this.pending.get(event.requestId)?.onProgress?.(event.step);
        return;
      case 'result':
        this.pending.get(event.requestId)?.resolve(event.value);
        this.pending.delete(event.requestId);
        return;
      case 'error': {
        const { code, message, retryable } = event.error;
        this.pending.get(event.requestId)?.reject(new RuntimeError(code, message, retryable));
        this.pending.delete(event.requestId);
        return;
      }
    }
  }

  private setStatus(status: RuntimeStatus): void {
    this.status = status;
    if (status.load) this.ranModel = true;
    for (const l of this.listeners) l();
  }

  private measure(m: Measurement): void {
    for (const l of this.measurementListeners) l(m);
  }

  private setStorage(report: StorageReport): void {
    this.storage = report;
    for (const l of this.listeners) l();
  }
}

let shared: RuntimeClient | null = null;

/** The app's single runtime client, created on first use. */
export function getRuntimeClient(): RuntimeClient {
  shared ??= new RuntimeClient();
  return shared;
}
