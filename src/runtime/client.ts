// Main-thread handle to the inference worker. This is the only runtime API the
// UI uses; it never touches ONNX Runtime directly.
//
// The client owns the worker's life: if the worker crashes or reports that it
// can't continue (GPU lost), the next load replaces it with a fresh one.
// `restart()` is also the last-resort way to stop a reply that won't stop.

import { RuntimeError, type GenerationRequest, type GenerationResult, type LoadResult } from './contract';
import type { CheckResult, DebugFault, RuntimeStatus, StorageReport, WorkerEvent, WorkerRequest, WorkerResults } from './worker/protocol';

type Pending = {
  resolve: (value: unknown) => void;
  reject: (err: RuntimeError) => void;
  onDelta?: (text: string) => void;
};

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export interface GenerationHandle {
  result: Promise<GenerationResult>;
  cancel(): void;
}

const IDLE: RuntimeStatus = { state: 'idle', modelId: null };
/** Until the worker reports, assume nothing is saved. */
const NO_STORAGE: StorageReport = { available: true, models: {}, otherBytes: 0 };

export class RuntimeClient {
  private worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private readonly listeners = new Set<() => void>();
  private nextId = 1;
  private status: RuntimeStatus = IDLE;
  private storage: StorageReport = NO_STORAGE;
  private persistenceRequested = false;

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

  load(modelId: string): Promise<LoadResult> {
    if (this.status.needsRestart) this.restart();
    this.requestPersistence();
    return this.request<'load'>({ kind: 'load', modelId });
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
    const result = this.request<'generate'>({ kind: 'generate', request }, onDelta);
    return { result, cancel: () => this.post({ kind: 'cancel', targetRequestId: requestId }) };
  }

  unload(): Promise<null> {
    return this.request<'unload'>({ kind: 'unload' });
  }

  /**
   * Terminates the worker and starts a fresh one: stops anything in progress
   * immediately and frees all GPU memory. The model has to be loaded again.
   */
  restart(): void {
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
    const worker = new Worker(new URL('./worker/runtime.worker.ts', import.meta.url), { type: 'module' });
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
  ): Promise<WorkerResults[K]> {
    const requestId = this.nextId++;
    return new Promise<WorkerResults[K]>((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject, onDelta });
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
    for (const l of this.listeners) l();
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
