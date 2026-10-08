// Main-thread handle to the inference worker. This is the only runtime API the
// UI uses; it never touches ONNX Runtime directly.

import { RuntimeError, type GenerationRequest, type GenerationResult, type LoadResult } from './contract';
import type { CheckResult, RuntimeStatus, WorkerEvent, WorkerRequest, WorkerResults } from './worker/protocol';

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

export class RuntimeClient {
  private readonly worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private readonly listeners = new Set<() => void>();
  private nextId = 1;
  private status: RuntimeStatus = { state: 'idle', modelId: null };

  constructor() {
    this.worker = new Worker(new URL('./worker/runtime.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerEvent>) => this.onEvent(e.data);
    this.worker.onerror = (e) => {
      const error = new RuntimeError('load-failed', `The inference worker crashed: ${e.message || 'unknown error'}`, true);
      for (const p of this.pending.values()) p.reject(error);
      this.pending.clear();
      this.setStatus({ state: 'failed', modelId: this.status.modelId, error: error.toInfo() });
    };
  }

  getStatus = (): RuntimeStatus => this.status;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  check(modelId: string): Promise<CheckResult> {
    return this.request<'check'>({ kind: 'check', modelId });
  }

  load(modelId: string): Promise<LoadResult> {
    return this.request<'load'>({ kind: 'load', modelId });
  }

  generate(request: GenerationRequest, onDelta: (text: string) => void): GenerationHandle {
    const requestId = this.nextId;
    const result = this.request<'generate'>({ kind: 'generate', request }, onDelta);
    return { result, cancel: () => this.post({ kind: 'cancel', targetRequestId: requestId }) };
  }

  unload(): Promise<null> {
    return this.request<'unload'>({ kind: 'unload' });
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
}

let shared: RuntimeClient | null = null;

/** The app's single runtime client, created on first use. */
export function getRuntimeClient(): RuntimeClient {
  shared ??= new RuntimeClient();
  return shared;
}
