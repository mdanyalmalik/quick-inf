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

import type { DeviceTestResult, DeviceTestSoFar, DeviceTestStep, MemoryCheck } from '../device/device-test';
import { findImageModel, imageModelBytes } from '../catalogue/image-models';
import { findModel, weightBytes } from '../catalogue/models';
import { findSpeechModel, speechModelBytes } from '../catalogue/speech-models';
import { simulation, workerName } from '../device/simulation';
import { backgroundDownloadsSupported, backgroundRegistration, downloadInBackground } from './background-download';
import { Breadcrumbs, interruptedDownload, killedForMemory, type Breadcrumb, type RiskyStep } from './breadcrumbs';
import { RuntimeError, type GenerationRequest, type GenerationResult, type LoadResult } from './contract';
import type { ImageGenerationProgress, ImageGenerationRequest, ImageGenerationResult } from './image/image-engine';
import type { SpeechGenerationProgress, SpeechGenerationRequest, SpeechGenerationResult } from './speech/speech-engine';
import type { CheckResult, DebugFault, DebugScore, RuntimeStatus, StorageReport, WorkerEvent, WorkerRequest, WorkerResults } from './worker/protocol';

type Pending = {
  resolve: (value: unknown) => void;
  reject: (err: RuntimeError) => void;
  onDelta?: (text: string) => void;
  onProgress?: (step: DeviceTestStep, sofar: DeviceTestSoFar) => void;
  onImageProgress?: (progress: ImageGenerationProgress) => void;
  onSpeechProgress?: (progress: SpeechGenerationProgress) => void;
  onMemoryStep?: (heldBytes: number, nextBytes: number) => void;
};

/** A finished load or reply, reported so performance can be recorded (see src/recommend/). */
export type Measurement =
  | { kind: 'load'; modelId: string; ok: true; result: LoadResult }
  | { kind: 'load'; modelId: string; ok: false; errorCode: string }
  | { kind: 'reply'; modelId: string; artifactId: string; result: GenerationResult }
  | { kind: 'reply-failed'; modelId: string; errorCode: string }
  | { kind: 'image'; modelId: string; settings: ImageGenerationRequest['settings']; ok: boolean; errorCode?: string };

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export interface GenerationHandle {
  result: Promise<GenerationResult>;
  cancel(): void;
}

const IDLE: RuntimeStatus = { state: 'idle', modelId: null };
/**
 * A worker that sends nothing at all for this long while loading or replying is stuck (a hung GPU job, a deadlock):
 * it's replaced, with an error, instead of leaving a spinner forever. Generous, as loading a very large model can
 * be silent for a while.
 */
const WATCHDOG_MS = (simulation()?.watchdogSeconds ?? 300) * 1000;
const BUSY_STATES = new Set(['checking', 'downloading', 'reading', 'verifying', 'loading', 'validating', 'generating', 'stopping']);
/** A load's states: the screen is kept awake through them, and leaving the page during them is noticed. */
const LOAD_STATES = new Set(['checking', 'waiting', 'downloading', 'verifying', 'reading', 'loading', 'validating']);
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
  private readonly breadcrumbs = new Breadcrumbs();
  /** Risky steps the browser killed the page during, on a previous visit (see breadcrumbs.ts). */
  private crashes: Breadcrumb[];

  /** The worker crashed and was terminated: the next request starts a fresh one (it would never answer). */
  private dead = false;
  /** When the worker last sent anything (see WATCHDOG_MS). */
  private lastHeard = Date.now();
  /**
   * Decides whether a model runs with streamed experts, and with how much memory for them (set by the page from its
   * memory limit, recommend/memory-evidence.ts); unset, models load whole.
   */
  streamingPlanner: ((modelId: string) => { cacheBytes: number } | null) | null = null;
  /** Stops a background download (see background-download.ts) when the load is cancelled. */
  private backgroundAbort: AbortController | null = null;
  /** The person left the page while this model was loading (see resumeAfterLeaving). */
  private leftDuringLoad: string | null = null;
  /** A download interrupted while the page was in the background, to carry on when it's back. */
  private resumeWhenVisible: string | null = null;
  private wakeLock: { release(): Promise<void> } | null = null;
  private wakeLockPending = false;

  constructor() {
    const leftovers = this.breadcrumbs.takeLeftovers();
    this.crashes = leftovers.filter(killedForMemory);
    this.worker = this.spawn();
    setInterval(() => this.watchdog(), Math.min(15_000, WATCHDOG_MS / 2));
    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', () => this.onVisibility());
    // Registered early, so a download started in the background on an earlier visit can report back.
    if (backgroundDownloadsSupported()) void backgroundRegistration();
    // This tab was closed or reloaded mid-download (phones discard background pages): carry on where it stopped.
    const interrupted = leftovers.filter((c) => c.ownTab && interruptedDownload(c)).at(-1)?.modelId;
    if (interrupted) setTimeout(() => this.resumeAfterLeaving(interrupted), 1500);
  }

  private onVisibility(): void {
    this.breadcrumbs.note();
    if (document.visibilityState === 'hidden') {
      if (LOAD_STATES.has(this.status.state) && this.status.modelId) this.leftDuringLoad = this.status.modelId;
      return;
    }
    this.updateWakeLock();
    const modelId = this.resumeWhenVisible;
    this.resumeWhenVisible = null;
    if (modelId) this.resumeAfterLeaving(modelId);
  }

  /** Starts `modelId`'s load again (it resumes from the saved part) once the page is in front. */
  private resumeAfterLeaving(modelId: string): void {
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      this.resumeWhenVisible = modelId;
      return;
    }
    if (this.status.state !== 'idle' && this.status.state !== 'failed') return;
    if (this.status.state === 'failed' && this.status.modelId !== modelId) return;
    this.load(modelId).catch(() => {}); // failures arrive through the runtime status
  }

  /** Keeps the screen on while a model downloads and loads (a locked phone pauses the page). */
  private updateWakeLock(): void {
    const wanted = LOAD_STATES.has(this.status.state) && typeof document !== 'undefined' && document.visibilityState === 'visible';
    if (!wanted) {
      void this.wakeLock?.release().catch(() => {});
      this.wakeLock = null;
      return;
    }
    const wakeLock = (navigator as Navigator & { wakeLock?: { request(type: 'screen'): Promise<EventTarget & { release(): Promise<void> }> } })
      .wakeLock;
    if (this.wakeLock || this.wakeLockPending || !wakeLock) return;
    this.wakeLockPending = true;
    wakeLock
      .request('screen')
      .then((lock) => {
        this.wakeLock = lock;
        // The browser releases it when the page is hidden; onVisibility asks again.
        lock.addEventListener('release', () => {
          if (this.wakeLock === lock) this.wakeLock = null;
        });
        if (!LOAD_STATES.has(this.status.state)) this.updateWakeLock();
      })
      .catch(() => {}) // refused (battery saver, no permission): the download just pauses if the screen locks
      .finally(() => {
        this.wakeLockPending = false;
      });
  }

  private watchdog(): void {
    const busy = this.pending.size > 0 && BUSY_STATES.has(this.status.state);
    const limit = this.silenceLimit();
    if (busy && Date.now() - this.lastHeard > limit) {
      this.crashed(`The model stopped responding (nothing for ${Math.round(limit / 1000)} s), so it was stopped.`);
    }
  }

  /**
   * How long the worker may stay silent. Building a model on the GPU sends nothing until it's done, and takes
   * about 3.5 s per GB on a fast computer, so a load gets WATCHDOG_MS or 100 s per GB of weights, whichever is longer.
   */
  private silenceLimit(): number {
    if (this.status.state !== 'loading' && this.status.state !== 'validating') return WATCHDOG_MS;
    const id = this.status.modelId;
    const bytes = id ? modelWeightBytes(id) : 0;
    return Math.max(WATCHDOG_MS, (bytes / 1e9) * 100_000);
  }

  /** Steps that killed the page last time (each reported once per visit until dismissed). */
  getCrashes = (): readonly Breadcrumb[] => this.crashes;

  dismissCrashes(): void {
    this.crashes = [];
    for (const l of this.listeners) l();
  }

  /** Notes `step` until `promise` settles, so a page killed meanwhile is noticed next time. */
  private track<T>(step: RiskyStep, modelId: string | undefined, promise: Promise<T>, bytes?: number): Promise<T> {
    const end = this.breadcrumbs.begin(step, modelId, bytes);
    promise.then(end, end);
    return promise;
  }

  getStatus = (): RuntimeStatus => this.status;

  /** Which models are saved on this device; updated by the worker whenever that changes. */
  getStorage = (): StorageReport => this.storage;

  /** Whether the worker has reported what's saved yet (before that, nothing looks saved). */
  hasStorageReport = (): boolean => this.storage !== NO_STORAGE;

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
    const streamExperts = this.streamingPlanner?.(modelId) ?? undefined;
    const attempt = () => this.request<'load'>({ kind: 'load', modelId, artifact: options.artifact, streamExperts });
    this.leftDuringLoad = typeof document !== 'undefined' && document.visibilityState === 'hidden' ? modelId : null;
    const run = async () => {
      if (!options.artifact && !alreadyLoaded) await this.downloadAheadInBackground(modelId);
      // The runtime's own files failing to download (a flaky connection) is retried once, in a fresh
      // worker: ONNX Runtime won't start again in the worker where it failed.
      return attempt().catch((err: unknown) => {
        if (!(err instanceof RuntimeError && err.code === 'runtime-files-unavailable')) throw err;
        this.restart();
        return attempt();
      });
    };
    const promise = this.track('load', modelId, run());
    // A download that failed because the person left the page (phones pause background pages) carries on when
    // they're back: the saved part is kept, so it resumes rather than restarts.
    promise.catch((err: unknown) => {
      if (this.leftDuringLoad !== modelId || !(err instanceof RuntimeError) || err.code !== 'download-failed' || !err.retryable) return;
      this.leftDuringLoad = null;
      if (document.visibilityState === 'hidden') this.resumeWhenVisible = modelId;
      else setTimeout(() => this.resumeAfterLeaving(modelId), 1000);
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
  runDeviceTest(onProgress?: (step: DeviceTestStep, sofar: DeviceTestSoFar) => void): Promise<DeviceTestResult> {
    return this.track('benchmark', undefined, this.request<'device-test'>({ kind: 'device-test' }, undefined, onProgress));
  }

  /**
   * Holds more and more GPU memory until it runs out or `targetBytes` is held (see measureMemory). Each step is noted
   * before it's taken, so if the browser closes the page instead of failing the allocation (phones), the next visit
   * knows how far it got (see getCrashes).
   */
  runMemoryCheck(targetBytes: number, onStep?: (heldBytes: number) => void): Promise<MemoryCheck> {
    const requestId = this.nextId;
    const promise = this.request<'memory-check'>({ kind: 'memory-check', targetBytes });
    this.pending.get(requestId)!.onMemoryStep = (heldBytes, nextBytes) => {
      this.breadcrumbs.note({ heldBytes, bytes: nextBytes });
      onStep?.(heldBytes);
    };
    // A fresh worker afterwards: it frees everything the check took on the GPU (Chrome may keep freed memory pooled
    // in the device), and replaces a GPU device the check lost by running out.
    const done = promise.then((result) => {
      if (result.status !== 'skipped') this.restart();
      return result;
    });
    return this.track('benchmark', undefined, done);
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
    this.backgroundAbort?.abort();
    this.resumeWhenVisible = null;
    this.post({ kind: 'cancel-load' });
  }

  /**
   * Where the browser can (Background Fetch), downloads the model's missing files with its download manager, which
   * carries on if the person leaves the page or closes the tab. Returns when they're in the browser's cache (the
   * load then moves them into the model store), or straight away where it can't, leaving the download to the load.
   */
  private async downloadAheadInBackground(modelId: string): Promise<void> {
    if (!backgroundDownloadsSupported()) return;
    const plan = await this.request<'download-plan'>({ kind: 'download-plan', modelId }).catch(() => null);
    if (!plan || !plan.canSave || plan.files.length === 0) return;
    const abort = new AbortController();
    this.backgroundAbort = abort;
    const started = performance.now();
    const totalBytes = plan.files.reduce((n, f) => n + f.sizeBytes, 0);
    const report = (loadedBytes: number) => {
      const seconds = (performance.now() - started) / 1000;
      const bytesPerSecond = loadedBytes > 8e6 && seconds > 1 ? loadedBytes / seconds : undefined;
      this.setStatus({
        state: 'downloading',
        modelId,
        progress: { phase: 'downloading', file: plan.files[0]!.path, loadedBytes, totalBytes, bytesPerSecond, background: true },
      });
    };
    report(0);
    const name = findModel(modelId)?.name ?? findImageModel(modelId)?.name ?? findSpeechModel(modelId)?.name ?? modelId;
    try {
      const outcome = await downloadInBackground(modelId, `Downloading ${name}`, plan, (loaded) => report(loaded), abort.signal);
      if (outcome === 'aborted') {
        this.setStatus(IDLE);
        throw new RuntimeError('cancelled', 'The download was cancelled.');
      }
      // 'failed' or 'unavailable': the load downloads what's still missing itself.
    } finally {
      if (this.backgroundAbort === abort) this.backgroundAbort = null;
    }
  }

  generate(request: GenerationRequest, onDelta: (text: string) => void): GenerationHandle {
    const requestId = this.nextId;
    const loaded = this.status.load;
    const result = this.track('reply', loaded?.modelId, this.request<'generate'>({ kind: 'generate', request }, onDelta));
    if (loaded && !loaded.engineOverride) {
      result.then(
        (r) => this.measure({ kind: 'reply', modelId: loaded.modelId, artifactId: loaded.artifactId, result: r }),
        (err: unknown) => {
          const code = err instanceof RuntimeError ? err.code : 'generation-failed';
          if (code !== 'cancelled') this.measure({ kind: 'reply-failed', modelId: loaded.modelId, errorCode: code });
        },
      );
    }
    return { result, cancel: () => this.post({ kind: 'cancel', targetRequestId: requestId }) };
  }

  /** An image from the loaded image model; `onProgress` reports each step (with a preview on several-step runs). */
  generateImage(
    request: ImageGenerationRequest,
    onProgress: (progress: ImageGenerationProgress) => void,
  ): { result: Promise<ImageGenerationResult>; cancel: () => void } {
    const requestId = this.nextId++;
    const result = this.track(
      'image',
      this.status.modelId ?? undefined,
      new Promise<ImageGenerationResult>((resolve, reject) => {
        this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject, onImageProgress: onProgress });
        this.post({ kind: 'generate-image', requestId, request });
      }),
    );
    const modelId = this.status.modelId;
    if (modelId) {
      result.then(
        () => this.measure({ kind: 'image', modelId, settings: request.settings, ok: true }),
        (err: unknown) => {
          const code = err instanceof RuntimeError ? err.code : 'generation-failed';
          if (code !== 'cancelled') this.measure({ kind: 'image', modelId, settings: request.settings, ok: false, errorCode: code });
        },
      );
    }
    return { result, cancel: () => this.post({ kind: 'cancel', targetRequestId: requestId }) };
  }

  /** Speech from the loaded speech model; `onProgress` hands over the audio a chunk of text at a time. */
  generateSpeech(
    request: SpeechGenerationRequest,
    onProgress: (progress: SpeechGenerationProgress) => void,
  ): { result: Promise<SpeechGenerationResult>; cancel: () => void } {
    const requestId = this.nextId++;
    const result = this.track(
      'speech',
      this.status.modelId ?? undefined,
      new Promise<SpeechGenerationResult>((resolve, reject) => {
        this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject, onSpeechProgress: onProgress });
        this.post({ kind: 'generate-speech', requestId, request });
      }),
    );
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
    this.dead = false;
    this.worker = this.spawn();
    this.setStatus(IDLE);
  }

  /** Stops the worker for good (before deleting everything the app stores, which it may hold open). */
  shutdown(): void {
    this.worker.terminate();
    this.rejectAll(new RuntimeError('cancelled', 'The runtime was stopped.'));
    this.setStatus(IDLE);
  }

  /** Development only: simulates a failure in the worker to exercise recovery. */
  debugFault(fault: DebugFault): void {
    if (import.meta.env.DEV) this.post({ kind: 'debug-fault', fault });
  }

  private spawn(): Worker {
    const worker = new Worker(new URL('./worker/runtime.worker.ts', import.meta.url), { type: 'module', name: workerName() });
    worker.onmessage = (e: MessageEvent<WorkerEvent>) => {
      if (worker !== this.worker) return;
      this.lastHeard = Date.now();
      this.onEvent(e.data);
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
    this.dead = true;
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
    onProgress?: (step: DeviceTestStep, sofar: DeviceTestSoFar) => void,
  ): Promise<WorkerResults[K]> {
    const requestId = this.nextId++;
    return new Promise<WorkerResults[K]>((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject, onDelta, onProgress });
      this.post({ ...msg, requestId } as WorkerRequest);
    });
  }

  private post(msg: WorkerRequest): void {
    // A new request starts the clock afresh (the worker may have been idle for a long time).
    if (this.pending.size <= 1) this.lastHeard = Date.now();
    if (this.dead) {
      // Nothing is loaded in a fresh worker, so a reply fails at once ("load the model first") instead of hanging.
      this.dead = false;
      this.worker = this.spawn();
    }
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
      case 'image-progress':
        this.pending.get(event.requestId)?.onImageProgress?.(event.progress);
        return;
      case 'speech-progress':
        this.pending.get(event.requestId)?.onSpeechProgress?.(event.progress);
        return;
      case 'device-test-progress':
        this.pending.get(event.requestId)?.onProgress?.(event.step, event.sofar);
        return;
      case 'memory-check-progress':
        this.pending.get(event.requestId)?.onMemoryStep?.(event.heldBytes, event.nextBytes);
        return;
      case 'result':
        this.pending.get(event.requestId)?.resolve(event.value);
        this.pending.delete(event.requestId);
        return;
      case 'simulated-kill':
        // As a mobile OS ends a tab that uses too much memory: no clean-up, the page starts again. Nothing more from
        // the worker is handled (a real kill runs no more code; a late result would clear the step's breadcrumb).
        if (!import.meta.env.DEV) return;
        this.breadcrumbs.freeze();
        this.worker.terminate();
        this.onEvent = () => {};
        location.reload();
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
    const phaseChanged = status.state !== this.status.state;
    this.status = status;
    if (status.load) this.ranModel = true;
    // A page closed during a load is judged by how far the load got (see breadcrumbs.ts).
    if (phaseChanged) {
      this.breadcrumbs.note({ phase: status.state });
      this.updateWakeLock();
    }
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

/** A model's weights (chat: its largest artifact; image and speech: all their files). */
function modelWeightBytes(modelId: string): number {
  const chat = findModel(modelId);
  if (chat) return Math.max(0, ...chat.artifacts.map(weightBytes));
  const image = findImageModel(modelId);
  if (image) return imageModelBytes(image);
  const speech = findSpeechModel(modelId);
  return speech ? speechModelBytes(speech) : 0;
}
