// Dedicated inference worker. Owns the device probe, the on-device model store
// (download, verify, read), and the runtime adapter; reports lifecycle status
// and saved-model changes to the main thread. One model is loaded at a time.
//
// Failure handling: ordinary errors (network, integrity, a failed load) leave
// the worker usable, so the user can retry. Fatal errors (GPU device lost,
// WebAssembly crash) mark the status `needsRestart`; the client then replaces
// this worker before the next load.

import { artifactUrl, catalogue, findModel, type ArtifactFile, type ModelSpec } from '../../catalogue/models';
import { runDeviceTest } from '../../device/device-test';
import { probeDevice } from '../../device/probe';
import { ArtifactError, fetchArtifact, verifySha256 } from '../../store/fetch-artifact';
import { dataBytes } from '../../store/data-size';
import { artifactKey, ModelStore } from '../../store/model-store';
import type { ByteSource } from '../onnx/external-weights';
import { RuntimeError, type LoadProgress, type LoadResult, type RuntimeErrorCode, type RuntimeErrorInfo } from '../contract';
import { OnnxWebGpuAdapter } from '../onnx/onnx-adapter';
import type {
  CheckResult,
  DebugScore,
  ModelStorage,
  RuntimeStatus,
  StorageReport,
  WorkerEvent,
  WorkerRequest,
  WorkerResults,
} from './protocol';

const PROGRESS_INTERVAL_MS = 100;
/** Speed estimates before this much data are too noisy to show. */
const MIN_BYTES_FOR_SPEED = 2_000_000;

const store = new ModelStore();
const adapter = new OnnxWebGpuAdapter({ onFatal: (err) => fail(status.modelId, err, true) });
let status: RuntimeStatus = { state: 'idle', modelId: null };
let loading: { modelId: string; promise: Promise<LoadResult>; abort: AbortController } | null = null;
let activeGeneration: number | null = null;
let lastProgressAt = 0;

/** Reads the live status (avoids TypeScript keeping a narrowing across awaits). */
function currentStatus(): RuntimeStatus {
  return status;
}

function post(event: WorkerEvent): void {
  self.postMessage(event);
}

function setStatus(next: RuntimeStatus): void {
  // Once restart is required, only a new worker clears it.
  if (status.needsRestart && !next.needsRestart) return;
  status = next;
  post({ kind: 'status', status });
}

function fail(
  modelId: string | null,
  err: unknown,
  needsRestart = false,
  fallback: RuntimeErrorCode = 'generation-failed',
): void {
  setStatus({ state: 'failed', modelId, error: toInfo(err, fallback), needsRestart: needsRestart || undefined });
}

function reportProgress(modelId: string, progress: LoadProgress): void {
  const now = performance.now();
  const isByteUpdate = progress.phase === 'downloading' && status.state === 'downloading';
  if (isByteUpdate && now - lastProgressAt < PROGRESS_INTERVAL_MS && progress.loadedBytes !== progress.totalBytes) return;
  lastProgressAt = now;
  setStatus({ state: progress.phase, modelId, progress });
}

/** `fallback` classifies unexpected errors: a failed load and a failed reply need different messages. */
function toInfo(err: unknown, fallback: RuntimeErrorCode = 'generation-failed'): RuntimeErrorInfo {
  if (err instanceof RuntimeError) return err.toInfo();
  if (err instanceof ArtifactError) return { code: err.code, message: err.message, retryable: err.code !== 'cancelled' };
  if (err instanceof DOMException && err.name === 'QuotaExceededError') {
    // Any storage write (IndexedDB or OPFS) can hit the quota, even after the up-front space check.
    return { code: 'storage-full', message: err.message || 'Browser storage is full.', retryable: true };
  }
  return { code: fallback, message: err instanceof Error ? err.message || err.name : String(err), retryable: true };
}

function isCancelled(err: unknown): boolean {
  return (err instanceof RuntimeError || err instanceof ArtifactError) && err.code === 'cancelled';
}

function throwIfCancelled(signal: AbortSignal): void {
  if (signal.aborted) throw new RuntimeError('cancelled', 'Loading was cancelled.');
}

function requireModel(modelId: string): ModelSpec {
  const model = findModel(modelId);
  if (!model) throw new RuntimeError('load-failed', `Unknown model ${modelId}`);
  return model;
}

async function check(modelId: string): Promise<CheckResult> {
  const model = requireModel(modelId);
  const device = await probeDevice();
  return { device, compatibility: adapter.checkCompatibility(model, device) };
}

async function load(modelId: string): Promise<LoadResult> {
  if (status.needsRestart && status.error) {
    throw new RuntimeError(status.error.code, status.error.message, status.error.retryable);
  }
  if (status.state === 'ready' && status.modelId === modelId && status.load) return status.load;
  if (loading?.modelId === modelId) return loading.promise;
  if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before loading a model.');
  if (loading) await cancelLoad();

  const abort = new AbortController();
  const promise = doLoad(modelId, abort.signal);
  loading = { modelId, promise, abort };
  try {
    return await promise;
  } finally {
    if (loading?.promise === promise) loading = null;
  }
}

/** Cancels an in-progress load and waits for it to settle. Downloads stop at once; GPU loading stops when it finishes. */
async function cancelLoad(): Promise<void> {
  const current = loading;
  if (!current) return;
  current.abort.abort();
  await current.promise.catch(() => {});
}

async function doLoad(modelId: string, signal: AbortSignal): Promise<LoadResult> {
  try {
    const model = requireModel(modelId);
    setStatus({ state: 'checking', modelId, progress: { phase: 'checking' } });
    await adapter.unload();
    const { device, compatibility } = await check(modelId);
    throwIfCancelled(signal);
    if (!compatibility.compatible || !compatibility.artifact) {
      const code = device.webgpu === 'available' ? 'incompatible-device' : 'webgpu-unavailable';
      throw new RuntimeError(code, compatibility.reasons.join(' '));
    }
    const artifact = compatibility.artifact;

    // Tokenizer first: small, and fails fast on network problems.
    const files = [model.tokenizer, artifact.graph];
    const loadGraph = async (obtained: Obtained, graph: Uint8Array<ArrayBuffer> | ByteSource) => {
      throwIfCancelled(signal);
      const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(obtained.bytes[0]));
      const times = await adapter.load(model, artifact, { graph, tokenizerJson }, (p) => reportProgress(modelId, p));
      return { obtained, ...times };
    };
    const { obtained, sessionMs, validateMs } = (await store.init())
      ? // The model file is read tensor by tensor while ONNX Runtime loads it, never whole,
        // so the lock is held (and the file open) until loading is done.
        await store.withLock(
          model,
          async () => {
            const obtained = await obtainSaved(model, files, signal, artifact.graph);
            const reader = await store.openReader(model, artifact.graph);
            try {
              return await loadGraph(obtained, reader);
            } finally {
              reader.close();
            }
          },
          { signal, onWaiting: () => reportProgress(modelId, { phase: 'waiting' }) },
        )
      : await obtainInMemory(model, files, signal).then((obtained) => loadGraph(obtained, obtained.bytes[1]!));
    if (signal.aborted) {
      await adapter.unload();
      throwIfCancelled(signal);
    }

    const result: LoadResult = {
      modelId,
      artifactId: artifact.id,
      precision: artifact.precision,
      downloadedBytes: obtained.downloadedBytes,
      timings: { ...obtained.timings, sessionMs, validateMs },
    };
    setStatus({ state: 'ready', modelId, load: result });
    return result;
  } catch (err) {
    if (isCancelled(err)) setStatus({ state: 'idle', modelId: null });
    // A failed ONNX Runtime start-up sticks for this worker's lifetime: retry in a fresh one.
    else fail(modelId, err, err instanceof RuntimeError && err.code === 'runtime-files-unavailable', 'load-failed');
    throw err;
  } finally {
    void publishStorage();
  }
}

interface Obtained {
  /** In the order requested (empty for a file left on disk). */
  bytes: Uint8Array<ArrayBuffer>[];
  downloadedBytes: number;
  timings: { downloadMs: number; verifyMs: number; readMs: number };
}

/**
 * Gets files from the model store, downloading (or resuming) and verifying any
 * not yet saved. `leaveOnDisk` is verified but not read into memory (the caller
 * reads it on demand). Runs under the model's lock.
 */
async function obtainSaved(
  model: ModelSpec,
  files: ArtifactFile[],
  signal: AbortSignal,
  leaveOnDisk?: ArtifactFile,
): Promise<Obtained> {
  const modelId = model.id;
  const states = await store.inspect(model, files);
  const missing = states.filter((s) => !s.complete);
  const bytes = new Map<ArtifactFile, Uint8Array<ArrayBuffer>>();
  const timings = { downloadMs: 0, verifyMs: 0, readMs: 0 };
  let downloadedBytes = 0;

  if (missing.length > 0) {
    await store.ensureSpace(missing.reduce((n, s) => n + s.file.sizeBytes - s.savedBytes, 0));
    const totalBytes = files.reduce((n, f) => n + f.sizeBytes, 0);
    const saved = new Map(states.map((s) => [s.file, s.savedBytes]));
    const startSaved = sum(saved.values());
    const resumed = missing.some((s) => s.savedBytes > 0);
    const downloadStart = performance.now();
    for (const { file } of missing) {
      throwIfCancelled(signal);
      downloadedBytes += await store.download(model, file, artifactUrl(model, file), {
        signal,
        onProgress: (savedBytes) => {
          saved.set(file, savedBytes);
          const loadedBytes = sum(saved.values());
          const sessionBytes = loadedBytes - startSaved;
          const seconds = (performance.now() - downloadStart) / 1000;
          const bytesPerSecond = sessionBytes >= MIN_BYTES_FOR_SPEED && seconds > 0 ? sessionBytes / seconds : undefined;
          reportProgress(modelId, { phase: 'downloading', file: file.path, loadedBytes, totalBytes, bytesPerSecond, resumed });
        },
      });
    }
    timings.downloadMs = performance.now() - downloadStart;

    reportProgress(modelId, { phase: 'verifying' });
    const verifyStart = performance.now();
    for (const { file } of missing) {
      const verified = await store.verifyAndCommit(model, file);
      if (file !== leaveOnDisk) bytes.set(file, verified);
    }
    timings.verifyMs = performance.now() - verifyStart;
  }

  const saved = states.filter((s) => s.complete && s.file !== leaveOnDisk);
  if (saved.length > 0) {
    throwIfCancelled(signal);
    reportProgress(modelId, { phase: 'reading' });
    const readStart = performance.now();
    for (const { file } of saved) bytes.set(file, await store.read(model, file));
    timings.readMs = performance.now() - readStart;
  }
  return { bytes: files.map((f) => bytes.get(f) ?? new Uint8Array(0)), downloadedBytes, timings };
}

/** Fallback when this browser can't save models: download into memory every time. */
async function obtainInMemory(model: ModelSpec, files: ArtifactFile[], signal: AbortSignal): Promise<Obtained> {
  const modelId = model.id;
  const totalBytes = files.reduce((n, f) => n + f.sizeBytes, 0);
  const downloadStart = performance.now();
  const bytes: Uint8Array<ArrayBuffer>[] = [];
  let before = 0;
  for (const file of files) {
    bytes.push(
      await fetchArtifact(
        artifactUrl(model, file),
        file,
        (p) => {
          const loadedBytes = before + p.loadedBytes;
          const seconds = (performance.now() - downloadStart) / 1000;
          const bytesPerSecond = loadedBytes >= MIN_BYTES_FOR_SPEED && seconds > 0 ? loadedBytes / seconds : undefined;
          reportProgress(modelId, { phase: 'downloading', file: file.path, loadedBytes, totalBytes, bytesPerSecond });
        },
        signal,
      ),
    );
    before += file.sizeBytes;
  }
  const downloadMs = performance.now() - downloadStart;

  reportProgress(modelId, { phase: 'verifying' });
  const verifyStart = performance.now();
  for (let i = 0; i < files.length; i++) await verifySha256(bytes[i]!, files[i]!);
  const verifyMs = performance.now() - verifyStart;
  return { bytes, downloadedBytes: totalBytes, timings: { downloadMs, verifyMs, readMs: 0 } };
}

function sum(values: Iterable<number>): number {
  let total = 0;
  for (const v of values) total += v;
  return total;
}

async function storageReport(): Promise<StorageReport> {
  const estimate = await navigator.storage?.estimate?.().catch(() => undefined);
  const persisted = await navigator.storage?.persisted?.().catch(() => undefined);
  const base = { usageBytes: estimate?.usage, quotaBytes: estimate?.quota, persisted };
  if (!(await store.init())) return { available: false, models: {}, otherBytes: 0, modelFileBytes: 0, recordBytes: 0, ...base };

  const allRecords = await store.records();
  const recordBytes = dataBytes(allRecords);
  const modelFileBytes = await store.diskBytes();
  const records = new Map(allRecords.map((r) => [r.key, r]));
  const used = new Set<string>();
  const models: Record<string, ModelStorage> = {};
  for (const model of catalogue) {
    const files = [model.tokenizer, ...model.artifacts.map((a) => a.graph)];
    let savedBytes = 0;
    let partial = false;
    const complete = new Set<ArtifactFile>();
    for (const file of files) {
      const key = artifactKey(model, file);
      const r = records.get(key);
      if (!r || r.sha256 !== file.sha256) continue;
      used.add(key);
      if (r.status === 'complete') {
        complete.add(file);
        savedBytes += r.sizeBytes;
      } else if (r.bytesWritten > 0) {
        partial = true;
        savedBytes += r.bytesWritten;
      }
    }
    if (savedBytes === 0) continue;
    const completeArtifacts = complete.has(model.tokenizer)
      ? model.artifacts.filter((a) => complete.has(a.graph)).map((a) => a.id)
      : [];
    models[model.id] = { savedBytes, completeArtifacts, partial };
  }
  let otherBytes = 0;
  for (const [key, r] of records) if (!used.has(key)) otherBytes += r.status === 'complete' ? r.sizeBytes : r.bytesWritten;
  return { available: true, models, otherBytes, modelFileBytes, recordBytes, ...base };
}

async function publishStorage(): Promise<void> {
  try {
    post({ kind: 'storage', report: await storageReport() });
  } catch {
    // Reporting is best effort; loads report their own errors.
  }
}

/** Runs the quick GPU test on the runtime's device. Not while a model is loading or replying. */
async function deviceTest(requestId: number, memoryBytes: number) {
  if (loading || activeGeneration !== null) {
    throw new RuntimeError('busy', 'Wait for the model to finish loading or replying, then run the test.');
  }
  return runDeviceTest({
    memoryBytes,
    // A loaded model already holds GPU memory, so the check would measure the wrong thing.
    skipMemoryReason: status.state === 'ready' ? 'a model is loaded' : undefined,
    onProgress: (step) => post({ kind: 'device-test-progress', requestId, step }),
  });
}

async function deleteModel(modelId: string): Promise<null> {
  const model = requireModel(modelId);
  if (loading?.modelId === modelId) throw new RuntimeError('busy', 'Cancel the load before deleting the model.');
  if (!(await store.init())) return null;
  try {
    await store.deleteModel(model);
  } finally {
    void publishStorage();
  }
  return null;
}

async function generate(requestId: number, msg: Extract<WorkerRequest, { kind: 'generate' }>) {
  if (status.state !== 'ready' || !status.load) throw new RuntimeError('not-loaded', 'Load the model first.');
  if (activeGeneration !== null) throw new RuntimeError('busy', 'A response is already being generated.');
  const ready = status;
  activeGeneration = requestId;
  setStatus({ ...ready, state: 'generating' });
  try {
    return await adapter.generate(msg.request, (text) => post({ kind: 'delta', requestId, text }));
  } finally {
    activeGeneration = null;
    // Don't overwrite a failure (e.g. device lost) reported while generating.
    const state = currentStatus().state;
    if (state === 'generating' || state === 'stopping') setStatus({ ...ready, state: 'ready' });
  }
}

async function handle(msg: WorkerRequest): Promise<void> {
  switch (msg.kind) {
    case 'check':
      return respond(msg.requestId, () => check(msg.modelId));
    case 'load':
      return respond(msg.requestId, () => load(msg.modelId), 'load-failed');
    case 'generate':
      return respond(msg.requestId, () => generate(msg.requestId, msg));
    case 'cancel':
      if (activeGeneration === msg.targetRequestId && status.state === 'generating') {
        adapter.cancel();
        setStatus({ ...status, state: 'stopping' });
      }
      return;
    case 'cancel-load':
      await cancelLoad();
      return;
    case 'reset':
      adapter.reset();
      return;
    case 'unload':
      return respond(msg.requestId, async () => {
        if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before unloading.');
        await cancelLoad();
        await adapter.unload();
        if (!status.needsRestart) setStatus({ state: 'idle', modelId: null });
        return null;
      });
    case 'storage':
      return respond(msg.requestId, storageReport);
    case 'delete-model':
      return respond(msg.requestId, () => deleteModel(msg.modelId));
    case 'device-test':
      return respond(msg.requestId, () => deviceTest(msg.requestId, msg.memoryBytes));
    case 'debug-score':
      if (!import.meta.env.DEV) return;
      return respond(msg.requestId, async () => {
        if (activeGeneration !== null) throw new RuntimeError('busy', 'A response is being generated.');
        const steps: DebugScore['steps'] = [];
        await adapter.scoreContinuation(msg.promptIds, msg.continuation, (step, logits) => {
          let top = 0;
          for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[top]!) top = i;
          steps.push({ logits: (msg.ids[step] ?? []).map((id) => logits[id]!), top: [top, logits[top]!] });
        });
        return { steps };
      });
    case 'debug-fault':
      if (!import.meta.env.DEV) return;
      if (msg.fault === 'worker-crash') {
        setTimeout(() => {
          throw new Error('Simulated worker crash');
        });
      } else {
        adapter.simulateFault(msg.fault);
      }
      return;
  }
}

async function respond<K extends keyof WorkerResults>(
  requestId: number,
  run: () => Promise<WorkerResults[K]>,
  fallback?: RuntimeErrorCode,
) {
  try {
    const value = await run();
    post({ kind: 'result', requestId, value });
  } catch (err) {
    post({ kind: 'error', requestId, error: toInfo(err, fallback) });
  }
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void handle(e.data);
};

// Anything that escapes the handlers above means the worker's state can't be trusted.
self.addEventListener('unhandledrejection', (e) => {
  fail(status.modelId, new RuntimeError('runtime-crashed', `Unexpected runtime error: ${String(e.reason)}`, true), true);
});

post({ kind: 'status', status });
void publishStorage();
