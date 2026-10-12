// Dedicated inference worker. Owns the device probe, the on-device model store
// (download, verify, read), and the ONNX Runtime adapter; reports lifecycle
// status and saved-model changes to the main thread. One model is loaded at a
// time, from the first of its artifacts this device can run.
//
// Failure handling: ordinary errors (network, integrity, a failed load) leave
// the worker usable, so the user can retry. Fatal errors (GPU device lost, out
// of GPU memory, WebAssembly crash) mark the status `needsRestart`; the client
// then replaces this worker before the next load.

import {
  artifactFiles,
  artifactUrl,
  onnxWeightFiles,
  catalogue,
  findModel,
  type ArtifactFile,
  type ModelArtifact,
  type ModelSpec,
  type OnnxArtifact,
} from '../../catalogue/models';
import { findImageModel, IMAGE_MODELS, imageModelFiles, runsInApp, type ImageModelSpec } from '../../catalogue/image-models';
import { runDeviceTest } from '../../device/device-test';
import { probeDevice } from '../../device/probe';
import { ArtifactError, fetchArtifact, verifySha256 } from '../../store/fetch-artifact';
import { dataBytes } from '../../store/data-size';
import { artifactKey, ModelStore } from '../../store/model-store';
import type { ByteSource } from '../onnx/external-weights';
import { checkCompatibility } from '../compatibility';
import {
  RuntimeError,
  type LoadProgress,
  type LoadResult,
  type RuntimeErrorCode,
  type RuntimeErrorInfo,
} from '../contract';
import { OnnxWebGpuAdapter } from '../onnx/onnx-adapter';
import { ImageEngine } from '../image/image-engine';
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

/** A model whose files the store keeps: chat models and the image models that run. */
interface StoredModel {
  id: string;
  source: ModelSpec['source'];
  files: ArtifactFile[];
  /** Each artifact (precision) and the files it needs. */
  artifacts: { id: string; files: ArtifactFile[] }[];
}

/** The one artifact of an image model (its ONNX files). */
const IMAGE_ARTIFACT = 'onnx';

const STORED: StoredModel[] = [
  ...catalogue.map((m) => ({
    id: m.id,
    source: m.source,
    files: [m.tokenizer, ...m.artifacts.flatMap(onnxWeightFiles)],
    artifacts: m.artifacts.map((a) => ({ id: a.id, files: [m.tokenizer, ...onnxWeightFiles(a)] })),
  })),
  ...IMAGE_MODELS.flatMap((m) =>
    runsInApp(m) ? [{ id: m.id, source: m.source!, files: imageModelFiles(m), artifacts: [{ id: IMAGE_ARTIFACT, files: imageModelFiles(m) }] }] : [],
  ),
];

// Every file the catalogues use, so the store can delete any they no longer do.
const store = new ModelStore(new Map(STORED.flatMap((model) => model.files.map((file) => [artifactKey(model, file), file.sha256] as const))));
const events = { onFatal: (err: RuntimeError) => fail(status.modelId, err, true) };
const adapter = new OnnxWebGpuAdapter(events);
const imageEngine = new ImageEngine();

/** An image model that runs in the app (has files), or undefined. */
function runnableImageModel(modelId: string): ImageModelSpec | undefined {
  const model = findImageModel(modelId);
  return model && runsInApp(model) ? model : undefined;
}

// WebLLM (removed 2026-10-11) kept its downloads in its own OPFS folder: delete them.
void (async () => {
  if (!(await store.init())) return;
  const root = await navigator.storage.getDirectory();
  await root.removeEntry('tvmjs-opfs-store', { recursive: true }).catch(() => {});
})();
let status: RuntimeStatus = { state: 'idle', modelId: null };
let loading: { modelId: string; artifactId?: string; promise: Promise<LoadResult>; abort: AbortController } | null = null;
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

/** Whether this device can run an image model (it needs the GPU features its files use). */
function imageCompatibility(model: ImageModelSpec, device: Awaited<ReturnType<typeof probeDevice>>): CheckResult['compatibility'] {
  if (device.webgpu !== 'available' || !device.gpu) return { compatible: false, reasons: ['WebGPU isn’t available in this browser.'] };
  const missing = (model.requiredGpuFeatures ?? []).filter((f) => !device.gpu!.features.includes(f));
  return missing.length === 0 ? { compatible: true, reasons: [] } : { compatible: false, reasons: [`${model.name} needs GPU feature(s): ${missing.join(', ')}`] };
}

/** What a load would use here. */
async function check(modelId: string): Promise<CheckResult> {
  const image = runnableImageModel(modelId);
  const device = await probeDevice();
  if (image) return { device, compatibility: imageCompatibility(image, device) };
  return { device, compatibility: checkCompatibility(requireModel(modelId), device) };
}

/** `artifactId` (development only) loads that artifact instead of the first this device can run. */
async function load(modelId: string, artifactId?: string): Promise<LoadResult> {
  if (status.needsRestart && status.error) {
    throw new RuntimeError(status.error.code, status.error.message, status.error.retryable);
  }
  const sameArtifact = (r: LoadResult) => !artifactId || r.artifactId === artifactId;
  if (status.state === 'ready' && status.modelId === modelId && status.load && sameArtifact(status.load)) return status.load;
  if (loading?.modelId === modelId && loading.artifactId === artifactId) return loading.promise;
  if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before loading a model.');
  if (loading) await cancelLoad();

  const abort = new AbortController();
  const promise = doLoad(modelId, abort.signal, artifactId);
  loading = { modelId, artifactId, promise, abort };
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

async function doLoad(modelId: string, signal: AbortSignal, artifactOverride?: string): Promise<LoadResult> {
  try {
    const image = runnableImageModel(modelId);
    if (image) {
      setStatus({ state: 'checking', modelId, progress: { phase: 'checking' } });
      const result = await loadImage(image, signal);
      setStatus({ state: 'ready', modelId, load: result });
      return result;
    }
    const model = requireModel(modelId);
    setStatus({ state: 'checking', modelId, progress: { phase: 'checking' } });
    await adapter.unload();
    await imageEngine.unload();
    const device = await probeDevice();
    const compatibility = checkCompatibility(model, device, { onlyArtifact: artifactOverride });
    throwIfCancelled(signal);
    if (!compatibility.compatible || !compatibility.artifact) {
      const code = device.webgpu === 'available' ? 'incompatible-device' : 'webgpu-unavailable';
      throw new RuntimeError(code, compatibility.reasons.join(' '));
    }
    let result = await loadArtifact(model, compatibility.artifact, signal);
    if (artifactOverride !== undefined) result = { ...result, engineOverride: true };
    setStatus({ state: 'ready', modelId, load: result });
    return result;
  } catch (err) {
    if (isCancelled(err)) setStatus({ state: 'idle', modelId: null });
    // A failed ONNX Runtime start-up sticks for this worker's lifetime: retry in a fresh one.
    // So does an out-of-memory one (what ONNX Runtime still holds is unknown).
    else fail(modelId, err, err instanceof RuntimeError && (err.code === 'runtime-files-unavailable' || err.code === 'out-of-memory'), 'load-failed');
    throw err;
  } finally {
    void publishStorage();
  }
}

async function loadArtifact(model: ModelSpec, artifact: ModelArtifact, signal: AbortSignal): Promise<LoadResult> {
  const loaded = await loadOnnx(model, artifact, signal);
  if (signal.aborted) {
    await adapter.unload();
    throwIfCancelled(signal);
  }
  return {
    modelId: model.id,
    artifactId: artifact.id,
    engine: artifact.engine,
    precision: artifact.precision,
    downloadedBytes: loaded.downloadedBytes,
    timings: loaded.timings,
  };
}

type Loaded = Pick<LoadResult, 'downloadedBytes' | 'timings'>;

async function loadOnnx(model: ModelSpec, artifact: OnnxArtifact, signal: AbortSignal): Promise<Loaded> {
  const modelId = model.id;
  const files = artifactFiles(model, artifact);
  // Weight files are read tensor by tensor while ONNX Runtime loads them, never whole.
  const weights = new Set(onnxWeightFiles(artifact));
  const loadGraphs = async (obtained: Obtained, source: (file: ArtifactFile) => Uint8Array<ArrayBuffer> | ByteSource) => {
    throwIfCancelled(signal);
    const small: Record<string, Uint8Array<ArrayBuffer>> = {};
    files.forEach((file, i) => {
      if (!weights.has(file)) small[file.path] = obtained.bytes[i]!;
    });
    const byName = (list: ArtifactFile[]) => Object.fromEntries(list.map((f) => [baseName(f.path), source(f)]));
    const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(small[model.tokenizer.path]));
    const loaded = {
      graph: source(artifact.graph),
      externalData: byName(artifact.externalData ?? []),
      embeddings: artifact.embeddings && { graph: source(artifact.embeddings.graph), externalData: byName(artifact.embeddings.externalData) },
      tokenizerJson,
      files: small,
    };
    const times = await adapter.load(model, artifact, loaded, (p) => reportProgress(modelId, p));
    return { obtained, ...times };
  };
  const { obtained, sessionMs, validateMs } = (await store.init())
    ? // The lock is held (and the files open) until loading is done.
      await store.withLock(
        model,
        async () => {
          const obtained = await obtainSaved(model, files, signal, weights);
          const readers = new Map<ArtifactFile, ByteSource & { close(): void }>();
          try {
            for (const file of weights) readers.set(file, await store.openReader(model, file));
            return await loadGraphs(obtained, (file) => readers.get(file)!);
          } finally {
            for (const reader of readers.values()) reader.close();
          }
        },
        { signal, onWaiting: () => reportProgress(modelId, { phase: 'waiting' }) },
      )
    : await obtainInMemory(model, files, signal).then((obtained) => loadGraphs(obtained, (file) => obtained.bytes[files.indexOf(file)]!));
  return { downloadedBytes: obtained.downloadedBytes, timings: { ...obtained.timings, sessionMs, validateMs } };
}

/** Downloads (or reads) an image model's files and creates its sessions. One model runs at a time. */
async function loadImage(model: ImageModelSpec, signal: AbortSignal): Promise<LoadResult> {
  await adapter.unload();
  const device = await probeDevice();
  const compatibility = imageCompatibility(model, device);
  throwIfCancelled(signal);
  if (!compatibility.compatible) {
    throw new RuntimeError(device.webgpu === 'available' ? 'incompatible-device' : 'webgpu-unavailable', compatibility.reasons.join(' '));
  }
  const stored = { id: model.id, source: model.source! };
  const files = imageModelFiles(model);
  // Graphs and weights are read tensor by tensor while ONNX Runtime loads them, never whole; tokenizers whole.
  const onDisk = new Set(files.filter((f) => !/\.(json|txt)$/.test(f.path)));
  const start = async (obtained: Obtained, source: (file: ArtifactFile) => Uint8Array<ArrayBuffer> | ByteSource) => {
    throwIfCancelled(signal);
    reportProgress(model.id, { phase: 'loading' });
    const { sessionMs } = await imageEngine.load(model, {
      bytes: (file) => obtained.bytes[files.indexOf(file)]!,
      source: (file) => {
        const s = source(file);
        return s instanceof Uint8Array ? { size: s.byteLength, read: (o: number, n: number) => s.subarray(o, o + n) } : s;
      },
    });
    return { obtained, sessionMs };
  };
  const { obtained, sessionMs } = (await store.init())
    ? await store.withLock(
        stored,
        async () => {
          const obtained = await obtainSaved(stored, files, signal, onDisk);
          const readers = new Map<ArtifactFile, ByteSource & { close(): void }>();
          try {
            for (const file of onDisk) readers.set(file, await store.openReader(stored, file));
            return await start(obtained, (file) => readers.get(file) ?? obtained.bytes[files.indexOf(file)]!);
          } finally {
            for (const reader of readers.values()) reader.close();
          }
        },
        { signal, onWaiting: () => reportProgress(model.id, { phase: 'waiting' }) },
      )
    : await obtainInMemory(stored, files, signal).then((obtained) => start(obtained, (file) => obtained.bytes[files.indexOf(file)]!));
  if (signal.aborted) {
    await imageEngine.unload();
    throwIfCancelled(signal);
  }
  return {
    modelId: model.id,
    artifactId: IMAGE_ARTIFACT,
    engine: 'onnx-webgpu',
    precision: model.precision ?? 'fp16',
    downloadedBytes: obtained.downloadedBytes,
    timings: { ...obtained.timings, sessionMs, validateMs: 0 },
  };
}

function baseName(path: string): string {
  return path.slice(path.lastIndexOf('/') + 1);
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
  model: Pick<StoredModel, 'id' | 'source'>,
  files: ArtifactFile[],
  signal: AbortSignal,
  leaveOnDisk: ReadonlySet<ArtifactFile> = new Set(),
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
      // Files left on disk are hashed in slices rather than read whole.
      const verified = await store.verifyAndCommit(model, file, { keepBytes: !leaveOnDisk.has(file) });
      if (!leaveOnDisk.has(file)) bytes.set(file, verified);
    }
    timings.verifyMs = performance.now() - verifyStart;
  }

  const saved = states.filter((s) => s.complete && !leaveOnDisk.has(s.file));
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
async function obtainInMemory(model: Pick<StoredModel, 'id' | 'source'>, files: ArtifactFile[], signal: AbortSignal): Promise<Obtained> {
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
  for (const model of STORED) {
    const files = model.files;
    let savedBytes = 0;
    // ONNX downloads resume byte for byte (Range), so everything saved counts.
    let resumableBytes = 0;
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
        resumableBytes += r.sizeBytes;
      } else if (r.bytesWritten > 0) {
        partial = true;
        savedBytes += r.bytesWritten;
        resumableBytes += r.bytesWritten;
      }
    }
    const completeArtifacts = model.artifacts.filter((a) => a.files.every((f) => complete.has(f))).map((a) => a.id);
    if (savedBytes === 0) continue;
    models[model.id] = { savedBytes, resumableBytes, completeArtifacts, partial };
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
  const model = STORED.find((m) => m.id === modelId);
  if (!model) throw new RuntimeError('load-failed', `Unknown model ${modelId}`);
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

async function generateImage(requestId: number, msg: Extract<WorkerRequest, { kind: 'generate-image' }>) {
  if (status.state !== 'ready' || !status.load || imageEngine.modelId !== status.modelId) {
    throw new RuntimeError('not-loaded', 'Load the model first.');
  }
  if (activeGeneration !== null) throw new RuntimeError('busy', 'An image is already being generated.');
  const ready = status;
  activeGeneration = requestId;
  setStatus({ ...ready, state: 'generating' });
  try {
    return await imageEngine.generate(msg.request, (progress) => post({ kind: 'image-progress', requestId, progress }));
  } finally {
    activeGeneration = null;
    const state = currentStatus().state;
    if (state === 'generating' || state === 'stopping') setStatus({ ...ready, state: 'ready' });
  }
}

async function handle(msg: WorkerRequest): Promise<void> {
  switch (msg.kind) {
    case 'check':
      return respond(msg.requestId, () => check(msg.modelId));
    case 'load':
      return respond(
        msg.requestId,
        () => load(msg.modelId, import.meta.env.DEV ? msg.artifact : undefined),
        'load-failed',
      );
    case 'generate':
      return respond(msg.requestId, () => generate(msg.requestId, msg));
    case 'generate-image':
      return respond(msg.requestId, () => generateImage(msg.requestId, msg));
    case 'cancel':
      if (activeGeneration === msg.targetRequestId && status.state === 'generating') {
        adapter.cancel();
        imageEngine.cancel();
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
        await imageEngine.unload();
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
