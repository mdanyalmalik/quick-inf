// Dedicated inference worker. Owns the device probe, the on-device model stores
// (ONNX files: download, verify, read; WebLLM's own files: sizes, checks,
// deletion), and the runtime adapters; reports lifecycle status and saved-model
// changes to the main thread. One model is loaded at a time, by the engine its
// artifact names (see EngineId).
//
// Engine choice: the first artifact this device can run (WebLLM before ONNX). If
// WebLLM fails to run a model here (GPU or driver errors, not network or
// storage), that's remembered for this GPU and browser version and the load
// continues with the next artifact, so the user gets the ONNX engine instead.
//
// Failure handling: ordinary errors (network, integrity, a failed load) leave
// the worker usable, so the user can retry. Fatal errors (GPU device lost,
// WebAssembly crash) mark the status `needsRestart`; the client then replaces
// this worker before the next load.

import { deviceEngines } from '../../catalogue/device-catalogue';
import {
  artifactBytes,
  artifactFiles,
  artifactUrl,
  onnxWeightFiles,
  catalogue,
  findModel,
  type ArtifactFile,
  type EngineId,
  type ModelArtifact,
  type ModelSpec,
  type OnnxArtifact,
  type WebLlmArtifact,
} from '../../catalogue/models';
import { runDeviceTest } from '../../device/device-test';
import { probeDevice } from '../../device/probe';
import { ArtifactError, fetchArtifact, verifySha256 } from '../../store/fetch-artifact';
import { dataBytes } from '../../store/data-size';
import { artifactKey, ModelStore } from '../../store/model-store';
import { webllmUrls, WebLlmStore } from '../../store/webllm-store';
import type { ByteSource } from '../onnx/external-weights';
import { checkCompatibility } from '../compatibility';
import { clearEngineFailure, deviceEnv, failedEngines, isEngineFailure, recordEngineFailure } from '../engine-failures';
import {
  RuntimeError,
  type LoadProgress,
  type LoadResult,
  type RuntimeAdapter,
  type RuntimeErrorCode,
  type RuntimeErrorInfo,
} from '../contract';
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

// Every file the catalogue uses, so the store can delete any it no longer does.
const store = new ModelStore(
  new Map(
    catalogue.flatMap((model) =>
      allFiles(model).map((file) => [artifactKey(model, file), file.sha256] as const),
    ),
  ),
);
const webllmStore = new WebLlmStore();
const events = { onFatal: (err: RuntimeError) => fail(status.modelId, err, true) };
const onnxAdapter = new OnnxWebGpuAdapter(events);
let webllmAdapter: Promise<RuntimeAdapter> | null = null;
/** The adapter of the loaded (or last loaded) model. */
let adapter: RuntimeAdapter = onnxAdapter;

function adapterFor(engine: EngineId): Promise<RuntimeAdapter> {
  switch (engine) {
    case 'onnx-webgpu':
      return Promise.resolve(onnxAdapter);
    case 'webllm':
      // Imported when first used: the library is 6.6 MB. A failed download sticks for this
      // worker's lifetime (the browser remembers failed module loads): retry in a fresh one.
      webllmAdapter ??= import('../webllm/webllm-adapter').then(
        (m) => new m.WebLlmAdapter(events),
        (err: unknown) => {
          webllmAdapter = null;
          throw new RuntimeError('runtime-files-unavailable', `WebLLM's code could not be downloaded: ${err instanceof Error ? err.message : String(err)}`, true);
        },
      );
      return webllmAdapter;
  }
}

function onnxArtifacts(model: ModelSpec): OnnxArtifact[] {
  return model.artifacts.filter((a): a is OnnxArtifact => a.engine === 'onnx-webgpu');
}

function webllmArtifacts(model: ModelSpec): WebLlmArtifact[] {
  return model.artifacts.filter((a): a is WebLlmArtifact => a.engine === 'webllm');
}

/** Every ONNX file of `model` (the ONNX store's files). */
function allFiles(model: ModelSpec): ArtifactFile[] {
  return [model.tokenizer, ...onnxArtifacts(model).flatMap(onnxWeightFiles)];
}

/** WebLLM builds are locked per repository revision, like ONNX files are per model. */
function lockKey(artifact: WebLlmArtifact) {
  return { source: { host: 'huggingface' as const, repo: artifact.build.repo, revision: artifact.build.revision } };
}

// WebLLM files the catalogue no longer uses (old builds, removed models) are deleted at start-up.
void (async () => {
  if (await store.init()) await webllmStore.removeUnlisted(catalogue.flatMap((m) => webllmArtifacts(m).flatMap((a) => webllmUrls(a).map((u) => u.url)))).catch(() => {});
})();
let status: RuntimeStatus = { state: 'idle', modelId: null };
let loading: { modelId: string; engine?: EngineId; artifactId?: string; promise: Promise<LoadResult>; abort: AbortController } | null = null;
let activeGeneration: number | null = null;
let lastProgressAt = 0;
/** Development: the next WebLLM load fails as if the GPU were lost (tests the fallback). */
let simulateWebLlmFailure = false;

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

/** What a load would use here, skipping engines that failed for this model on this device. */
async function check(modelId: string): Promise<CheckResult> {
  const model = requireModel(modelId);
  const device = await probeDevice();
  const avoid = await failedEngines(deviceEnv(device), modelId);
  if (!(await store.init())) avoid.add('webllm');
  return { device, compatibility: checkCompatibility(model, device, { avoid, only: deviceOnlyEngine() }) };
}

/** The one engine this kind of device is limited to, if it is (phones: the ONNX engine). */
function deviceOnlyEngine(): EngineId | undefined {
  const engines = deviceEngines();
  return engines.length === 1 ? engines[0] : undefined;
}

/** `engine` (development only) loads the artifact with a different engine than the catalogue names, to compare them. */
async function load(modelId: string, engine?: EngineId, artifactId?: string): Promise<LoadResult> {
  if (status.needsRestart && status.error) {
    throw new RuntimeError(status.error.code, status.error.message, status.error.retryable);
  }
  const sameEngine = (r: LoadResult) => (!engine || r.engine === engine) && (!artifactId || r.artifactId === artifactId);
  if (status.state === 'ready' && status.modelId === modelId && status.load && sameEngine(status.load)) return status.load;
  if (loading?.modelId === modelId && loading.engine === engine && loading.artifactId === artifactId) return loading.promise;
  if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before loading a model.');
  if (loading) await cancelLoad();

  const abort = new AbortController();
  const promise = doLoad(modelId, abort.signal, engine, artifactId);
  loading = { modelId, engine, artifactId, promise, abort };
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

async function doLoad(modelId: string, signal: AbortSignal, engineOverride?: EngineId, artifactOverride?: string): Promise<LoadResult> {
  try {
    const model = requireModel(modelId);
    setStatus({ state: 'checking', modelId, progress: { phase: 'checking' } });
    await adapter.unload();
    const device = await probeDevice();
    const env = deviceEnv(device);
    // Engines known not to work for this model here, and WebLLM when its files can't be saved.
    const forced = engineOverride !== undefined || artifactOverride !== undefined;
    const avoid = forced ? new Set<EngineId>() : await failedEngines(env, modelId);
    if (!(await store.init())) avoid.add('webllm');
    // Each kind of device has its engines (src/catalogue/device-catalogue.ts): phones only the ONNX one.
    const engines = deviceEngines();
    if (!forced && !model.artifacts.some((a) => engines.includes(a.engine))) {
      throw new RuntimeError('incompatible-device', `On phones and tablets this app runs only models its own engine supports, and ${model.name} isn’t one of them yet.`);
    }
    // A strict limit (not `avoid`, which yields when nothing else fits).
    const only = engineOverride ?? (forced ? undefined : deviceOnlyEngine());
    const first = checkCompatibility(model, device, { avoid, only, onlyArtifact: artifactOverride });
    throwIfCancelled(signal);
    if (!first.compatible || !first.artifact) {
      const code = device.webgpu === 'available' ? 'incompatible-device' : 'webgpu-unavailable';
      throw new RuntimeError(code, first.reasons.join(' '));
    }

    let result: LoadResult;
    try {
      result = await loadArtifact(model, first.artifact, signal);
      if (forced) await clearEngineFailure(env, first.artifact.engine, modelId);
    } catch (err) {
      const info = toInfo(err, 'load-failed');
      if (forced || isCancelled(err) || first.artifact.engine === 'onnx-webgpu' || !isEngineFailure(info)) throw err;
      // This engine can't run the model on this device: remember that, and use the next one.
      await recordEngineFailure(env, first.artifact.engine, modelId, info);
      avoid.add(first.artifact.engine);
      const next = checkCompatibility(model, device, { avoid, only });
      if (!next.compatible || !next.artifact) throw err;
      result = { ...(await loadArtifact(model, next.artifact, signal)), fallback: { from: first.artifact.engine, reason: info.message } };
    }
    if (forced) result = { ...result, engineOverride: true };
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

/** Loads one artifact with its engine (no fallback). */
async function loadArtifact(model: ModelSpec, artifact: ModelArtifact, signal: AbortSignal): Promise<LoadResult> {
  adapter = await adapterFor(artifact.engine);
  const loaded = artifact.engine === 'webllm' ? await loadWebLlm(model, artifact, signal) : await loadOnnx(model, artifact, signal);
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

function baseName(path: string): string {
  return path.slice(path.lastIndexOf('/') + 1);
}

/**
 * WebLLM downloads (or reads) and loads its files itself. Around that: the
 * build's lock (one tab downloads at a time), a space check, and after a
 * download, a check of every weight shard against its MD5.
 */
async function loadWebLlm(model: ModelSpec, artifact: WebLlmArtifact, signal: AbortSignal): Promise<Loaded> {
  const modelId = model.id;
  if (simulateWebLlmFailure) {
    simulateWebLlmFailure = false;
    throw new RuntimeError('device-lost', 'Simulated WebLLM failure (development).', true);
  }
  return store.withLock(
    lockKey(artifact),
    async () => {
      const before = await webllmStore.state(artifact, await webllmStore.list());
      const total = artifactBytes(model, artifact);
      if (!before.complete) await store.ensureSpace(total - before.resumableBytes);
      throwIfCancelled(signal);

      const start = performance.now();
      let downloadEnd = start;
      const { sessionMs, validateMs } = await adapter.load(
        model,
        artifact,
        null,
        (p) => {
          if (p.phase === 'downloading') downloadEnd = performance.now();
          reportProgress(modelId, p);
        },
        { signal },
      );
      const downloadMs = before.complete ? 0 : downloadEnd - start;

      let verifyMs = 0;
      if (!(await webllmStore.isVerified(artifact))) {
        reportProgress(modelId, { phase: 'verifying' });
        const verifyStart = performance.now();
        const bad = await webllmStore.verify(artifact);
        verifyMs = performance.now() - verifyStart;
        if (bad.length > 0) {
          await adapter.unload();
          await webllmStore.delete(artifact);
          throw new RuntimeError('integrity-mismatch', `Downloaded weights didn’t match their checksums (${bad.join(', ')}). Load again to download them.`, true);
        }
      }
      return {
        downloadedBytes: before.complete ? 0 : total - before.resumableBytes,
        timings: { downloadMs, readMs: 0, verifyMs, sessionMs: sessionMs - downloadMs, validateMs },
      };
    },
    { signal, onWaiting: () => reportProgress(modelId, { phase: 'waiting' }) },
  );
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
  const webllmSaved = await webllmStore.list();
  let webllmBytes = 0;
  const used = new Set<string>();
  const models: Record<string, ModelStorage> = {};
  for (const model of catalogue) {
    const files = allFiles(model);
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
    const completeArtifacts: string[] = complete.has(model.tokenizer)
      ? onnxArtifacts(model).filter((a) => onnxWeightFiles(a).every((f) => complete.has(f))).map((a) => a.id)
      : [];
    for (const artifact of webllmArtifacts(model)) {
      const state = await webllmStore.state(artifact, webllmSaved);
      savedBytes += state.savedBytes;
      resumableBytes += state.resumableBytes;
      webllmBytes += state.savedBytes;
      partial ||= state.partial;
      if (state.complete) completeArtifacts.push(artifact.id);
    }
    if (savedBytes === 0) continue;
    models[model.id] = { savedBytes, resumableBytes, completeArtifacts, partial };
  }
  let otherBytes = 0;
  for (const [key, r] of records) if (!used.has(key)) otherBytes += r.status === 'complete' ? r.sizeBytes : r.bytesWritten;
  const allWebllmBytes = [...webllmSaved.values()].reduce((n, f) => n + f.payloadBytes, 0);
  otherBytes += allWebllmBytes - webllmBytes;
  return { available: true, models, otherBytes, modelFileBytes: modelFileBytes + allWebllmBytes, recordBytes, ...base };
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
    for (const artifact of webllmArtifacts(model)) {
      const deleted = await store.ifUnlocked(lockKey(artifact), () => webllmStore.delete(artifact));
      if (deleted === null) throw new RuntimeError('in-use-elsewhere', 'Another tab is using this model’s files. Close it, then delete again.', true);
    }
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
      return respond(
        msg.requestId,
        () => load(msg.modelId, import.meta.env.DEV ? msg.engine : undefined, import.meta.env.DEV ? msg.artifact : undefined),
        'load-failed',
      );
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
        if (adapter !== onnxAdapter) throw new RuntimeError('busy', 'Scoring needs the onnx-webgpu engine.');
        await onnxAdapter.scoreContinuation(msg.promptIds, msg.continuation, (step, logits) => {
          let top = 0;
          for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[top]!) top = i;
          steps.push({ logits: (msg.ids[step] ?? []).map((id) => logits[id]!), top: [top, logits[top]!] });
        });
        return { steps };
      });
    case 'debug-fault':
      if (!import.meta.env.DEV) return;
      if (msg.fault === 'webllm-load-fails') {
        simulateWebLlmFailure = true;
      } else if (msg.fault === 'worker-crash') {
        setTimeout(() => {
          throw new Error('Simulated worker crash');
        });
      } else {
        onnxAdapter.simulateFault(msg.fault);
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
