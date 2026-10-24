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
import { findSpeechModel, SPEECH_MODELS, speechModelFiles, type SpeechModelSpec } from '../../catalogue/speech-models';
import { measureMemory, runDeviceTest, type MemoryCheck } from '../../device/device-test';
import { probeDevice } from '../../device/probe';
import { ArtifactError, fetchArtifact, hasBackgroundDownload, verifySha256 } from '../../store/fetch-artifact';
import { dataBytes } from '../../store/data-size';
import { artifactKey, ModelStore } from '../../store/model-store';
import type { ByteSource } from '../onnx/external-weights';
import { checkCompatibility, CPU_MAX_WEIGHT_BYTES, noGpuReason, processorFor } from '../compatibility';
import { browserSignals, chooseBuild } from '../onnx/ort-env';
import { simulation } from '../../device/simulation';
import {
  RuntimeError,
  type LoadProgress,
  type LoadResult,
  type RuntimeErrorCode,
  type RuntimeErrorInfo,
} from '../contract';
import { OnnxWebGpuAdapter } from '../onnx/onnx-adapter';
import { getGpuDevice } from '../../device/gpu';
import { startPeer } from '../../distributed/peer-stage';
import { buffersOf, type Channel, type DeviceInfo, type PipelineMessage } from '../../distributed/protocol';
import { PipelineLink } from '../../distributed/remote-stage';
import { describePlan, planShare, shareStages } from '../../distributed/share-plan';
import { deleteRanges, keepMade, keptMade, keptRange, keptRangeBytes, pruneRanges } from '../../store/range-cache';
import { ImageEngine } from '../image/image-engine';
import { SpeechEngine } from '../speech/speech-engine';
import type {
  CheckResult,
  DebugScore,
  DownloadPlan,
  ModelStorage,
  RuntimeStatus,
  StorageReport,
  WorkerEvent,
  ShareRequest,
  WorkerRequest,
  WorkerResults,
} from './protocol';

const PROGRESS_INTERVAL_MS = 100;
/** Speed estimates before this much data are too noisy to show. */
const MIN_BYTES_FOR_SPEED = 2_000_000;

/** A model whose files the store keeps: chat models, the image models that run, and speech models. */
interface StoredModel {
  id: string;
  source: ModelSpec['source'];
  files: ArtifactFile[];
  /** Each artifact (precision) and the files it needs. */
  artifacts: { id: string; files: ArtifactFile[] }[];
}

/** The one artifact of an image or speech model (its ONNX files). */
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
  ...SPEECH_MODELS.map((m) => ({ id: m.id, source: m.source, files: speechModelFiles(m), artifacts: [{ id: IMAGE_ARTIFACT, files: speechModelFiles(m) }] })),
];

// Every file the catalogues use, so the store can delete any they no longer do.
const store = new ModelStore(new Map(STORED.flatMap((model) => model.files.map((file) => [artifactKey(model, file), file.sha256] as const))));
const events = { onFatal: (err: RuntimeError) => fail(status.modelId, err, true) };
const adapter = new OnnxWebGpuAdapter(events);
const imageEngine = new ImageEngine();
const speechEngine = new SpeechEngine();

/** An image model that runs in the app (has files), or undefined. */
function runnableImageModel(modelId: string): ImageModelSpec | undefined {
  const model = findImageModel(modelId);
  return model && runsInApp(model) ? model : undefined;
}

/** Unloads whatever model is loaded, of any kind. */
async function unloadAll(): Promise<void> {
  await adapter.unload();
  await imageEngine.unload();
  await speechEngine.unload();
}

// WebLLM (removed 2026-10-11) kept its downloads in its own OPFS folder: delete them.
void (async () => {
  if (!(await store.init())) return;
  const root = await navigator.storage.getDirectory();
  await root.removeEntry('tvmjs-opfs-store', { recursive: true }).catch(() => {});
})();
let status: RuntimeStatus = { state: 'idle', modelId: null };
let loading: { modelId: string; artifactId?: string; streamed: boolean; promise: Promise<LoadResult>; abort: AbortController } | null = null;
let activeGeneration: number | null = null;
let lastProgressAt = 0;

/** Reads the live status (avoids TypeScript keeping a narrowing across awaits). */
function currentStatus(): RuntimeStatus {
  return status;
}

function post(event: WorkerEvent): void {
  self.postMessage(event);
}

// Sharing a model across devices (src/distributed): pipeline messages travel through the page, which holds the
// WebRTC channels. A host's link to its devices is made once and kept; a device runs its part once it has joined.
const pipeHandlers: ((message: PipelineMessage) => void)[] = [];
const pipe: Channel = {
  send: (message) => self.postMessage({ kind: 'pipe', message } satisfies WorkerEvent, { transfer: buffersOf(message) }),
  onMessage: (handler) => void pipeHandlers.push(handler),
  close() {},
};
let pipelineLink: PipelineLink | null = null;
/** Time one message takes between devices on a local network, for planning. */
const SHARE_HOP_MS = 3;
let joined = false;

/** Runs part of another device's model: its byte ranges kept in this browser's storage, checked against its hashes. */
async function join(id: string, device: DeviceInfo): Promise<null> {
  if (joined) return null;
  joined = true;
  // The parts of the latest assignment: once it's loaded, the rest (an earlier share of the model) are deleted.
  let assigned = { key: '', parts: new Set<string>() };
  startPeer({
    id,
    device,
    gpu: await getGpuDevice(),
    channel: pipe,
    fetchPart: (model, from, length, onBytes) => {
      assigned.parts.add(from.sha256);
      return keptRange(model, { ...from, length }, onBytes);
    },
    keptMade: (model, sha256, length) => {
      assigned.parts.add(sha256);
      return keptMade(model, sha256, length);
    },
    keepMade: (model, sha256, length) => {
      assigned.parts.add(sha256);
      return keepMade(model, sha256, length);
    },
    log: (text) => console.info(`[shared model] ${text}`),
    onState: (state) => {
      post({ kind: 'peer-state', state });
      const key = state.phase === 'loading' ? `${state.modelId}:${state.layers.join('-')}` : assigned.key;
      if (key !== assigned.key) assigned = { key, parts: new Set() };
      // Its part is kept here now (only this one): the storage figures include it.
      const model = state.phase === 'ready' ? findModel(state.modelId) : undefined;
      if (model) void pruneRanges(model, assigned.parts).then(publishStorage);
      else if (state.phase === 'failed') void publishStorage();
    },
  });
  return null;
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
  if (err instanceof ArtifactError) {
    // A missing or forbidden file won't download by trying again; a dropped connection will.
    const retryable = err.code === 'download-failed' ? err.transient : err.code !== 'cancelled';
    return { code: err.code, message: err.message, retryable };
  }
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

/** No GPU path in this browser even with WebGPU present (WebKit without JSPI; see onnx/ort-env.ts). */
function cpuOnly(): boolean {
  return (simulation()?.ort ?? chooseBuild(browserSignals())) === 'cpu';
}

/**
 * Whether this device can run an image or speech model: on the GPU it needs the features its files use; without
 * a usable GPU, small models run on the CPU.
 */
function filesCompatibility(
  model: { name: string; requiredGpuFeatures?: GPUFeatureName[] },
  files: ArtifactFile[],
  device: Awaited<ReturnType<typeof probeDevice>>,
): CheckResult['compatibility'] {
  if (processorFor(device, cpuOnly()) === 'cpu') {
    return files.reduce((n, f) => n + f.sizeBytes, 0) <= CPU_MAX_WEIGHT_BYTES
      ? { compatible: true, processor: 'cpu', reasons: [`${noGpuReason(device)} It runs on the CPU instead, much more slowly.`] }
      : { compatible: false, processor: 'cpu', reasons: [`${noGpuReason(device)} ${model.name} is too big to run on the CPU here.`] };
  }
  const missing = (model.requiredGpuFeatures ?? []).filter((f) => !device.gpu!.features.includes(f));
  return missing.length === 0 ? { compatible: true, processor: 'gpu', reasons: [] } : { compatible: false, reasons: [`${model.name} needs GPU feature(s): ${missing.join(', ')}`] };
}

function imageCompatibility(model: ImageModelSpec, device: Awaited<ReturnType<typeof probeDevice>>): CheckResult['compatibility'] {
  return filesCompatibility(model, imageModelFiles(model), device);
}

/** Why a model can't load here: too big for the CPU (no usable GPU), or a GPU feature is missing. */
function incompatible(compatibility: CheckResult['compatibility']): RuntimeError {
  return new RuntimeError(compatibility.processor === 'cpu' ? 'too-big-for-cpu' : 'incompatible-device', compatibility.reasons.join(' '));
}

/** What a load would use here. */
async function check(modelId: string): Promise<CheckResult> {
  const image = runnableImageModel(modelId);
  const speech = findSpeechModel(modelId);
  const device = await probeDevice();
  if (image) return { device, compatibility: imageCompatibility(image, device) };
  if (speech) return { device, compatibility: filesCompatibility(speech, speechModelFiles(speech), device) };
  return { device, compatibility: checkCompatibility(requireModel(modelId), device, { cpuOnly: cpuOnly() }) };
}

/** `artifactId` (development only) loads that artifact instead of the first this device can run. */
/** What loading `modelId` here would still download (files not yet saved, with their addresses). */
async function downloadPlan(modelId: string): Promise<DownloadPlan> {
  const image = runnableImageModel(modelId);
  const speech = findSpeechModel(modelId);
  let stored: Pick<StoredModel, 'id' | 'source'>;
  let files: ArtifactFile[];
  if (image) {
    stored = { id: image.id, source: image.source! };
    files = imageModelFiles(image);
  } else if (speech) {
    stored = speech;
    files = speechModelFiles(speech);
  } else {
    const model = requireModel(modelId);
    const compatibility = checkCompatibility(model, await probeDevice(), { cpuOnly: cpuOnly() });
    if (!compatibility.compatible || !compatibility.artifact) return { files: [], canSave: false };
    stored = model;
    files = artifactFiles(model, compatibility.artifact);
  }
  if (!(await store.init())) return { files: [], canSave: false };
  const states = await store.inspect(stored, files);
  return {
    canSave: true,
    files: states
      .filter((s) => !s.complete)
      .map((s) => ({ url: artifactUrl(stored, s.file), path: s.file.path, sizeBytes: s.file.sizeBytes })),
  };
}

/** The streaming plan of the load in progress (one load runs at a time), from the page's memory limit. */
let streamRequest: { cacheBytes: number } | undefined;
/** The devices the load in progress is shared with, if it is. */
let shareRequest: ShareRequest | undefined;

async function load(modelId: string, artifactId?: string, streamExperts?: { cacheBytes: number }, share?: ShareRequest): Promise<LoadResult> {
  if (status.needsRestart && status.error) {
    throw new RuntimeError(status.error.code, status.error.message, status.error.retryable);
  }
  // The same build, run the same way (streamed experts or not).
  const same = (r: LoadResult) => (!artifactId || r.artifactId === artifactId) && !r.streamedExperts === !streamExperts && !r.shared && !share;
  if (status.state === 'ready' && status.modelId === modelId && status.load && same(status.load)) return status.load;
  if (loading?.modelId === modelId && loading.artifactId === artifactId && loading.streamed === !!streamExperts) return loading.promise;
  if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before loading a model.');
  if (loading) await cancelLoad();

  const abort = new AbortController();
  streamRequest = streamExperts;
  shareRequest = share;
  const promise = doLoad(modelId, abort.signal, artifactId);
  loading = { modelId, artifactId, streamed: !!streamExperts, promise, abort };
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
    const speech = findSpeechModel(modelId);
    if (speech) {
      setStatus({ state: 'checking', modelId, progress: { phase: 'checking' } });
      const result = await loadSpeech(speech, signal);
      setStatus({ state: 'ready', modelId, load: result });
      return result;
    }
    const model = requireModel(modelId);
    setStatus({ state: 'checking', modelId, progress: { phase: 'checking' } });
    await unloadAll();
    const device = await probeDevice();
    const compatibility = checkCompatibility(model, device, { onlyArtifact: artifactOverride, cpuOnly: cpuOnly() });
    throwIfCancelled(signal);
    if (!compatibility.compatible || !compatibility.artifact) throw incompatible(compatibility);
    let result = await loadArtifact(model, compatibility.artifact, signal, compatibility.processor ?? 'gpu');
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

async function loadArtifact(model: ModelSpec, artifact: ModelArtifact, signal: AbortSignal, processor: 'gpu' | 'cpu'): Promise<LoadResult> {
  const loaded = await loadOnnx(model, artifact, signal, processor);
  if (signal.aborted) {
    await adapter.unload();
    throwIfCancelled(signal);
  }
  return {
    processor,
    modelId: model.id,
    artifactId: artifact.id,
    engine: artifact.engine,
    precision: artifact.precision,
    downloadedBytes: loaded.downloadedBytes,
    timings: loaded.timings,
    ...(loaded.streamedExperts && { streamedExperts: loaded.streamedExperts }),
    ...(loaded.shared && { shared: loaded.shared }),
  };
}

type Loaded = Pick<LoadResult, 'downloadedBytes' | 'timings' | 'streamedExperts' | 'shared'>;

async function loadOnnx(model: ModelSpec, artifact: OnnxArtifact, signal: AbortSignal, processor: 'gpu' | 'cpu'): Promise<Loaded> {
  const modelId = model.id;
  const files = artifactFiles(model, artifact);
  // Weight files are read tensor by tensor while ONNX Runtime loads them, never whole.
  const weights = new Set(onnxWeightFiles(artifact));
  const loadGraphs = async (
    obtained: Obtained,
    source: (file: ArtifactFile) => Uint8Array<ArrayBuffer> | ByteSource,
    keepOpen?: () => () => void,
  ) => {
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
      keepOpen,
    };
    // Development: ?sim=stream=<GB> streams any mixture-of-experts model; otherwise the person chose it when loading.
    const simulated = simulation()?.streamExperts;
    const streamExperts = typeof simulated === 'number' ? { cacheBytes: simulated * 1e9 } : streamRequest;
    // Shared across devices: the plan from each one's memory and speed, this device's layers here, theirs on them.
    const share = shareRequest?.devices.length && processor === 'gpu' ? shareRequest : undefined;
    const devices = share && [{ id: 'host', ...share.host }, ...share.devices.map((d) => ({ id: d.id, ...d.device }))];
    const shared = devices && planShare(model, artifact, loaded, devices, model.generation.contextTokens, SHARE_HOP_MS);
    if (devices && !shared) throw new RuntimeError('not-shareable', 'The devices connected can’t hold this model together.');
    const label = (id: string) => (id === 'host' ? 'this device' : (share?.devices.find((d) => d.id === id)?.device.label ?? id));
    if (shared) console.info(`[shared model] ${describePlan(shared.plan, label)} (${shared.summary})`);
    pipelineLink ??= new PipelineLink(pipe);
    const times = await adapter.load(model, artifact, loaded, (p) => reportProgress(modelId, p), {
      processor,
      ...(streamExperts && { streamExperts }),
      ...(shared && { splitLayers: shared.cuts, remoteStage: shareStages(pipelineLink, model, shared) }),
    });
    // The adapter streams only a mixture of experts' experts, and only on the GPU.
    const streamed = streamExperts && artifact.experts && processor === 'gpu' ? streamExperts : undefined;
    const sharedResult: LoadResult['shared'] = shared
      ? {
          hostLayers: [shared.plan.shares[0]!.from, shared.plan.shares[0]!.to],
          devices: shared.plan.shares.slice(1).map((s) => ({ id: s.id, label: label(s.id), layers: [s.from, s.to] })),
        }
      : undefined;
    return { obtained, ...times, streamed, sharedResult };
  };
  const { obtained, sessionMs, validateMs, streamed, sharedResult } = (await store.init())
    ? // The lock is held (and the files open) until loading is done, or while the engine keeps reading them.
      await new Promise<Awaited<ReturnType<typeof loadGraphs>>>((resolve, reject) => {
        store
          .withLock(
            model,
            async () => {
              const obtained = await obtainSaved(model, files, signal, weights);
              const readers = new Map<ArtifactFile, ByteSource & { close(): void }>();
              let kept: Promise<void> | null = null;
              const keepOpen = () => {
                let done!: () => void;
                kept = new Promise<void>((r) => (done = r));
                return done;
              };
              try {
                for (const file of weights) readers.set(file, await store.openReader(model, file));
                resolve(await loadGraphs(obtained, (file) => readers.get(file)!, keepOpen));
                if (kept) await kept;
              } finally {
                for (const reader of readers.values()) reader.close();
              }
            },
            { signal, onWaiting: () => reportProgress(modelId, { phase: 'waiting' }) },
          )
          .catch(reject);
      })
    : await obtainInMemory(model, files, signal).then((obtained) => loadGraphs(obtained, (file) => obtained.bytes[files.indexOf(file)]!));
  return {
    downloadedBytes: obtained.downloadedBytes,
    timings: { ...obtained.timings, sessionMs, validateMs },
    ...(streamed && { streamedExperts: streamed }),
    ...(sharedResult && { shared: sharedResult }),
  };
}

/** A model's files for an engine: large ones read on demand, small ones (tokenizers, configs) whole. */
interface EngineFiles {
  bytes(file: ArtifactFile): Uint8Array<ArrayBuffer>;
  source(file: ArtifactFile): ByteSource;
}

/**
 * Downloads (or reads) the files of an image or speech model and hands them to `start`, which creates its
 * sessions; returns what was obtained and how long the sessions took.
 */
async function loadFiles(
  stored: Pick<StoredModel, 'id' | 'source'>,
  files: ArtifactFile[],
  signal: AbortSignal,
  start: (files: EngineFiles) => Promise<{ sessionMs: number }>,
): Promise<{ obtained: Obtained; sessionMs: number }> {
  // Graphs and weights are read tensor by tensor while ONNX Runtime loads them, never whole; tokenizers whole.
  const onDisk = new Set(files.filter((f) => !/\.(json|txt)$/.test(f.path)));
  const run = async (obtained: Obtained, source: (file: ArtifactFile) => Uint8Array<ArrayBuffer> | ByteSource) => {
    throwIfCancelled(signal);
    reportProgress(stored.id, { phase: 'loading' });
    const { sessionMs } = await start({
      bytes: (file) => obtained.bytes[files.indexOf(file)]!,
      source: (file) => {
        const s = source(file);
        return s instanceof Uint8Array ? { size: s.byteLength, read: (o: number, n: number) => s.subarray(o, o + n) } : s;
      },
    });
    return { obtained, sessionMs };
  };
  return (await store.init())
    ? await store.withLock(
        stored,
        async () => {
          const obtained = await obtainSaved(stored, files, signal, onDisk);
          const readers = new Map<ArtifactFile, ByteSource & { close(): void }>();
          try {
            for (const file of onDisk) readers.set(file, await store.openReader(stored, file));
            return await run(obtained, (file) => readers.get(file) ?? obtained.bytes[files.indexOf(file)]!);
          } finally {
            for (const reader of readers.values()) reader.close();
          }
        },
        { signal, onWaiting: () => reportProgress(stored.id, { phase: 'waiting' }) },
      )
    : await obtainInMemory(stored, files, signal).then((obtained) => run(obtained, (file) => obtained.bytes[files.indexOf(file)]!));
}

/** Downloads (or reads) an image model's files and creates its sessions. One model runs at a time. */
async function loadImage(model: ImageModelSpec, signal: AbortSignal): Promise<LoadResult> {
  await adapter.unload();
  await speechEngine.unload();
  const device = await probeDevice();
  const compatibility = imageCompatibility(model, device);
  throwIfCancelled(signal);
  if (!compatibility.compatible) throw incompatible(compatibility);
  const { obtained, sessionMs } = await loadFiles({ id: model.id, source: model.source! }, imageModelFiles(model), signal, (files) =>
    imageEngine.load(model, files, compatibility.processor ?? 'gpu'),
  );
  if (signal.aborted) {
    await imageEngine.unload();
    throwIfCancelled(signal);
  }
  return {
    processor: compatibility.processor ?? 'gpu',
    modelId: model.id,
    artifactId: IMAGE_ARTIFACT,
    engine: 'onnx-webgpu',
    precision: model.precision ?? 'fp16',
    downloadedBytes: obtained.downloadedBytes,
    timings: { ...obtained.timings, sessionMs, validateMs: 0 },
  };
}

/** Downloads (or reads) a speech model's files and creates its sessions. */
async function loadSpeech(model: SpeechModelSpec, signal: AbortSignal): Promise<LoadResult> {
  await adapter.unload();
  await imageEngine.unload();
  const device = await probeDevice();
  const compatibility = filesCompatibility(model, speechModelFiles(model), device);
  throwIfCancelled(signal);
  if (!compatibility.compatible) throw incompatible(compatibility);
  const { obtained, sessionMs } = await loadFiles(model, speechModelFiles(model), signal, (files) =>
    speechEngine.load(model, files, compatibility.processor ?? 'gpu'),
  );
  if (signal.aborted) {
    await speechEngine.unload();
    throwIfCancelled(signal);
  }
  return {
    processor: compatibility.processor ?? 'gpu',
    modelId: model.id,
    artifactId: IMAGE_ARTIFACT,
    engine: 'onnx-webgpu',
    precision: model.precision,
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
    // Files the browser downloaded in the background already take their space; each is moved into the store and
    // then dropped from there, so only the largest needs room twice.
    const background = await Promise.all(missing.map((s) => hasBackgroundDownload(artifactUrl(model, s.file))));
    const fromNetwork = missing.reduce((n, s, i) => n + (background[i] ? 0 : s.file.sizeBytes - s.savedBytes), 0);
    const largestMoved = Math.max(0, ...missing.filter((_, i) => background[i]).map((s) => s.file.sizeBytes));
    await store.ensureSpace(fromNetwork + largestMoved);
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
  let modelFileBytes = await store.diskBytes();
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
    // Chat models' parts kept from running them for another device.
    const sharedPartBytes = findModel(model.id) ? await keptRangeBytes(model) : 0;
    modelFileBytes += sharedPartBytes;
    if (savedBytes === 0 && sharedPartBytes === 0) continue;
    models[model.id] = { savedBytes, resumableBytes, completeArtifacts, partial, sharedPartBytes };
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
async function deviceTest(requestId: number) {
  if (loading || activeGeneration !== null) {
    throw new RuntimeError('busy', 'Wait for the model to finish loading or replying, then run the test.');
  }
  return runDeviceTest({ onProgress: (step, sofar) => post({ kind: 'device-test-progress', requestId, step, sofar }) });
}

async function memoryCheck(requestId: number, targetBytes: number): Promise<MemoryCheck> {
  if (loading || activeGeneration !== null) {
    throw new RuntimeError('busy', 'Wait for the model to finish loading or replying, then run the test.');
  }
  // A loaded model already holds GPU memory, so the check would measure the wrong thing.
  if (status.state === 'ready') return { status: 'skipped', reason: 'a model is loaded' };
  return measureMemory(targetBytes, (heldBytes, nextBytes) => post({ kind: 'memory-check-progress', requestId, heldBytes, nextBytes }));
}

async function deleteModel(modelId: string): Promise<null> {
  const model = STORED.find((m) => m.id === modelId);
  if (!model) throw new RuntimeError('load-failed', `Unknown model ${modelId}`);
  if (loading?.modelId === modelId) throw new RuntimeError('busy', 'Cancel the load before deleting the model.');
  if (!(await store.init())) return null;
  try {
    await store.deleteModel(model);
    await deleteRanges(model);
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
  } catch (err) {
    // A device running part of a shared model went away: the model needs loading again.
    if (err instanceof RuntimeError && err.code === 'device-disconnected') fail(ready.modelId, err);
    throw err;
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

async function generateSpeech(requestId: number, msg: Extract<WorkerRequest, { kind: 'generate-speech' }>) {
  if (status.state !== 'ready' || !status.load || speechEngine.modelId !== status.modelId) {
    throw new RuntimeError('not-loaded', 'Load the model first.');
  }
  if (activeGeneration !== null) throw new RuntimeError('busy', 'Speech is already being generated.');
  const ready = status;
  activeGeneration = requestId;
  setStatus({ ...ready, state: 'generating' });
  try {
    return await speechEngine.generate(msg.request, (progress) => {
      const event: WorkerEvent = { kind: 'speech-progress', requestId, progress };
      self.postMessage(event, { transfer: [progress.audio.buffer] });
    });
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
        () => load(msg.modelId, import.meta.env.DEV ? msg.artifact : undefined, msg.streamExperts, msg.share),
        'load-failed',
      );
    case 'generate':
      return respond(msg.requestId, () => generate(msg.requestId, msg));
    case 'pipe':
      for (const handler of pipeHandlers) handler(msg.message);
      return;
    case 'join':
      return respond(msg.requestId, () => join(msg.id, msg.device));
    case 'generate-image':
      return respond(msg.requestId, () => generateImage(msg.requestId, msg));
    case 'generate-speech':
      return respond(msg.requestId, () => generateSpeech(msg.requestId, msg));
    case 'cancel':
      if (activeGeneration === msg.targetRequestId && status.state === 'generating') {
        adapter.cancel();
        imageEngine.cancel();
        speechEngine.cancel();
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
        await unloadAll();
        if (!status.needsRestart) setStatus({ state: 'idle', modelId: null });
        return null;
      });
    case 'download-plan':
      return respond(msg.requestId, () => downloadPlan(msg.modelId));
    case 'storage':
      return respond(msg.requestId, storageReport);
    case 'delete-model':
      return respond(msg.requestId, () => deleteModel(msg.modelId));
    case 'device-test':
      return respond(msg.requestId, () => deviceTest(msg.requestId));
    case 'memory-check':
      return respond(msg.requestId, () => memoryCheck(msg.requestId, msg.targetBytes));
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
