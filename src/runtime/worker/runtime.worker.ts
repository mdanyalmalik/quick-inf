// Dedicated inference worker. Owns the device probe, artifact download and
// verification, and the runtime adapter; reports lifecycle status to the main
// thread. One model is loaded at a time.

import { artifactUrl, findModel, type ModelSpec } from '../../catalogue/models';
import { probeDevice } from '../../device/probe';
import { ArtifactError, fetchArtifact, verifySha256 } from '../../store/fetch-artifact';
import { RuntimeError, type LoadProgress, type LoadResult, type RuntimeErrorInfo } from '../contract';
import { OnnxWebGpuAdapter } from '../onnx/onnx-adapter';
import type { CheckResult, RuntimeStatus, WorkerEvent, WorkerRequest, WorkerResults } from './protocol';

const PROGRESS_INTERVAL_MS = 100;

const adapter = new OnnxWebGpuAdapter();
let status: RuntimeStatus = { state: 'idle', modelId: null };
let loading: { modelId: string; promise: Promise<LoadResult> } | null = null;
let activeGeneration: number | null = null;
let lastProgressAt = 0;

function post(event: WorkerEvent): void {
  self.postMessage(event);
}

function setStatus(next: RuntimeStatus): void {
  status = next;
  post({ kind: 'status', status });
}

function reportProgress(modelId: string, progress: LoadProgress): void {
  const state = progress.phase;
  const now = performance.now();
  const isByteUpdate = progress.phase === 'downloading' && status.state === 'downloading';
  if (isByteUpdate && now - lastProgressAt < PROGRESS_INTERVAL_MS && progress.loadedBytes !== progress.totalBytes) return;
  lastProgressAt = now;
  setStatus({ state, modelId, progress });
}

function toInfo(err: unknown): RuntimeErrorInfo {
  if (err instanceof RuntimeError) return err.toInfo();
  if (err instanceof ArtifactError) return { code: err.code, message: err.message, retryable: err.code === 'download-failed' };
  return { code: 'generation-failed', message: err instanceof Error ? err.message : String(err), retryable: true };
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
  if (status.state === 'ready' && status.modelId === modelId && status.load) return status.load;
  if (loading?.modelId === modelId) return loading.promise;
  if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before loading a model.');

  const promise = doLoad(modelId);
  loading = { modelId, promise };
  try {
    return await promise;
  } finally {
    if (loading?.promise === promise) loading = null;
  }
}

async function doLoad(modelId: string): Promise<LoadResult> {
  try {
    const model = requireModel(modelId);
    setStatus({ state: 'checking', modelId });
    await adapter.unload();
    const { device, compatibility } = await check(modelId);
    if (!compatibility.compatible || !compatibility.artifact) {
      const code = device.webgpu === 'available' ? 'incompatible-device' : 'webgpu-unavailable';
      throw new RuntimeError(code, compatibility.reasons.join(' '));
    }
    const artifact = compatibility.artifact;

    // Download (tokenizer first: small, and fails fast on network problems).
    const downloadStart = performance.now();
    const total = model.tokenizer.sizeBytes + artifact.graph.sizeBytes;
    const tokenizerBytes = await fetchArtifact(artifactUrl(model, model.tokenizer), model.tokenizer, (p) =>
      reportProgress(modelId, { phase: 'downloading', file: p.file, loadedBytes: p.loadedBytes, totalBytes: total }),
    );
    const graphBytes = await fetchArtifact(artifactUrl(model, artifact.graph), artifact.graph, (p) =>
      reportProgress(modelId, {
        phase: 'downloading',
        file: p.file,
        loadedBytes: model.tokenizer.sizeBytes + p.loadedBytes,
        totalBytes: total,
      }),
    );
    const downloadMs = performance.now() - downloadStart;

    reportProgress(modelId, { phase: 'verifying' });
    const verifyStart = performance.now();
    await verifySha256(tokenizerBytes, model.tokenizer);
    await verifySha256(graphBytes, artifact.graph);
    const verifyMs = performance.now() - verifyStart;

    const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(tokenizerBytes));
    const { sessionMs, validateMs } = await adapter.load(model, artifact, { graph: graphBytes, tokenizerJson }, (p) =>
      reportProgress(modelId, p),
    );

    const result: LoadResult = {
      modelId,
      artifactId: artifact.id,
      precision: artifact.precision,
      timings: { downloadMs, verifyMs, sessionMs, validateMs },
    };
    setStatus({ state: 'ready', modelId, load: result });
    return result;
  } catch (err) {
    setStatus({ state: 'failed', modelId, error: toInfo(err) });
    throw err;
  }
}

async function handle(msg: WorkerRequest): Promise<void> {
  switch (msg.kind) {
    case 'check':
      return respond(msg.requestId, () => check(msg.modelId));
    case 'load':
      return respond(msg.requestId, () => load(msg.modelId));
    case 'generate':
      return respond(msg.requestId, async () => {
        if (status.state !== 'ready' || !status.load) throw new RuntimeError('not-loaded', 'Load the model first.');
        if (activeGeneration !== null) throw new RuntimeError('busy', 'A response is already being generated.');
        const ready = status;
        activeGeneration = msg.requestId;
        setStatus({ ...ready, state: 'generating' });
        try {
          return await adapter.generate(msg.request, (text) => post({ kind: 'delta', requestId: msg.requestId, text }));
        } finally {
          activeGeneration = null;
          setStatus({ ...ready, state: 'ready' });
        }
      });
    case 'cancel':
      if (activeGeneration === msg.targetRequestId) {
        adapter.cancel();
        setStatus({ ...status, state: 'stopping' });
      }
      return;
    case 'reset':
      adapter.reset();
      return;
    case 'unload':
      return respond(msg.requestId, async () => {
        if (activeGeneration !== null) throw new RuntimeError('busy', 'Stop the current response before unloading.');
        await adapter.unload();
        setStatus({ state: 'idle', modelId: null });
        return null;
      });
  }
}

async function respond<K extends keyof WorkerResults>(requestId: number, run: () => Promise<WorkerResults[K]>) {
  try {
    const value = await run();
    post({ kind: 'result', requestId, value });
  } catch (err) {
    post({ kind: 'error', requestId, error: toInfo(err) });
  }
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void handle(e.data);
};

post({ kind: 'status', status });
