// Development speed bench: loads a catalogue model through the real ONNX adapter (from the dev server's local model
// cache) and measures decoding and a long prompt, with the adapter's per-phase timings (onnx-adapter.ts:
// profilePhase). Served at /bench.html by the dev server; driven by tools/bench/bench.mjs.
//
//   /bench.html?model=<catalogue id>[&artifact=<id>][&tokens=128][&prompt=1000][&session=<json>][&static=0][&capture=0][&context=N][&repeats=3]
// Results are exposed as `window.__bench`.

import { artifactUrl, findModel, type ArtifactFile } from '../catalogue/models';
import type { GenerationOptions } from '../runtime/contract';
import { checkCompatibility } from '../runtime/compatibility';
import type { ByteSource } from '../runtime/onnx/external-weights';
import { OnnxWebGpuAdapter } from '../runtime/onnx/onnx-adapter';
import { probeDevice } from '../device/probe';

const params = new URLSearchParams(location.search);
const modelId = params.get('model') ?? 'smollm2-360m-instruct';
const tokens = Number(params.get('tokens') ?? 128);
const promptTokens = Number(params.get('prompt') ?? 1000);
/** Decoding runs to measure: the best and the median are reported (this laptop's GPU slows down as it heats up). */
const repeats = Number(params.get('repeats') ?? 3);
const sessionOverrides = JSON.parse(params.get('session') ?? '{}') as Record<string, unknown>;

type Profile = Record<string, [number, number]>;
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

const CHUNK = 64 * 1024 * 1024;

/** A file from the dev server, kept as 64 MB chunks (one buffer can't hold a 2 GB file). No integrity check. */
async function fetchFile(model: Parameters<typeof artifactUrl>[0], file: ArtifactFile): Promise<ByteSource> {
  const response = await fetch(artifactUrl(model, file));
  if (!response.ok || !response.body) throw new Error(`${file.path}: HTTP ${response.status}`);
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let current = new Uint8Array(CHUNK);
  let filled = 0;
  let size = 0;
  for (const reader = response.body.getReader(); ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    for (let at = 0; at < value.byteLength; ) {
      const take = Math.min(CHUNK - filled, value.byteLength - at);
      current.set(value.subarray(at, at + take), filled);
      filled += take;
      at += take;
      if (filled === CHUNK) {
        chunks.push(current);
        current = new Uint8Array(CHUNK);
        filled = 0;
      }
    }
  }
  if (filled > 0) chunks.push(current.slice(0, filled));
  if (size !== file.sizeBytes) throw new Error(`${file.path}: ${size} bytes, expected ${file.sizeBytes}`);
  return {
    size,
    read(offset, length) {
      const bytes = new Uint8Array(length);
      for (let done = 0; done < length; ) {
        const at = offset + done;
        const chunk = chunks[Math.floor(at / CHUNK)]!;
        const from = at % CHUNK;
        const take = Math.min(length - done, chunk.byteLength - from);
        bytes.set(chunk.subarray(from, from + take), done);
        done += take;
      }
      return bytes;
    },
  };
}

/** Per-token milliseconds for each phase. */
function perStep(profile: Profile): Record<string, string> {
  return Object.fromEntries(Object.entries(profile).map(([k, [count, ms]]) => [k, `${(ms / count).toFixed(2)} ms × ${count}`]));
}

async function run(): Promise<void> {
  const model = findModel(modelId);
  if (!model) throw new Error(`Unknown model ${modelId}`);
  const device = await probeDevice();
  const gpu = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  state.results.gpu = gpu && { vendor: gpu.info.vendor, architecture: gpu.info.architecture, features: [...gpu.features].sort() };
  log(`GPU ${JSON.stringify(state.results.gpu)}`);
  const forced = params.get('artifact');
  const artifact = forced ? model.artifacts.find((a) => a.id === forced) : checkCompatibility(model, device).artifact;
  if (!artifact) throw new Error('No artifact this device can run');
  log(`${model.name} (${artifact.id})`);

  const t0 = performance.now();
  const byName = async (files: ArtifactFile[]) =>
    Object.fromEntries(await Promise.all(files.map(async (f) => [f.path.slice(f.path.lastIndexOf('/') + 1), await fetchFile(model, f)] as const)));
  const tokenizerBytes = await (await fetch(artifactUrl(model, model.tokenizer))).arrayBuffer();
  const files = {
    graph: await fetchFile(model, artifact.graph),
    externalData: await byName(artifact.externalData ?? []),
    embeddings: artifact.embeddings && { graph: await fetchFile(model, artifact.embeddings.graph), externalData: await byName(artifact.embeddings.externalData) },
    tokenizerJson: JSON.parse(new TextDecoder().decode(tokenizerBytes)) as unknown,
  };
  const readMs = performance.now() - t0;
  const adapter = new OnnxWebGpuAdapter();
  const load = await adapter.load(model, artifact, files, () => {}, {
    sessionOverrides,
    staticCache: params.get('static') !== '0',
    graphCapture: params.get('capture') !== '0',
  });
  log(`read ${Math.round(readMs)} ms, session ${Math.round(load.sessionMs)} ms`);

  const greedy = (maxNewTokens: number): GenerationOptions => ({
    maxNewTokens,
    temperature: 0,
    topP: 1,
    repetitionPenalty: 1,
    repetitionWindow: 0,
    contextTokens: Number(params.get('context') ?? model.generation.contextTokens),
  });
  const global = globalThis as { __quickInfProfile?: Profile };

  // Warm-up (shader compilation, buffer caches).
  await adapter.generate({ turns: [{ role: 'user', content: 'Say hello.' }], options: greedy(16) }, () => {});
  adapter.reset();

  const speeds: number[] = [];
  let decode: Awaited<ReturnType<typeof adapter.generate>> | null = null;
  let decodeProfile: Record<string, string> = {};
  for (let i = 0; i < repeats; i++) {
    global.__quickInfProfile = {};
    const run = await adapter.generate(
      { turns: [{ role: 'user', content: 'Explain how a refrigerator works, step by step, in detail.' }], options: greedy(tokens) },
      () => {},
    );
    adapter.reset();
    speeds.push(run.stats.decodeTokensPerSecond);
    // The profile of the fastest run.
    if (run.stats.decodeTokensPerSecond >= Math.max(...speeds)) {
      decode = run;
      decodeProfile = perStep(global.__quickInfProfile);
    }
  }
  const sorted = [...speeds].sort((a, b) => a - b);
  state.results.decode = {
    tokPerSec: +sorted[sorted.length - 1]!.toFixed(1),
    median: +sorted[Math.floor(sorted.length / 2)]!.toFixed(1),
    tokens: decode!.stats.generatedTokens,
    profile: decodeProfile,
  };
  log(`decode ${JSON.stringify(state.results.decode)}`);

  // About `promptTokens` tokens of prompt (14-token sentences).
  const filler = 'The quick brown fox jumps over the lazy dog near the quiet river bank. '.repeat(Math.max(1, Math.round(promptTokens / 15)));
  global.__quickInfProfile = {};
  const long = await adapter.generate({ turns: [{ role: 'user', content: `Summarise this in one sentence. ${filler}` }], options: greedy(1) }, () => {});
  const longProfile = perStep(global.__quickInfProfile);
  delete global.__quickInfProfile;
  state.results.prompt = { tokens: long.stats.promptTokens, ms: Math.round(long.stats.timeToFirstTokenMs), tokPerSec: +((long.stats.promptTokens * 1000) / long.stats.timeToFirstTokenMs).toFixed(1), profile: longProfile };
  log(`prompt ${long.stats.promptTokens} tokens in ${Math.round(long.stats.timeToFirstTokenMs)} ms ${JSON.stringify(longProfile)}`);
  state.results.sample = decode!.text.slice(0, 160);
  await adapter.unload();
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
