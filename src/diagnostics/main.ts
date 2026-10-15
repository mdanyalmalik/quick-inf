// Development diagnostics: runs the real ONNX adapter on this browser's GPU and
// compares it with reference fixtures from tools/reference/make_fixtures.py.
// Served at /diagnostics.html by the dev server; not part of the app build.
// Results are also exposed as `window.__diagnostics` for automated runs.

import { artifactUrl, findModel, weightBytes, type ArtifactFile, type ModelSpec } from '../catalogue/models';
import { probeDevice } from '../device/probe';
import { RuntimeError, type GenerationOptions } from '../runtime/contract';
import { checkCompatibility } from '../runtime/compatibility';
import { OnnxWebGpuAdapter } from '../runtime/onnx/onnx-adapter';
import { getOrt, loadOrt } from '../runtime/onnx/ort-env';
import { fetchArtifact, verifySha256 } from '../store/fetch-artifact';
import { Sha256 } from '../store/sha256';
import type { ByteSource } from '../runtime/onnx/external-weights';
import { BpeTokenizer } from '../tokenizer/bpe-tokenizer';
import { createPromptFormatter, type ChatTurn } from '../tokenizer/prompt-format';
import { referenceFor, tokenizerFixturesFor } from './fixtures';
// ?artifact=<id> forces an artifact instead of the device's best (e.g. onnx-q4).
const params = new URLSearchParams(location.search);
const modelId = params.get('model') ?? 'smollm2-360m-instruct';
const forcedArtifact = params.get('artifact');
// ?session=<json> merges extra ONNX Runtime session options (memory experiments).
const sessionOverrides = JSON.parse(params.get('session') ?? '{}') as Record<string, unknown>;
/** ?bindingLimit=134217728 runs on a device with that per-binding limit (a phone's), to reproduce phone-only bugs. */
const bindingLimit = Number(params.get('bindingLimit') ?? 0);
/** ?speed skips the correctness checks and measures speed only. */
const speedOnly = params.has('speed');
/** ?head=0|4|8: bits for a full-precision output layer (0 keeps it; the adapter's default otherwise). */
const quantizeOutput = params.has('head') ? (Number(params.get('head')) as 0 | 4 | 8) : undefined;

const CHUNK = 64 * 1024 * 1024;

/** Random access over consecutive chunks. */
function chunkedSource(chunks: Uint8Array<ArrayBuffer>[], size: number): ByteSource {
  return {
    size,
    read(offset, length) {
      const out = new Uint8Array(length);
      for (let done = 0; done < length; ) {
        const at = offset + done;
        const chunk = chunks[Math.floor(at / CHUNK)]!;
        const from = at % CHUNK;
        const take = Math.min(length - done, chunk.byteLength - from);
        out.set(chunk.subarray(from, from + take), done);
        done += take;
      }
      return out;
    },
  };
}

async function cappedDevice(limit: number): Promise<GPUDevice> {
  const gpuAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!gpuAdapter) throw new Error('no adapter');
  (globalThis as { __heldAdapter?: GPUAdapter }).__heldAdapter = gpuAdapter;
  const requiredLimits: Record<string, number> = {};
  for (const key in gpuAdapter.limits) {
    const value = gpuAdapter.limits[key as keyof GPUSupportedLimits];
    if (typeof value === 'number') requiredLimits[key] = value;
  }
  requiredLimits.maxStorageBufferBindingSize = limit;
  return gpuAdapter.requestDevice({
    requiredFeatures: (['shader-f16', 'timestamp-query'] as const).filter((f) => gpuAdapter.features.has(f)),
    requiredLimits,
  });
}
const MODEL: ModelSpec = findModel(modelId) ?? (() => { throw new Error(`Unknown model ${modelId}`); })();
const tokenizerFixtures = tokenizerFixturesFor(modelId) ?? (() => { throw new Error(`No reference fixtures for ${modelId}`); })();
/** Llama's template writes a date; the reference used a fixed one. */
const formatOptions = { date: (tokenizerFixtures.templateKwargs as { date_string?: string }).date_string };
document.getElementById('model-name')!.textContent = `${MODEL.name} (${modelId})`;

type Outcome = 'pass' | 'fail' | 'info';
interface CheckRecord {
  name: string;
  outcome: Outcome;
  details: unknown;
}

const results: CheckRecord[] = [];
const state: { done: boolean; error?: string; results: CheckRecord[] } = { done: false, results };
(window as unknown as { __diagnostics: typeof state }).__diagnostics = state;

const tbody = document.getElementById('results')!;
const button = document.getElementById('run') as HTMLButtonElement;

function record(name: string, outcome: Outcome, details: unknown): void {
  results.push({ name, outcome, details });
  const row = document.createElement('tr');
  const pre = document.createElement('pre');
  pre.textContent = typeof details === 'string' ? details : JSON.stringify(details, null, 1);
  row.innerHTML = `<td>${name}</td><td class="${outcome}">${outcome.toUpperCase()}</td><td></td>`;
  row.lastElementChild!.appendChild(pre);
  tbody.appendChild(row);
}

const greedy = (maxNewTokens: number): GenerationOptions => ({
  maxNewTokens,
  temperature: 0,
  topP: 1,
  repetitionPenalty: 1,
  repetitionWindow: 0,
  contextTokens: MODEL.generation.contextTokens,
});

async function run(): Promise<void> {
  const device = await probeDevice();
  record('Device probe', device.webgpu === 'available' ? 'info' : 'fail', { ortWeb: await loadOrt().then(() => getOrt().env.versions.web ?? 'unknown'), ...device });

  const adapter = new OnnxWebGpuAdapter();
  const compat = checkCompatibility(MODEL, device);
  record('Compatibility', compat.compatible ? 'pass' : 'fail', { artifact: compat.artifact?.id, reasons: compat.reasons });
  if (!compat.artifact) return;
  const found = forcedArtifact ? MODEL.artifacts.find((a) => a.id === forcedArtifact) : compat.artifact;
  if (!found) {
    record('Compatibility', 'fail', { reason: `no artifact ${forcedArtifact}` });
    return;
  }
  const artifact = found;
  const reference = referenceFor(modelId, artifact.precision)!;

  // Tokenizer fixtures, in this browser's JS engine.
  const t0 = performance.now();
  const tokBytes = await fetchArtifact(artifactUrl(MODEL, MODEL.tokenizer), MODEL.tokenizer, () => {});
  await verifySha256(tokBytes, MODEL.tokenizer);
  const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(tokBytes));
  const tokenizer = BpeTokenizer.fromJSON(tokenizerJson);
  const formatter = createPromptFormatter(MODEL.promptFormat.kind, tokenizer, { ...MODEL.promptFormat, ...formatOptions });
  // Same rules as tokenizer.test.ts: decoding returns normalized text, and text that
  // spells a non-special added token (e.g. "<think>") is kept plain on purpose.
  const tj = tokenizerJson as { normalizer: { type: string } | null; added_tokens: { content: string; special: boolean }[] };
  const normalized = (text: string) => (tj.normalizer?.type === 'NFC' ? text.normalize('NFC') : text);
  const addedText = tj.added_tokens.filter((t) => !t.special).map((t) => t.content);
  const tokFailures: string[] = [];
  for (const c of tokenizerFixtures.text.filter((t) => !addedText.some((a) => t.text.includes(a)))) {
    if (JSON.stringify(tokenizer.encode(c.text)) !== JSON.stringify(c.ids)) tokFailures.push(`encode ${JSON.stringify(c.text)}`);
    if (tokenizer.decode(c.ids) !== normalized(c.text)) tokFailures.push(`decode ${JSON.stringify(c.text)}`);
  }
  for (const c of tokenizerFixtures.chat) {
    if (JSON.stringify(formatter.format(c.messages as ChatTurn[])) !== JSON.stringify(c.ids)) tokFailures.push('chat format');
  }
  const tokCases = tokenizerFixtures.text.length * 2 + tokenizerFixtures.chat.length;
  record('Tokenizer fixtures', tokFailures.length ? 'fail' : 'pass', { cases: tokCases, failures: tokFailures, ms: Math.round(performance.now() - t0) });

  // Download, verify, load.
  let downloadMs = 0;
  let verifyMs = 0;
  // Weight files are kept as 64 MB chunks (one buffer can't hold a 2 GB file), hashed as they arrive.
  const fetchFile = async (file: ArtifactFile): Promise<ByteSource> => {
    const dl = performance.now();
    const response = await fetch(artifactUrl(MODEL, file));
    if (!response.ok || !response.body) throw new Error(`${file.path}: HTTP ${response.status}`);
    const sha = new Sha256();
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let current = new Uint8Array(CHUNK);
    let filled = 0;
    let size = 0;
    for (const reader = response.body.getReader(); ; ) {
      const { done, value } = await reader.read();
      if (done) break;
      sha.update(value);
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
    downloadMs += performance.now() - dl;
    const vf = performance.now();
    const hex = sha.hex();
    if (size !== file.sizeBytes || hex !== file.sha256) throw new Error(`${file.path} failed its integrity check (sha256 ${hex})`);
    verifyMs += performance.now() - vf;
    return chunkedSource(chunks, size);
  };
  const byName = async (files: ArtifactFile[]) =>
    Object.fromEntries(await Promise.all(files.map(async (f) => [f.path.slice(f.path.lastIndexOf('/') + 1), await fetchFile(f)] as const)));
  const graph = await fetchFile(artifact.graph);
  const loadedFiles = {
    graph,
    externalData: await byName(artifact.externalData ?? []),
    embeddings: artifact.embeddings && { graph: await fetchFile(artifact.embeddings.graph), externalData: await byName(artifact.embeddings.externalData) },
    tokenizerJson,
  };
  const gpuDevice = bindingLimit ? await cappedDevice(bindingLimit) : undefined;
  const loadTimes = await adapter.load(MODEL, artifact, loadedFiles, () => {}, { sessionOverrides, device: gpuDevice, formatOptions, quantizeOutput });
  record('Download, verify, load', 'pass', {
    artifact: artifact.id,
    sizeMB: +(weightBytes(artifact) / 1e6).toFixed(1),
    downloadMs: Math.round(downloadMs),
    verifyMs: Math.round(verifyMs),
    sessionMs: Math.round(loadTimes.sessionMs),
    validateMs: Math.round(loadTimes.validateMs),
  });

  if (speedOnly) {
    await speedRuns(adapter);
    return;
  }

  const sameArtifactAsReference = artifact.graph.path === reference.source.artifact;
  const stopIds = new Set(MODEL.promptFormat.stopTokens.map((t) => tokenizer.specialTokenId(t)));

  // Teacher-forced comparison against reference logits.
  /** Per case: true when every argmax mismatch was at a near-tie in the reference. */
  const nearTieOnly: boolean[] = [];
  for (const [i, c] of reference.cases.entries()) {
    let argmaxMatches = 0;
    let maxAbsDiff = 0;
    let marginOfMismatches = 0;
    let steps = 0;
    await adapter.scoreContinuation(c.promptIds, c.greedyIds, (step, logits) => {
      const ref = c.topK[step]!;
      steps++;
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j]! > logits[best]!) best = j;
      if (best === ref.ids[0]) argmaxMatches++;
      else marginOfMismatches = Math.max(marginOfMismatches, ref.logits[0]! - ref.logits[1]!);
      ref.ids.forEach((id, k) => {
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(logits[id]! - ref.logits[k]!));
      });
    });
    // Mismatches are acceptable only where the reference's top-2 were nearly tied.
    const ok = sameArtifactAsReference ? argmaxMatches === steps || marginOfMismatches < 0.25 : true;
    nearTieOnly[i] = argmaxMatches < steps && marginOfMismatches < 0.25;
    record(`Teacher-forced logits #${i + 1}`, sameArtifactAsReference ? (ok ? 'pass' : 'fail') : 'info', {
      steps,
      argmaxMatches,
      maxAbsLogitDiffTop10: +maxAbsDiff.toFixed(4),
      largestRefMarginAtMismatch: +marginOfMismatches.toFixed(4),
      note: sameArtifactAsReference ? undefined : `reference uses ${reference.source.artifact}`,
    });
  }

  // Greedy generation through the public generate() path.
  for (const [i, c] of reference.cases.entries()) {
    const expected = c.greedyIds.filter((id) => !stopIds.has(id));
    const res = await adapter.generate({ turns: c.messages as ChatTurn[], options: greedy(c.greedyIds.length) }, () => {});
    let firstDiff = -1;
    for (let k = 0; k < Math.max(expected.length, res.tokenIds.length); k++) {
      if (expected[k] !== res.tokenIds[k]) {
        firstDiff = k;
        break;
      }
    }
    const refStopped = stopIds.has(c.greedyIds[c.greedyIds.length - 1]!);
    // Greedy output can only diverge at a near-tie in the reference (rounding picks the other
    // top token; a reused cache rounds differently from a fresh prefill), after which the rest
    // of the text legitimately differs.
    const top = firstDiff === -1 ? undefined : c.topK[firstDiff];
    const nearTieAtDiff = top !== undefined && top.logits[0]! - top.logits[1]! < 0.25;
    const outcome: Outcome = !sameArtifactAsReference ? 'info' : firstDiff === -1 || nearTieOnly[i] || nearTieAtDiff ? 'pass' : 'fail';
    record(`Greedy generation #${i + 1}`, outcome, {
      identicalTokens: firstDiff === -1,
      divergedAfterNearTie: firstDiff !== -1 && (nearTieOnly[i] || nearTieAtDiff),
      firstDifferenceAt: firstDiff === -1 ? null : firstDiff,
      finishReason: res.finishReason,
      expectedFinish: refStopped ? 'stop' : 'length',
      text: res.text,
      stats: roundStats(res.stats),
    });
    if (refStopped && res.finishReason !== 'stop') record('Stops on end-of-turn', 'fail', res.finishReason);
  }

  // Length limit.
  const short = await adapter.generate({ turns: [{ role: 'user', content: 'Count from one to twenty.' }], options: greedy(5) }, () => {});
  record('Length limit', short.finishReason === 'length' && short.tokenIds.length === 5 ? 'pass' : 'fail', {
    finishReason: short.finishReason,
    tokens: short.tokenIds.length,
  });

  // Cancellation at the next step boundary.
  let deltas = 0;
  let cancelAt = 0;
  const cancelResult = await adapter.generate(
    { turns: [{ role: 'user', content: 'Write a long story about a lighthouse keeper.' }], options: greedy(400) },
    () => {
      if (++deltas === 5) {
        cancelAt = performance.now();
        adapter.cancel();
      }
    },
  );
  const stopLatency = performance.now() - cancelAt;
  record('Cancellation', cancelResult.finishReason === 'cancelled' && cancelResult.tokenIds.length < 20 ? 'pass' : 'fail', {
    finishReason: cancelResult.finishReason,
    tokensBeforeStop: cancelResult.tokenIds.length,
    stopLatencyMs: Math.round(stopLatency),
  });

  // Context handling: oversize message is rejected; long history is trimmed.
  try {
    await adapter.generate({ turns: [{ role: 'user', content: 'word '.repeat(6000) }], options: greedy(4) }, () => {});
    record('Context overflow rejected', 'fail', 'generation unexpectedly succeeded');
  } catch (err) {
    record('Context overflow rejected', err instanceof RuntimeError && err.code === 'context-overflow' ? 'pass' : 'fail', String(err));
  }
  const history: ChatTurn[] = [];
  for (let k = 0; k < 40; k++) {
    history.push({ role: 'user', content: `Question ${k}: ${'tell me something interesting. '.repeat(8)}` });
    history.push({ role: 'assistant', content: `Answer ${k}: ${'here is an interesting fact. '.repeat(8)}` });
  }
  history.push({ role: 'user', content: 'What number was the last question?' });
  const trimmed = await adapter.generate({ turns: history, options: { ...greedy(8), contextTokens: 1024 } }, () => {});
  record('Context trimming', trimmed.stats.droppedTurns > 0 && trimmed.stats.promptTokens <= 1024 ? 'pass' : 'fail', {
    droppedTurns: trimmed.stats.droppedTurns,
    promptTokens: trimmed.stats.promptTokens,
  });

  // Unload and reload from the same verified bytes; output must not change.
  await adapter.unload();
  const reload = await adapter.load(MODEL, artifact, loadedFiles, () => {}, { sessionOverrides, device: gpuDevice, formatOptions, quantizeOutput });
  const first = reference.cases[0]!;
  const again = await adapter.generate({ turns: first.messages as ChatTurn[], options: greedy(first.greedyIds.length) }, () => {});
  const before = results.find((r) => r.name === 'Greedy generation #1')?.details as { text: string } | undefined;
  record('Unload and reload', again.text === before?.text ? 'pass' : 'fail', {
    sessionMs: Math.round(reload.sessionMs),
    text: again.text,
  });

  // Throughput on a longer greedy answer.
  const perf = await adapter.generate(
    { turns: [{ role: 'user', content: 'Explain how a refrigerator works in detail.' }], options: greedy(256) },
    () => {},
  );
  record('Throughput (greedy, 256 max)', 'info', roundStats(perf.stats));

  // Prompt processing on a long prompt (~1,500 tokens), for calibrating estimates.
  const longPrompt = `Summarise this text in one sentence. ${'The quick brown fox jumps over the lazy dog near the quiet river bank. '.repeat(100)}`;
  const prefill = await adapter.generate({ turns: [{ role: 'user', content: longPrompt }], options: greedy(1) }, () => {});
  record('Prefill (long prompt)', 'info', roundStats(prefill.stats));
  await adapter.unload();
}

/** ?speed: decode speed (2 × 128 greedy tokens) and a ~1,500-token prompt, after a warm-up. */
async function speedRuns(adapter: OnnxWebGpuAdapter): Promise<void> {
  const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
  await adapter.generate({ turns: [{ role: 'user', content: 'Say hello.' }], options: greedy(16) }, () => {});
  const decode: number[] = [];
  const tokens: number[] = [];
  for (let i = 0; i < 2; i++) {
    const r = await adapter.generate({ turns: [{ role: 'user', content: `(${i}) Explain how a refrigerator works in detail.` }], options: greedy(128) }, () => {});
    decode.push(r.stats.decodeTokensPerSecond);
    tokens.push(r.stats.generatedTokens);
  }
  record('Speed: decode', 'info', { tokPerSec: +median(decode).toFixed(1), runs: decode.map((d) => +d.toFixed(1)), tokens });
  const filler = 'The quick brown fox jumps over the lazy dog near the quiet river bank. '.repeat(100);
  const waits: number[] = [];
  let promptTokens = 0;
  for (let i = 0; i < 1; i++) {
    const r = await adapter.generate({ turns: [{ role: 'user', content: `(${i}) Summarise this text in one sentence. ${filler}` }], options: greedy(1) }, () => {});
    waits.push(r.stats.timeToFirstTokenMs);
    promptTokens = r.stats.promptTokens;
  }
  record('Speed: long prompt', 'info', { promptTokens, firstTokenMs: Math.round(median(waits)), runs: waits.map(Math.round) });
  await adapter.unload();
}

function roundStats(stats: object): Record<string, number> {
  return Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, Math.round((v as number) * 10) / 10]));
}

button.addEventListener('click', () => {
  button.disabled = true;
  tbody.textContent = '';
  results.length = 0;
  run()
    .catch((err: unknown) => {
      state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      record('Unexpected error', 'fail', state.error);
    })
    .finally(() => {
      state.done = true;
      button.disabled = false;
    });
});

if (new URLSearchParams(location.search).has('autorun')) button.click();
