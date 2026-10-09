// Development diagnostics: runs the real ONNX adapter on this browser's GPU and
// compares it with reference fixtures from tools/reference/make_fixtures.py.
// Served at /diagnostics.html by the dev server; not part of the app build.
// Results are also exposed as `window.__diagnostics` for automated runs.

import { artifactUrl, findModel, type ModelSpec } from '../catalogue/models';
import { probeDevice } from '../device/probe';
import { RuntimeError, type GenerationOptions } from '../runtime/contract';
import { OnnxWebGpuAdapter } from '../runtime/onnx/onnx-adapter';
import { ORT_VERSION } from '../runtime/onnx/ort-env';
import { fetchArtifact, verifySha256 } from '../store/fetch-artifact';
import { BpeTokenizer } from '../tokenizer/bpe-tokenizer';
import { ChatMlFormatter, type ChatTurn } from '../tokenizer/chatml';
import smollm2Tok from '../tokenizer/__fixtures__/smollm2-tokenizer.json';
import qwen25smallTok from '../tokenizer/__fixtures__/qwen2.5-0.5b-tokenizer.json';
import qwen3Tok from '../tokenizer/__fixtures__/qwen3-0.6b-tokenizer.json';
import smollm2LargeTok from '../tokenizer/__fixtures__/smollm2-1.7b-tokenizer.json';
import smollm2Ref from '../runtime/__fixtures__/smollm2-q4f16-reference.json';
import qwen25smallRef from '../runtime/__fixtures__/qwen2.5-0.5b-q4f16-reference.json';
import qwen3Ref from '../runtime/__fixtures__/qwen3-0.6b-q4f16-reference.json';
import smollm2LargeRef from '../runtime/__fixtures__/smollm2-1.7b-q4f16-reference.json';
import qwen25smallQ4Ref from '../runtime/__fixtures__/qwen2.5-0.5b-q4-reference.json';

// ?model=<catalogue id> picks the model (default: SmolLM2).
const FIXTURES: Record<string, { tokenizer: typeof smollm2Tok; reference: typeof smollm2Ref }> = {
  'smollm2-360m-instruct': { tokenizer: smollm2Tok, reference: smollm2Ref },
  'qwen2.5-0.5b-instruct': { tokenizer: qwen25smallTok, reference: qwen25smallRef },
  'qwen3-0.6b': { tokenizer: qwen3Tok, reference: qwen3Ref },
  'smollm2-1.7b-instruct': { tokenizer: smollm2LargeTok, reference: smollm2LargeRef },
};
// References from the q4 (fp32) files, for ?artifact=onnx-q4.
const Q4_REFERENCES: Record<string, typeof smollm2Ref> = { 'qwen2.5-0.5b-instruct': qwen25smallQ4Ref };
// ?artifact=<id> forces an artifact instead of the device's best (e.g. onnx-q4).
const params = new URLSearchParams(location.search);
const modelId = params.get('model') ?? 'smollm2-360m-instruct';
const forcedArtifact = params.get('artifact');
// ?session=<json> merges extra ONNX Runtime session options (memory experiments).
const sessionOverrides = JSON.parse(params.get('session') ?? '{}') as Record<string, unknown>;
/** ?bindingLimit=134217728 runs on a device with that per-binding limit (a phone's), to reproduce phone-only bugs. */
const bindingLimit = Number(params.get('bindingLimit') ?? 0);

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
const tokenizerFixtures = FIXTURES[modelId]!.tokenizer;
const reference = (forcedArtifact === 'onnx-q4' ? Q4_REFERENCES[modelId] : undefined) ?? FIXTURES[modelId]!.reference;
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
  record('Device probe', device.webgpu === 'available' ? 'info' : 'fail', { ortWeb: ORT_VERSION, ...device });

  const adapter = new OnnxWebGpuAdapter();
  const compat = adapter.checkCompatibility(MODEL, device);
  record('Compatibility', compat.compatible ? 'pass' : 'fail', { artifact: compat.artifact?.id, reasons: compat.reasons });
  if (!compat.artifact) return;
  const artifact = forcedArtifact ? MODEL.artifacts.find((a) => a.id === forcedArtifact)! : compat.artifact;

  // Tokenizer fixtures, in this browser's JS engine.
  const t0 = performance.now();
  const tokBytes = await fetchArtifact(artifactUrl(MODEL, MODEL.tokenizer), MODEL.tokenizer, () => {});
  await verifySha256(tokBytes, MODEL.tokenizer);
  const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(tokBytes));
  const tokenizer = BpeTokenizer.fromJSON(tokenizerJson);
  const formatter = new ChatMlFormatter(tokenizer, {
    defaultSystemPrompt: MODEL.promptFormat.defaultSystemPrompt,
    disableThinking: MODEL.promptFormat.disableThinking,
  });
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
  const dl = performance.now();
  const graph = await fetchArtifact(artifactUrl(MODEL, artifact.graph), artifact.graph, () => {});
  const downloadMs = performance.now() - dl;
  const vf = performance.now();
  await verifySha256(graph, artifact.graph);
  const verifyMs = performance.now() - vf;
  const gpuDevice = bindingLimit ? await cappedDevice(bindingLimit) : undefined;
  const loadTimes = await adapter.load(MODEL, artifact, { graph, tokenizerJson }, () => {}, { sessionOverrides, device: gpuDevice });
  record('Download, verify, load', 'pass', {
    artifact: artifact.id,
    sizeMB: +(artifact.graph.sizeBytes / 1e6).toFixed(1),
    downloadMs: Math.round(downloadMs),
    verifyMs: Math.round(verifyMs),
    sessionMs: Math.round(loadTimes.sessionMs),
    validateMs: Math.round(loadTimes.validateMs),
  });

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
    // Greedy output can only diverge where teacher forcing found a mismatch; after a
    // near-tie (rounding picks the other top token) the rest of the text legitimately differs.
    const outcome: Outcome = !sameArtifactAsReference ? 'info' : firstDiff === -1 || nearTieOnly[i] ? 'pass' : 'fail';
    record(`Greedy generation #${i + 1}`, outcome, {
      identicalTokens: firstDiff === -1,
      divergedAfterNearTie: firstDiff !== -1 && nearTieOnly[i],
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
  const reload = await adapter.load(MODEL, artifact, { graph, tokenizerJson }, () => {}, { device: gpuDevice });
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
