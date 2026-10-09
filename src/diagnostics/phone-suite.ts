// Development: full test of every catalogue model on this device, through the
// app's real runtime (worker, model store, recommendations). Built for phones:
// results go to the dev server (/__devlog → .devlog/devlog.jsonl on the Mac),
// and progress is kept in localStorage, because a GPU reset blocks WebGPU until
// Chrome restarts and a crashed tab reloads the page. Models are downloaded into
// the app's own store, so the app can use them afterwards. ?reset starts over.

import { catalogue, type ModelSpec } from '../catalogue/models';
import { recommendationStore } from '../recommend/recommendation-store';
import { getRuntimeClient } from '../runtime/client';
import { RuntimeError, type ChatTurn, type GenerationOptions } from '../runtime/contract';
import smollm2Ref from '../runtime/__fixtures__/smollm2-q4f16-reference.json';
import qwen25Ref from '../runtime/__fixtures__/qwen2.5-0.5b-q4f16-reference.json';
import qwen3Ref from '../runtime/__fixtures__/qwen3-0.6b-q4f16-reference.json';
import smollm2LargeRef from '../runtime/__fixtures__/smollm2-1.7b-q4f16-reference.json';

/** CPU references for the artifact each model uses on a GPU with fp16. */
const REFERENCES: Record<string, typeof smollm2Ref> = {
  'smollm2-360m-instruct': smollm2Ref,
  'qwen2.5-0.5b-instruct': qwen25Ref,
  'qwen3-0.6b': qwen3Ref,
  'smollm2-1.7b-instruct': smollm2LargeRef,
};
// v2: repeated speed runs, follow-up-message timing, crash phase; new key so the phone reruns everything.
// v13: Qwen2.5 0.5B on its fused-attention export, with the causal-mask fix; new key so the phone reruns everything.
const PROGRESS_KEY = 'quick-inf:phone-suite-progress-v13';
/** Timed generation runs per model (after one untimed warm-up). */
const SPEED_RUNS = 3;
const LONG_RUNS = 2;
/** A greedy divergence is acceptable where the reference's top two logits were this close. */
const NEAR_TIE = 0.25;
/** Steps the reference generated when it didn't stop early (tools/reference/make_fixtures.py). */
const REFERENCE_STEPS = 32;

type Status = 'pass' | 'fail' | 'gpu-reset' | 'tab-crashed' | 'running';

interface Stat {
  mean: number;
  sd: number;
  runs: number[];
}

interface ModelResult {
  modelId: string;
  status: Status;
  /** Step in progress (kept when the tab crashes, to show where). */
  phase?: string;
  load?: { downloadedMB: number; downloadS: number; readMs: number; gpuLoadMs: number; artifact: string };
  correctness?: { case: number; verdict: 'identical' | 'near-tie' | 'diverged'; firstDiff: number | null; text: string }[];
  /**
   * Teacher-forced: the reference's tokens fed one by one, comparing logits for the
   * reference's top 10 at each step. `maxDiff` ignores a constant offset (which
   * doesn't change probabilities); `topChanged` lists steps where our top token
   * differs, with the reference's margin between its top two there.
   */
  logits?: { case: number; steps: number; maxDiff: number; meanDiff: number; topChanged: { step: number; margin: number }[] }[];
  speed?: { tokensPerSecond: Stat; firstTokenMs: Stat; tokens: number; text: string };
  longPrompt?: { promptTokens: number; firstTokenMs: Stat };
  /** A follow-up message: first-word time reusing the previous reply's cache vs from scratch. */
  followUp?: { promptTokens: number; reusedTokens: number; reusedFirstTokenMs: number; freshFirstTokenMs: number };
  error?: string;
}

interface Progress {
  device?: unknown;
  models: Record<string, ModelResult>;
}

const out = document.getElementById('out')!;
const show = (line: string) => {
  out.textContent += `${line}\n`;
  out.scrollTop = out.scrollHeight;
};

function report(entry: unknown): void {
  const body = JSON.stringify(entry);
  if (!navigator.sendBeacon('/__devlog', new Blob([body], { type: 'text/plain' }))) {
    void fetch('/__devlog', { method: 'POST', body, keepalive: true }).catch(() => {});
  }
}

function load(): Progress {
  if (new URLSearchParams(location.search).has('reset')) localStorage.removeItem(PROGRESS_KEY);
  return JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? '{"models":{}}') as Progress;
}

function save(progress: Progress): void {
  localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
}

void run();

async function run(): Promise<void> {
  const progress = load();
  report({ kind: 'phone-suite-start', userAgent: navigator.userAgent, done: Object.keys(progress.models) });
  for (const r of Object.values(progress.models)) {
    // A model still "running" from a previous visit means the tab crashed or reloaded during it.
    if (r.status === 'running') {
      r.status = 'tab-crashed';
      r.error = `The page reloaded during "${r.phase ?? 'start'}" (likely out of memory).`;
      report({ kind: 'phone-suite-model', ...r });
    }
    show(`${label(r.status)} ${r.modelId} (earlier)`);
  }
  save(progress);

  const client = getRuntimeClient();
  const check = await client.check(catalogue[0]!.id).catch(() => null);
  if (!check || check.device.webgpu !== 'available') {
    show('\nWebGPU is not available. If an earlier test reset the GPU, fully close Chrome (swipe it away), reopen it and load this page again.');
    report({ kind: 'phone-suite-blocked' });
    return;
  }

  if (!progress.device) {
    show('Running the device test…');
    await recommendationStore.init();
    await recommendationStore.runDeviceTest();
    const test = recommendationStore.getSnapshot().test;
    progress.device = test;
    save(progress);
    report({ kind: 'phone-suite-device', test });
    if (test) {
      show(
        `Device: ${Math.round(test.bandwidthGBps)} GB/s raw, ${Math.round(test.matvecGBps ?? 0)} GB/s 4-bit, ` +
          `${Math.round(test.computeGflops)} GFLOP/s ${test.computePrecision}, ${test.dispatchUs.toFixed(1)} µs/job GPU, ` +
          `${(test.stepUs ?? 0).toFixed(1)} µs/job wall, memory check ${test.memory.status} (${Math.round(test.durationMs / 100) / 10} s)`,
      );
    }
  }

  for (const model of catalogue) {
    if (progress.models[model.id]) continue;
    const result: ModelResult = { modelId: model.id, status: 'running' };
    progress.models[model.id] = result;
    save(progress);
    show(`\n${model.name}`);
    const phase = (name: string) => {
      result.phase = name;
      save(progress);
    };
    try {
      await testModel(model, result, phase);
      result.status = result.correctness?.some((c) => c.verdict === 'diverged') ? 'fail' : 'pass';
    } catch (err) {
      const code = err instanceof RuntimeError ? err.code : '';
      result.status = code === 'device-lost' || code === 'runtime-crashed' ? 'gpu-reset' : 'fail';
      result.error = err instanceof Error ? err.message : String(err);
    }
    await client.unload().catch(() => {});
    save(progress);
    report({ kind: 'phone-suite-model', ...result });
    show(`${label(result.status)} ${model.name}${result.error ? `: ${result.error}` : ''}`);
    if (result.status === 'gpu-reset') {
      show('\nThe GPU reset, so Chrome blocks WebGPU here until it restarts. Fully close Chrome, reopen it and load this page again to continue with the next model.');
      return;
    }
  }

  report({ kind: 'phone-suite-summary', models: Object.values(progress.models).map(({ modelId, status, error }) => ({ modelId, status, error })) });
  show('\nDone. Results were sent to the Mac. Add ?reset to the address to run everything again.');
}

async function testModel(model: ModelSpec, result: ModelResult, phase: (name: string) => void): Promise<void> {
  const client = getRuntimeClient();
  phase('load');
  const stopReporting = watchProgress();
  const loaded = await client.load(model.id).finally(stopReporting);
  const t = loaded.timings;
  result.load = {
    downloadedMB: Math.round(loaded.downloadedBytes / 1e6),
    downloadS: Math.round(t.downloadMs / 100) / 10,
    readMs: Math.round(t.readMs),
    gpuLoadMs: Math.round(t.sessionMs + t.validateMs),
    artifact: loaded.artifactId,
  };
  show(`  loaded (${loaded.artifactId}): ${result.load.downloadedMB} MB downloaded, GPU load ${result.load.gpuLoadMs} ms`);

  // Correctness: greedy decoding must match the CPU reference token for token, except
  // after a step where the reference's top two candidates were nearly tied.
  phase('correctness');
  const reference = REFERENCES[model.id]!;
  result.correctness = [];
  for (const [i, c] of reference.cases.entries()) {
    // The reference stops early only on a stop token, which it records and our results don't.
    const expected = c.greedyIds.length < REFERENCE_STEPS ? c.greedyIds.slice(0, -1) : c.greedyIds;
    const res = await client.generate({ turns: c.messages as ChatTurn[], options: greedy(model, c.greedyIds.length) }, () => {}).result;
    let firstDiff: number | null = null;
    for (let k = 0; k < Math.min(expected.length, res.tokenIds.length); k++) {
      if (expected[k] !== res.tokenIds[k]) {
        firstDiff = k;
        break;
      }
    }
    const margin = firstDiff === null ? Infinity : (c.topK[firstDiff]!.logits[0]! - c.topK[firstDiff]!.logits[1]!);
    const verdict = firstDiff === null ? 'identical' : margin < NEAR_TIE ? 'near-tie' : 'diverged';
    result.correctness.push({ case: i + 1, verdict, firstDiff, text: res.text.slice(0, 80) });
    show(`  correctness #${i + 1}: ${verdict}${firstDiff !== null ? ` at token ${firstDiff}` : ''}`);
  }

  phase('logits');
  result.logits = [];
  for (const [i, c] of reference.cases.entries()) {
    const score = await client.debugScore(c.promptIds, c.greedyIds, c.topK.map((t) => t.ids));
    let maxDiff = 0;
    let sumDiff = 0;
    const topChanged: { step: number; margin: number }[] = [];
    score.steps.forEach((step, k) => {
      const ref = c.topK[k]!;
      const diffs = step.logits.map((v, j) => v - ref.logits[j]!);
      const offset = diffs.reduce((a, b) => a + b, 0) / diffs.length;
      const shape = Math.max(...diffs.map((d) => Math.abs(d - offset)));
      maxDiff = Math.max(maxDiff, shape);
      sumDiff += shape;
      if (step.top[0] !== ref.ids[0]) topChanged.push({ step: k, margin: Math.round((ref.logits[0]! - ref.logits[1]!) * 1000) / 1000 });
    });
    const round = (x: number) => Math.round(x * 1000) / 1000;
    result.logits.push({ case: i + 1, steps: score.steps.length, maxDiff: round(maxDiff), meanDiff: round(sumDiff / score.steps.length), topChanged });
    show(`  logits #${i + 1}: max diff ${round(maxDiff)}, mean ${round(sumDiff / score.steps.length)}, top changed at ${JSON.stringify(topChanged)}`);
  }

  // Speed with the model's normal sampling settings: one untimed warm-up, then
  // SPEED_RUNS timed runs (fixed seeds, so reruns are comparable).
  phase('speed');
  const ask = [{ role: 'user' as const, content: 'Explain how a refrigerator works, in about 150 words.' }];
  await client.generate({ turns: ask, options: { ...model.generation, maxNewTokens: 16, seed: 0 } }, () => {}).result;
  const speeds: number[] = [];
  const firsts: number[] = [];
  let tokens = 0;
  let text = '';
  for (let run = 1; run <= SPEED_RUNS; run++) {
    // A different first message each run, so no run reuses the previous one's cache.
    const turns = [{ role: 'user' as const, content: `${ask[0]!.content} (${run})` }];
    const res = await client.generate({ turns, options: { ...model.generation, maxNewTokens: 128, seed: run } }, () => {}).result;
    speeds.push(res.stats.decodeTokensPerSecond);
    firsts.push(res.stats.timeToFirstTokenMs);
    tokens += res.stats.generatedTokens;
    text = res.text;
  }
  result.speed = { tokensPerSecond: stat(speeds, 1), firstTokenMs: stat(firsts, 0), tokens, text: text.slice(0, 100) };
  show(`  speed: ${fmt(result.speed.tokensPerSecond)} tok/s, first word ${fmt(result.speed.firstTokenMs)} ms`);

  // A follow-up message: the previous reply's cache is reused, compared with the same prompt from scratch.
  phase('follow-up');
  const first = await client.generate({ turns: ask, options: { ...greedy(model, 64) } }, () => {}).result;
  const followTurns: ChatTurn[] = [...ask, { role: 'assistant', content: first.text }, { role: 'user', content: 'Now explain it to a five-year-old.' }];
  const reused = await client.generate({ turns: followTurns, options: greedy(model, 1) }, () => {}).result;
  client.forgetLastReply();
  const fresh = await client.generate({ turns: followTurns, options: greedy(model, 1) }, () => {}).result;
  result.followUp = {
    promptTokens: reused.stats.promptTokens,
    reusedTokens: reused.stats.reusedPromptTokens,
    reusedFirstTokenMs: Math.round(reused.stats.timeToFirstTokenMs),
    freshFirstTokenMs: Math.round(fresh.stats.timeToFirstTokenMs),
  };
  show(
    `  follow-up (${result.followUp.promptTokens} tokens, ${result.followUp.reusedTokens} reused): ` +
      `first word ${result.followUp.reusedFirstTokenMs} ms (${result.followUp.freshFirstTokenMs} ms from scratch)`,
  );

  // A long message (~1,500 tokens).
  phase('long-prompt');
  const longFirsts: number[] = [];
  let promptTokens = 0;
  for (let run = 0; run < LONG_RUNS; run++) {
    const long = await client.generate(
      {
        turns: [{ role: 'user', content: `Summarise this in one sentence (${run}). ${'The quick brown fox jumps over the lazy dog near the quiet river bank. '.repeat(100)}` }],
        options: greedy(model, 1),
      },
      () => {},
    ).result;
    longFirsts.push(long.stats.timeToFirstTokenMs);
    promptTokens = long.stats.promptTokens;
  }
  result.longPrompt = { promptTokens, firstTokenMs: stat(longFirsts, 0) };
  show(`  long message: ${promptTokens} tokens, first word ${fmt(result.longPrompt.firstTokenMs)} ms`);
  phase('done');
}

function stat(values: number[], digits: number): Stat {
  const round = (x: number) => Math.round(x * 10 ** digits) / 10 ** digits;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, values.length - 1));
  return { mean: round(mean), sd: round(sd), runs: values.map(round) };
}

function fmt(s: Stat): string {
  return `${s.mean} ± ${s.sd}`;
}

function greedy(model: ModelSpec, maxNewTokens: number): GenerationOptions {
  return { maxNewTokens, temperature: 0, topP: 1, repetitionPenalty: 1, repetitionWindow: 0, contextTokens: model.generation.contextTokens };
}

/** Shows download progress while a model loads. */
function watchProgress(): () => void {
  const client = getRuntimeClient();
  let last = '';
  return client.subscribe(() => {
    const p = client.getStatus().progress;
    const line =
      p?.phase === 'downloading'
        ? `  downloading ${Math.round(p.loadedBytes / 1e6)} / ${Math.round(p.totalBytes / 1e6)} MB`
        : p
          ? `  ${p.phase}…`
          : '';
    if (line && line.split(' / ')[0]!.replace(/\d+$/, (n) => String(Math.floor(Number(n) / 50) * 50)) !== last) {
      last = line.split(' / ')[0]!.replace(/\d+$/, (n) => String(Math.floor(Number(n) / 50) * 50));
      show(line);
    }
  });
}

function label(status: Status): string {
  return { pass: 'PASS', fail: 'FAIL', 'gpu-reset': 'GPU RESET', 'tab-crashed': 'TAB CRASHED', running: 'RUNNING' }[status];
}
