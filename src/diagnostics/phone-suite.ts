// Development: tests catalogue models on this device, on each engine they have,
// through the app's real runtime (worker, model stores, recommendations).
// Built for phones and for leaving the phone running while fixes are made:
//
// - Results go to the dev server (/__devlog → .devlog/devlog.jsonl on the Mac),
//   including which step is running.
// - Progress is kept in localStorage: a GPU reset blocks WebGPU until Chrome
//   restarts, and a crashed tab reloads the page. A reload caused by a code change
//   (Vite) reruns the interrupted test instead of counting it as a crash.
// - Storage: each model's files are deleted once its tests are done, so the phone
//   holds one model at a time; when the run finishes the page wipes everything this
//   origin stored (models, databases, caches) except a small record of the run.
//   ?clear wipes it all, including that record, without running anything.
// - Remote control: while idle the page polls /__phone-control; a new command there
//   starts a run, a clear or a reload, so the Mac can rerun tests without anyone
//   touching the phone. The screen is kept awake.
//
// Models: the original six by default; ?all adds every model up to 4B (never the
// 7B+ or experimental ones); ?only=<id>,<id> picks models; ?engine=onnx|webllm tests
// one engine. ?reset starts over. For iterating (e.g. over USB, driven by
// tools/phone/phone.mjs): ?quick times each thing once instead of repeating it, and
// ?keep leaves models on the device between runs (no deleting, no wipe at the end),
// so a rerun doesn't download them again.
// Per model: ONNX first (known to work on the test phone), then WebLLM (open
// Adreno issue: start-up failures). Last, one normal load without forcing an
// engine, to see what the app picks and whether the fallback works.

import { catalogue, engineName, type EngineId, type ModelSpec } from '../catalogue/models';
import { recommendationStore } from '../recommend/recommendation-store';
import { getRuntimeClient } from '../runtime/client';
import { RuntimeError, type ChatTurn, type GenerationOptions } from '../runtime/contract';
import { deviceEngines } from '../catalogue/device-catalogue';
import { referenceFor } from './fixtures';
// v14: each model on each engine (ONNX, WebLLM), then a normal load.
// v15: models one at a time, deleted after their tests; storage wiped at the end; remote control.
const PROGRESS_KEY = 'quick-inf:phone-suite-progress-v15';
/** Set (for this tab) just before the dev server reloads the page. */
const RELOAD_KEY = 'quick-inf:phone-suite-reload-expected';
/** The last remote command acted on, so each runs once. */
const CONTROL_KEY = 'quick-inf:phone-suite-control';
/** Models tested without ?all or ?only: those the suite covered before the catalogue grew. */
const CORE_MODELS = new Set(['smollm2-360m-instruct', 'qwen2.5-0.5b-instruct', 'qwen3-0.6b', 'llama-3.2-1b-instruct', 'smollm2-1.7b-instruct', 'qwen3-1.7b']);
/** The model the normal (unforced) load uses. */
const DEFAULT_CHECK_MODEL = 'smollm2-360m-instruct';
/** Timed generation runs per model (after one untimed warm-up). */
const SPEED_RUNS = 3;
const LONG_RUNS = 2;
/** Timed repetitions; ?quick makes them 1. */
let speedRuns = { speed: SPEED_RUNS, long: LONG_RUNS };
/** Options a run keeps across reloads (see the header). */
const RUN_OPTIONS = ['all', 'only', 'engine', 'quick', 'keep'];
/** A greedy divergence is acceptable where the reference's top two logits were this close. */
const NEAR_TIE = 0.25;
/** Steps the reference generated when it didn't stop early (tools/reference/make_fixtures.py). */
const REFERENCE_STEPS = 32;
const CONTROL_POLL_MS = 5000;

type Status = 'pass' | 'fail' | 'gpu-reset' | 'tab-crashed' | 'running';

interface Stat {
  mean: number;
  sd: number;
  runs: number[];
}

interface ModelResult {
  /** `${modelId}@${engine}`, or `${modelId}@default` for the normal load. */
  unit: string;
  modelId: string;
  /** The engine forced for this unit, or 'default'. */
  engine: EngineId | 'default';
  /** The engine the load used (differs from the default's first choice after a fallback). */
  usedEngine?: EngineId;
  fallback?: { from: EngineId; reason: string };
  status: Status;
  /** Step in progress (kept when the tab crashes, to show where). */
  phase?: string;
  load?: { downloadedMB: number; downloadS: number; readMs: number; gpuLoadMs: number; artifact: string };
  /**
   * ONNX: greedy tokens against the CPU reference of the same file. WebLLM: a
   * different 4-bit build, so only the text is compared ('differs' isn't a failure).
   */
  correctness?: { case: number; verdict: 'identical' | 'near-tie' | 'diverged' | 'differs'; firstDiff: number | null; text: string }[];
  /**
   * Teacher-forced: the reference's tokens fed one by one, comparing logits for the
   * reference's top 10 at each step. `maxDiff` ignores a constant offset (which
   * doesn't change probabilities); `topChanged` lists steps where our top token
   * differs, with the reference's margin between its top two there.
   */
  logits?: { case: number; steps: number; maxDiff: number; meanDiff: number; topChanged: { step: number; margin: number }[] }[];
  /** "What is the capital of France?": the answer must mention Paris (every model, every engine). */
  sanity?: { ok: boolean; text: string };
  speed?: { tokensPerSecond: Stat; firstTokenMs: Stat; tokens: number; text: string };
  longPrompt?: { promptTokens: number; firstTokenMs: Stat };
  /** A follow-up message: first-word time reusing the previous reply's cache vs from scratch. */
  followUp?: { promptTokens: number; reusedTokens: number; reusedFirstTokenMs: number; freshFirstTokenMs: number };
  error?: string;
}

interface Progress {
  /** The address options this run started with (kept across reloads). */
  params: string;
  device?: unknown;
  models: Record<string, ModelResult>;
  /** Models whose files were deleted after their tests. */
  cleaned: string[];
  finished?: boolean;
}

const params = new URLSearchParams(location.search);
const out = document.getElementById('out')!;
const show = (line: string) => {
  out.textContent += `${line}\n`;
  out.scrollTop = out.scrollHeight;
};

function report(entry: Record<string, unknown>): void {
  const body = JSON.stringify(entry);
  if (!navigator.sendBeacon('/__devlog', new Blob([body], { type: 'text/plain' }))) {
    void fetch('/__devlog', { method: 'POST', body, keepalive: true }).catch(() => {});
  }
}

function loadProgress(): Progress | null {
  try {
    return JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? 'null') as Progress | null;
  } catch {
    return null;
  }
}

function save(progress: Progress): void {
  localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
}

void main();

async function main(): Promise<void> {
  void keepAwake();
  if (params.has('clear')) {
    show('Clearing everything this test stored…');
    const cleared = await wipeStorage({ keepProgress: false });
    report({ kind: 'phone-suite-cleared', ...cleared });
    show(`Cleared: ${cleared.removed.join(', ') || 'nothing was stored'}. Storage in use now: ${cleared.usageMB} MB.`);
    return idle();
  }
  if (params.has('reset')) {
    localStorage.removeItem(PROGRESS_KEY);
    // A reload (a code change, a crash) must carry on with this run, not start it again.
    const rest = new URLSearchParams([...params].filter(([k]) => k !== 'reset'));
    history.replaceState(null, '', `${location.pathname}${rest.size ? `?${rest}` : ''}`);
  }
  let progress = loadProgress();
  if (progress?.finished) {
    show('The last run finished (results are on the Mac). Add ?reset to the address to run again, or ?clear to wipe the test storage.');
    summarise(progress);
    return idle();
  }
  if (!progress) {
    // A new run keeps its options, so a reload mid-run carries on with the same models.
    const kept = new URLSearchParams([...params].filter(([k]) => RUN_OPTIONS.includes(k)));
    progress = { params: kept.toString(), models: {}, cleaned: [] };
    save(progress);
  }
  if ((await run(progress)) === 'cleaning-up') return; // the page is navigating to the clean-up
  return idle();
}

/** Runs the remaining tests; 'cleaning-up' once all are done and the page is leaving to wipe storage. */
async function run(progress: Progress): Promise<'stopped' | 'cleaning-up'> {
  report({ kind: 'phone-suite-start', userAgent: navigator.userAgent, params: progress.params, done: Object.keys(progress.models) });
  // Reloads the dev server causes (a code change, or the server restarting) aren't crashes: the
  // page notes them before they happen (below), and the interrupted test runs again.
  const expected = sessionStorage.getItem(RELOAD_KEY);
  sessionStorage.removeItem(RELOAD_KEY);
  for (const [unit, r] of Object.entries(progress.models)) {
    if (r.status === 'running' && expected) {
      delete progress.models[unit];
      show(`(reloaded by the dev server during ${unit}; running it again)`);
    }
  }
  for (const r of Object.values(progress.models)) {
    // A test still "running" from a previous visit means the tab crashed or reloaded during it.
    if (r.status === 'running') {
      r.status = 'tab-crashed';
      r.error = `The page reloaded during "${r.phase ?? 'start'}" (likely out of memory).`;
      report({ kind: 'phone-suite-model', ...r });
    }
    show(`${label(r.status)} ${r.unit} (earlier)`);
  }
  save(progress);
  const expectReload = () => sessionStorage.setItem(RELOAD_KEY, '1');
  import.meta.hot?.on('vite:beforeFullReload', expectReload);
  // A restarting server drops the connection first, then reloads the page when it's back.
  import.meta.hot?.on('vite:ws:disconnect', expectReload);

  const client = getRuntimeClient();
  const check = await client.check(catalogue[0]!.id).catch(() => null);
  if (!check || check.device.webgpu !== 'available') {
    show('\nWebGPU is not available. If an earlier test reset the GPU, fully close Chrome (swipe it away), reopen it and load this page again.');
    report({ kind: 'phone-suite-blocked' });
    return 'stopped';
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

  const options = new URLSearchParams(progress.params);
  const plan = units(options);
  if (options.has('quick')) speedRuns = { speed: 1, long: 1 };
  for (const [i, { model, engine }] of plan.entries()) {
    const unit = `${model.id}@${engine}`;
    if (!progress.models[unit]) {
      const result: ModelResult = { unit, modelId: model.id, engine, status: 'running' };
      progress.models[unit] = result;
      save(progress);
      show(`\n${model.name} · ${engine === 'default' ? 'normal load' : engineName(engine)}`);
      const phase = (name: string) => {
        result.phase = name;
        save(progress);
        report({ kind: 'phone-suite-phase', unit, phase: name });
      };
      try {
        await testModel(model, engine, result, phase);
        result.status = result.correctness?.some((c) => c.verdict === 'diverged') || result.sanity?.ok === false ? 'fail' : 'pass';
      } catch (err) {
        const code = err instanceof RuntimeError ? err.code : '';
        result.status = code === 'device-lost' || code === 'runtime-crashed' ? 'gpu-reset' : 'fail';
        result.error = err instanceof Error ? err.message : String(err);
      }
      await client.unload().catch(() => {});
      save(progress);
      report({ kind: 'phone-suite-model', ...result });
      show(`${label(result.status)} ${unit}${result.error ? `: ${result.error}` : ''}`);
      if (result.status === 'gpu-reset') {
        show('\nThe GPU reset, so Chrome blocks WebGPU here until it restarts. Fully close Chrome, reopen it and load this page again to continue with the next test.');
        return 'stopped';
      }
    }
    // Once a model's last test is done, delete its files (both engines) before the next model.
    const lastForModel = !plan.slice(i + 1).some((u) => u.model.id === model.id);
    if (lastForModel && !options.has('keep') && !progress.cleaned.includes(model.id)) {
      const deleteError = await client.deleteModel(model.id).then(
        () => undefined,
        (err: unknown) => (err instanceof Error ? err.message : String(err)),
      );
      progress.cleaned.push(model.id);
      save(progress);
      const usageMB = Math.round(((await navigator.storage.estimate().catch(() => undefined))?.usage ?? 0) / 1e6);
      const stored = await storedFiles();
      report({ kind: 'phone-suite-deleted', modelId: model.id, usageMB, deleteError, stored });
      show(`  deleted ${model.name}'s files (${usageMB} MB still stored)${deleteError ? `; error: ${deleteError}` : ''}`);
    }
  }

  progress.finished = true;
  save(progress);
  report({
    kind: 'phone-suite-summary',
    units: Object.values(progress.models).map(({ unit, status, usedEngine, fallback, error }) => ({ unit, status, usedEngine, fallback, error })),
  });
  summarise(progress);
  if (options.has('keep')) {
    show('\nModels kept on this device (?keep); open ?clear to wipe them.');
    return 'stopped';
  }
  // Wipe what the run stored, from a fresh page so no database or file is still open.
  show('\nCleaning up the test storage…');
  location.replace(`${location.pathname}?after-run`);
  return 'cleaning-up';
}

/** Planned tests: per model, ONNX then WebLLM; then the normal load. */
function units(options: URLSearchParams): { model: ModelSpec; engine: EngineId | 'default' }[] {
  const has = (model: ModelSpec, engine: EngineId) => model.artifacts.some((a) => a.engine === engine);
  const only = options.get('only')?.split(',').filter(Boolean);
  // By default the engines this device uses (on phones, only the ONNX one); ?engine= forces one to test.
  const engines = options.get('engine') ? (['onnx-webgpu', 'webllm'] as const).filter((e) => e.startsWith(options.get('engine')!)) : deviceEngines();
  const all = options.has('all');
  const included = (m: ModelSpec) =>
    only ? only.includes(m.id) : CORE_MODELS.has(m.id) || (all && !m.experimental && m.architecture.parameterCount <= 4.5e9);
  const plan: { model: ModelSpec; engine: EngineId | 'default' }[] = [];
  for (const model of catalogue.filter(included)) {
    for (const engine of engines) if (has(model, engine)) plan.push({ model, engine });
  }
  if (!only && !options.get('engine')) plan.push({ model: catalogue.find((m) => m.id === DEFAULT_CHECK_MODEL)!, engine: 'default' });
  return plan;
}

function summarise(progress: Progress): void {
  const results = Object.values(progress.models);
  const count = (s: Status) => results.filter((r) => r.status === s).length;
  show(`\nSummary: ${count('pass')} passed, ${count('fail')} failed, ${count('gpu-reset')} GPU resets, ${count('tab-crashed')} tab crashes.`);
  for (const r of results) {
    const speed = r.speed ? `, ${r.speed.tokensPerSecond.mean} tok/s` : '';
    show(`  ${label(r.status)} ${r.unit}${speed}${r.error ? `: ${r.error.slice(0, 120)}` : ''}`);
  }
}

/** What's left in this origin's files, by folder two levels deep, in MB (for the clean-up reports). */
async function storedFiles(): Promise<Record<string, number>> {
  const sizes: Record<string, number> = {};
  const walk = async (dir: FileSystemDirectoryHandle, path: string[]) => {
    for await (const [name, handle] of (dir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
      if (handle.kind === 'directory') await walk(handle as FileSystemDirectoryHandle, [...path, name]);
      else {
        const key = path.slice(0, 2).join('/') || '(root)';
        sizes[key] = (sizes[key] ?? 0) + (await (handle as FileSystemFileHandle).getFile()).size / 1e6;
      }
    }
  };
  await walk(await navigator.storage.getDirectory(), []).catch(() => {});
  return Object.fromEntries(Object.entries(sizes).map(([k, v]) => [k, Math.round(v)]));
}

/**
 * Removes everything this origin stored: saved model files (both engines),
 * databases, caches and localStorage (except the run record, unless asked).
 */
async function wipeStorage({ keepProgress }: { keepProgress: boolean }): Promise<{ removed: string[]; usageMB: number }> {
  const removed: string[] = [];
  try {
    const root = await navigator.storage.getDirectory();
    for await (const [name] of (root as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
      await root.removeEntry(name, { recursive: true });
      removed.push(`files/${name}`);
    }
  } catch (err) {
    show(`  (files: ${String(err)})`);
  }
  for (const db of (await indexedDB.databases?.().catch(() => [])) ?? []) {
    if (!db.name) continue;
    await new Promise<void>((resolve) => {
      const request = indexedDB.deleteDatabase(db.name!);
      request.onsuccess = request.onerror = () => resolve();
      request.onblocked = () => resolve();
    });
    removed.push(`database ${db.name}`);
  }
  for (const key of (await caches?.keys().catch(() => [])) ?? []) {
    await caches.delete(key);
    removed.push(`cache ${key}`);
  }
  const record = localStorage.getItem(PROGRESS_KEY);
  const control = localStorage.getItem(CONTROL_KEY);
  localStorage.clear();
  if (keepProgress && record) localStorage.setItem(PROGRESS_KEY, record);
  if (control) localStorage.setItem(CONTROL_KEY, control);
  const usage = (await navigator.storage.estimate().catch(() => undefined))?.usage ?? 0;
  return { removed, usageMB: Math.round(usage / 1e6) };
}

/**
 * Waits for remote commands from the Mac (.devlog/phone-control.json via the dev
 * server): { id, action: 'run' | 'clear' | 'reload', params? }. Each id is acted on once.
 */
async function idle(): Promise<void> {
  if (params.has('after-run')) {
    const cleared = await wipeStorage({ keepProgress: true });
    report({ kind: 'phone-suite-cleared', afterRun: true, ...cleared });
    show(`Test storage wiped (${cleared.removed.length} items); ${cleared.usageMB} MB still in use.`);
    history.replaceState(null, '', location.pathname);
  }
  show('\nWaiting for the next command from the Mac (you can leave this page open).');
  report({ kind: 'phone-suite-idle' });
  for (;;) {
    await new Promise((r) => setTimeout(r, CONTROL_POLL_MS));
    const command = await fetch('/__phone-control', { cache: 'no-store' })
      .then((r) => (r.ok ? (r.json() as Promise<{ id?: string; action?: string; params?: string }>) : null))
      .catch(() => null);
    if (!command?.id || command.id === localStorage.getItem(CONTROL_KEY)) continue;
    localStorage.setItem(CONTROL_KEY, command.id);
    report({ kind: 'phone-suite-command', ...command });
    if (command.action === 'run') {
      localStorage.removeItem(PROGRESS_KEY);
      location.replace(`${location.pathname}?${command.params ?? ''}`);
    } else if (command.action === 'clear') {
      location.replace(`${location.pathname}?clear`);
    } else if (command.action === 'reload') {
      location.replace(`${location.pathname}${command.params ? `?${command.params}` : ''}`);
    }
    return;
  }
}

/** Keeps the screen on while the page is visible. */
async function keepAwake(): Promise<void> {
  const nav = navigator as Navigator & { wakeLock?: { request(type: 'screen'): Promise<unknown> } };
  const request = () => nav.wakeLock?.request('screen').catch(() => {});
  await request();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void request();
  });
}

async function testModel(model: ModelSpec, engine: EngineId | 'default', result: ModelResult, phase: (name: string) => void): Promise<void> {
  const client = getRuntimeClient();
  phase('load');
  const stopReporting = watchProgress();
  const loaded = await client.load(model.id, engine === 'default' ? {} : { engine }).finally(stopReporting);
  result.usedEngine = loaded.engine;
  result.fallback = loaded.fallback;
  const t = loaded.timings;
  result.load = {
    downloadedMB: Math.round(loaded.downloadedBytes / 1e6),
    downloadS: Math.round(t.downloadMs / 100) / 10,
    readMs: Math.round(t.readMs),
    gpuLoadMs: Math.round(t.sessionMs + t.validateMs),
    artifact: loaded.artifactId,
  };
  show(
    `  loaded (${engineName(loaded.engine)}, ${loaded.artifactId}): ${result.load.downloadedMB} MB downloaded, GPU load ${result.load.gpuLoadMs} ms` +
      (loaded.fallback ? ` (fell back from ${engineName(loaded.fallback.from)}: ${loaded.fallback.reason})` : ''),
  );
  phase('sanity');
  const answer = await client.generate({ turns: [{ role: 'user', content: 'What is the capital of France? Answer in one sentence.' }], options: greedy(model, 30) }, () => {}).result;
  result.sanity = { ok: /Paris/i.test(answer.text), text: answer.text.slice(0, 80) };
  show(`  sanity: ${result.sanity.ok ? 'ok' : 'WRONG'}: ${JSON.stringify(result.sanity.text)}`);
  if (engine === 'default') return; // the normal load only checks which engine the app picks

  // Correctness: greedy decoding must match the CPU reference token for token, except
  // after a step where the reference's top two candidates were nearly tied. WebLLM runs a
  // different 4-bit build: its text is compared and shown, not judged.
  phase('correctness');
  const artifact = model.artifacts.find((a) => a.id === loaded.artifactId);
  const reference = artifact && referenceFor(model.id, artifact.precision);
  // Llama's prompt carries today's date; its reference was made with a fixed one.
  const textOnly = loaded.engine === 'webllm' || model.promptFormat.kind === 'llama3';
  result.correctness = [];
  for (const [i, c] of (reference?.cases ?? []).entries()) {
    if (textOnly) {
      const res = await client.generate({ turns: c.messages as ChatTurn[], options: greedy(model, c.greedyIds.length) }, () => {}).result;
      const verdict = res.text.trim().startsWith(c.greedyText.trim()) ? 'identical' : 'differs';
      result.correctness.push({ case: i + 1, verdict, firstDiff: null, text: res.text.slice(0, 80) });
      show(`  answer #${i + 1}: ${verdict}: ${JSON.stringify(res.text.slice(0, 60))}`);
      continue;
    }
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

  // Teacher-forced logits: ONNX engine only (WebLLM doesn't expose them).
  phase('logits');
  result.logits = [];
  for (const [i, c] of (loaded.engine === 'onnx-webgpu' ? (reference?.cases ?? []) : []).entries()) {
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
  for (let run = 1; run <= speedRuns.speed; run++) {
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
  for (let run = 0; run < speedRuns.long; run++) {
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
