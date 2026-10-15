// Turns the device test, the catalogue and what has really happened on this
// device into a ranked list of models with a verdict each. Pure: the caller
// supplies the inputs (see recommendation-store.ts).
//
// Measured beats estimated: a model that has run here shows its measured speed,
// and the ratio between measured and estimated speed across those models
// corrects the estimates of the others. A failed load is remembered until a
// later load of the same model succeeds (or the environment changes, which
// gives a fresh set of records).

import { artifactFingerprint, loadedWeightBytes, type ModelArtifact, type ModelSpec } from '../catalogue/models';
import type { DeviceTestResult } from '../device/device-test';
import { chatQuality } from './quality';
import { type Calibration, DEFAULT_CALIBRATION, estimateModel, memoryNeeded, type ModelEstimate } from './estimate';
import type { RunRecord } from './run-records';
import { fitChat, limitText, streamingEnabled, streamPlan, type Budget } from './budget';

export type Verdict = 'good' | 'slow' | 'may-not-fit' | 'failed' | 'unsupported' | 'untested';

/** Generation speed and long-prompt wait that count as "runs well". */
const GOOD_TOKENS_PER_SECOND = 15;
const GOOD_LONG_PROMPT_MS = 8000;
const VERY_SLOW_TOKENS_PER_SECOND = 5;
export const STREAMED_REASON =
  'Its experts don’t all fit in this device’s memory, so they’re read from storage as it runs: slower than with more memory.';
/** Load failures that mean "this model doesn't run here" (not network or user problems). */
const RUN_FAILURES = new Set(['load-failed', 'validation-failed', 'device-lost', 'runtime-crashed', 'out-of-memory', 'page-killed']);

export interface Measured {
  tokensPerSecond?: number;
  firstTokenShortMs?: number;
  firstTokenLongMs?: number;
  replies: number;
  /** The latest load attempt failed for a reason that suggests it can't run here. */
  lastLoadFailed?: { code: string; at: number };
}

export interface Recommendation {
  model: ModelSpec;
  /** The file this device would use; absent when none can run here. */
  artifact?: ModelArtifact;
  verdict: Verdict;
  /** Plain-language reasons for the verdict, most important first. */
  reasons: string[];
  /** Estimates (already corrected by measurements of other models), when a device test exists. */
  estimate?: ModelEstimate;
  measured: Measured;
  memoryBytes?: number;
  /** The single best choice for this device: exactly one once the device is benchmarked (if any model could run), none before. */
  recommended: boolean;
  /** Why the recommended model was picked: it runs well, it's the least slow, or the smallest that might fit. */
  pick?: 'runs-well' | 'least-slow' | 'smallest';
}

const VERDICT_ORDER: Verdict[] = ['good', 'slow', 'untested', 'may-not-fit', 'failed', 'unsupported'];

export function recommend(
  catalogue: readonly ModelSpec[],
  test: DeviceTestResult | null,
  runs: readonly RunRecord[],
  calibration: Calibration = DEFAULT_CALIBRATION,
  /** The device's memory limit (budget.ts); without one (no memory check yet), fit isn't judged. */
  budget?: Budget,
  /** Expert streaming on (budget.ts's streamingEnabled by default). */
  streaming = streamingEnabled(),
): Recommendation[] {
  const artifactFor = (model: ModelSpec) => (test ? pickArtifact(model, test.features) : undefined);
  // Only measurements of the file this device would use (models move to new files);
  // without a device test, of any of the model's current files.
  const measured = new Map(
    catalogue.map((m) => {
      const picked = artifactFor(m);
      const files = new Set((picked ? [picked] : m.artifacts).map(artifactFingerprint));
      return [m.id, summarise(runs.filter((r) => r.modelId === m.id && r.fileSha256 !== undefined && files.has(r.fileSha256)))];
    }),
  );

  // How far reality is from the formulas on this device, from models that have run here.
  const ratios: number[] = [];
  if (test) {
    for (const model of catalogue) {
      const m = measured.get(model.id)!;
      const artifact = artifactFor(model);
      if (!m.tokensPerSecond || !artifact) continue;
      ratios.push(m.tokensPerSecond / estimateModel(model, artifact, test, calibration).tokensPerSecond);
    }
  }
  const correction = ratios.length
    ? Math.min(3, Math.max(1 / 3, Math.exp(ratios.reduce((s, r) => s + Math.log(r), 0) / ratios.length)))
    : 1;

  const quality = chatQuality(catalogue);
  const score = (r: Recommendation) => quality.get(r.model.id)?.score ?? 0;
  const list = catalogue.map((model) => judge(model, artifactFor(model), test, measured.get(model.id)!, correction, calibration, budget, streaming));
  list.sort(
    (a, b) =>
      VERDICT_ORDER.indexOf(a.verdict) - VERDICT_ORDER.indexOf(b.verdict) ||
      score(b) - score(a) ||
      speed(b) - speed(a),
  );
  // Nothing is recommended before a benchmark.
  const pick = test ? pickRecommended(list) : null;
  if (pick) {
    pick.rec.recommended = true;
    pick.rec.pick = pick.why;
  }
  return list;
}

/**
 * The one model to recommend (from a benchmarked list sorted best first): the best one that runs well; else, of the
 * slow ones, the best that manages VERY_SLOW_TOKENS_PER_SECOND, or the fastest; as a last resort, the smallest one
 * that might fit. Never an experimental model.
 */
function pickRecommended(list: Recommendation[]): { rec: Recommendation; why: NonNullable<Recommendation['pick']> } | null {
  const candidates = list.filter((r) => !r.model.experimental);
  const good = candidates.find((r) => r.verdict === 'good');
  if (good) return { rec: good, why: 'runs-well' };
  const slow = candidates.filter((r) => r.verdict === 'slow');
  if (slow.length) {
    const usable = slow.find((r) => speed(r) >= VERY_SLOW_TOKENS_PER_SECOND);
    return { rec: usable ?? [...slow].sort((a, b) => speed(b) - speed(a))[0]!, why: 'least-slow' };
  }
  const smallest = candidates
    .filter((r) => r.verdict === 'may-not-fit')
    .sort((a, b) => minWeightBytes(a.model) - minWeightBytes(b.model))[0];
  return smallest ? { rec: smallest, why: 'smallest' } : null;
}

function minWeightBytes(model: ModelSpec): number {
  return Math.min(...model.artifacts.map(loadedWeightBytes));
}

/** Generation speed to show: measured when available, else estimated. */
export function speed(r: Recommendation): number {
  return r.measured.tokensPerSecond ?? r.estimate?.tokensPerSecond ?? 0;
}

/** First artifact (preferred first) whose GPU features this device has. */
export function pickArtifact(model: ModelSpec, features: readonly string[]): ModelArtifact | undefined {
  const have = new Set(features);
  return model.artifacts.find((a) => a.requiredGpuFeatures.every((f) => have.has(f)));
}

function judge(
  model: ModelSpec,
  artifact: ModelArtifact | undefined,
  test: DeviceTestResult | null,
  measured: Measured,
  correction: number,
  calibration: Calibration,
  budget?: Budget,
  streaming = false,
): Recommendation {
  const base = { model, measured, recommended: false } as const;
  // Nothing is judged before a benchmark, not even from earlier runs.
  if (!test) return { ...base, verdict: 'untested', reasons: ['Run the benchmark to see how this model should run here.'] };

  if (!artifact) {
    return { ...base, verdict: 'unsupported', reasons: ['This GPU lacks the 16-bit maths support this model needs.'] };
  }
  // Only the whole-buffer size is a hard limit: ONNX Runtime copes with tensors larger than the
  // per-binding limit (Qwen3 0.6B and SmolLM2 1.7B ran correctly with bindings capped at 128 MB, as on some phones).
  const limit = test.limits.maxBufferSize;
  if (artifact.engine === 'onnx-webgpu' && artifact.largestTensorBytes > limit) {
    return {
      ...base,
      artifact,
      verdict: 'unsupported',
      reasons: [`One block of its weights (${mb(artifact.largestTensorBytes)}) is larger than this GPU allows in one piece (${mb(limit)}).`],
    };
  }

  const raw = estimateModel(model, artifact, test, calibration);
  const estimate: ModelEstimate = {
    ...raw,
    tokensPerSecond: raw.tokensPerSecond * correction,
    firstTokenShortMs: raw.firstTokenShortMs / correction,
    firstTokenLongMs: raw.firstTokenLongMs / correction,
  };
  const memoryBytes = memoryNeeded(model, artifact);
  const withEstimate = { ...base, artifact, estimate, memoryBytes };

  if (measured.lastLoadFailed) {
    return { ...withEstimate, verdict: 'failed', reasons: [failureReason(measured.lastLoadFailed.code)] };
  }

  // Running here proves it fits.
  const ranHere = measured.replies > 0;
  const experimental = model.experimental ? [model.experimental] : [];
  // A mixture of experts too big as a whole, whose other weights fit: its experts are read from storage as needed.
  // Slower than the estimate (which assumes everything on the GPU) until it has been measured here.
  if (budget && !measured.tokensPerSecond && streamPlan(model, artifact, budget.bytes, { enabled: streaming })) {
    return { ...withEstimate, verdict: 'slow', reasons: [...experimental, STREAMED_REASON] };
  }
  // Over the device's memory limit (budget.ts), even with the shortest conversation: it may not fit.
  if (!ranHere && budget && memoryBytes > budget.bytes) {
    const fit = fitChat(model, artifact, budget.bytes);
    if (fit.status === 'too-big') {
      return {
        ...withEstimate,
        verdict: 'may-not-fit',
        reasons: [...experimental, `Needs about ${mb(fit.bytes)} of memory; ${limitText(budget, mb)}.`],
      };
    }
  }

  const tps = measured.tokensPerSecond ?? estimate.tokensPerSecond;
  const longMs = measured.firstTokenLongMs ?? estimate.firstTokenLongMs;
  const source = measured.tokensPerSecond ? 'measured' : 'estimated';
  if (tps >= GOOD_TOKENS_PER_SECOND && longMs <= GOOD_LONG_PROMPT_MS) {
    return { ...withEstimate, verdict: 'good', reasons: [...experimental, `About ${Math.round(tps)} tokens per second (${source}).`] };
  }
  const reasons: string[] = [...experimental];
  if (tps < GOOD_TOKENS_PER_SECOND) {
    reasons.push(
      tps < VERY_SLOW_TOKENS_PER_SECOND
        ? `Very slow: about ${tps.toFixed(1)} tokens per second (${source}).`
        : `About ${Math.round(tps)} tokens per second (${source}), slower than comfortable reading.`,
    );
  }
  if (longMs > GOOD_LONG_PROMPT_MS) reasons.push(`Long messages take about ${Math.round(longMs / 1000)} s before the reply starts.`);
  return { ...withEstimate, verdict: 'slow', reasons };
}

function summarise(runs: readonly RunRecord[]): Measured {
  const loads = runs.filter((r) => r.kind === 'load').sort((a, b) => a.at - b.at);
  const last = loads[loads.length - 1];
  const replies = runs.filter((r) => r.kind === 'reply' && r.ok);
  const typical = replies.filter((r) => (r.generatedTokens ?? 0) >= 16 && (r.promptTokens ?? 0) <= 1000);
  const short = replies.filter((r) => (r.promptTokens ?? 0) <= 400 && r.firstTokenMs);
  const long = replies.filter((r) => (r.promptTokens ?? 0) >= 1000 && r.firstTokenMs);
  return {
    tokensPerSecond: median(typical.map((r) => r.tokensPerSecond ?? 0).filter((v) => v > 0)),
    firstTokenShortMs: median(short.map((r) => r.firstTokenMs!)),
    firstTokenLongMs: median(long.map((r) => r.firstTokenMs!)),
    replies: replies.length,
    lastLoadFailed:
      last && !last.ok && last.errorCode && RUN_FAILURES.has(last.errorCode) ? { code: last.errorCode, at: last.at } : undefined,
  };
}

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function failureReason(code: string): string {
  if (code === 'device-lost') return "The GPU stopped responding while running this model here; it may need more memory than is free.";
  if (code === 'out-of-memory') return 'This model ran out of GPU memory here last time.';
  if (code === 'validation-failed') return 'This model loaded but produced invalid output on this GPU.';
  if (code === 'runtime-crashed') return 'The runtime crashed while running this model here.';
  if (code === 'page-killed') return 'The browser closed the page while this model was loading or running here, most likely for lack of memory.';
  return "This model didn't load on this device last time.";
}

function mb(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}
