// Turns the device test, the catalogue and what has really happened on this
// device into a ranked list of models with a verdict each. Pure: the caller
// supplies the inputs (see recommendation-store.ts).
//
// Measured beats estimated: a model that has run here shows its measured speed,
// and the ratio between measured and estimated speed across those models
// corrects the estimates of the others. A failed load is remembered until a
// later load of the same model succeeds (or the environment changes, which
// gives a fresh set of records).

import { artifactFingerprint, type ModelArtifact, type ModelSpec } from '../catalogue/models';
import type { DeviceTestResult } from '../device/device-test';
import { type Calibration, DEFAULT_CALIBRATION, estimateModel, memoryNeeded, type ModelEstimate } from './estimate';
import type { RunRecord } from './run-records';

export type Verdict = 'good' | 'slow' | 'may-not-fit' | 'failed' | 'unsupported' | 'untested';

/** Generation speed and long-prompt wait that count as "runs well". */
const GOOD_TOKENS_PER_SECOND = 15;
const GOOD_LONG_PROMPT_MS = 8000;
const VERY_SLOW_TOKENS_PER_SECOND = 5;
/** Without a memory check, assume a model fits if it needs at most this share of reported RAM. */
const RAM_SHARE = 0.6;
/** Load failures that mean "this model doesn't run here" (not network or user problems). */
const RUN_FAILURES = new Set(['load-failed', 'validation-failed', 'device-lost', 'runtime-crashed', 'out-of-memory']);

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
  /** The single best choice for this device. */
  recommended: boolean;
}

const VERDICT_ORDER: Verdict[] = ['good', 'slow', 'untested', 'may-not-fit', 'failed', 'unsupported'];

export function recommend(
  catalogue: readonly ModelSpec[],
  test: DeviceTestResult | null,
  runs: readonly RunRecord[],
  calibration: Calibration = DEFAULT_CALIBRATION,
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

  const list = catalogue.map((model) => judge(model, artifactFor(model), test, measured.get(model.id)!, correction, calibration));
  list.sort(
    (a, b) =>
      VERDICT_ORDER.indexOf(a.verdict) - VERDICT_ORDER.indexOf(b.verdict) ||
      b.model.quality.rank - a.model.quality.rank ||
      speed(b) - speed(a),
  );
  const best = list.find((r) => r.verdict === 'good' && !r.model.experimental);
  if (best) best.recommended = true;
  return list;
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
): Recommendation {
  const base = { model, measured, recommended: false } as const;
  if (!test) {
    return measured.lastLoadFailed
      ? { ...base, verdict: 'failed', reasons: [failureReason(measured.lastLoadFailed.code)] }
      : { ...base, verdict: 'untested', reasons: ['Run the benchmark to see how this model should run here.'] };
  }

  if (!artifact) {
    return { ...base, verdict: 'unsupported', reasons: ['This GPU lacks the 16-bit maths support this model needs.'] };
  }
  // Only the whole-buffer size is a hard limit: ONNX Runtime copes with tensors larger than the
  // per-binding limit (Qwen3 0.6B and SmolLM2 1.7B ran correctly with bindings capped at 128 MB, as on a Galaxy S24).
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

  // Running here proves it fits, whatever the memory check says.
  const ranHere = measured.replies > 0;
  const memory = test.memory;
  const experimental = model.experimental ? [model.experimental] : [];
  if (!ranHere && memory.status === 'failed' && memory.failedAtBytes < memoryBytes) {
    return {
      ...withEstimate,
      verdict: 'may-not-fit',
      reasons: [...experimental, `Needs about ${mb(memoryBytes)} of GPU memory; the benchmark could only hold ${mb(memory.failedAtBytes)}.`],
    };
  }
  // The check holds only so much (less on phones, where trying too much can crash the browser).
  if (!ranHere && memory.status === 'ok' && memory.testedBytes < memoryBytes) {
    return {
      ...withEstimate,
      verdict: 'may-not-fit',
      reasons: [...experimental, `Needs about ${mb(memoryBytes)} of GPU memory; the benchmark only checked ${mb(memory.testedBytes)}.`],
    };
  }
  if (memory.status === 'skipped' && test.deviceMemoryGb && memoryBytes > test.deviceMemoryGb * 1e9 * RAM_SHARE) {
    return {
      ...withEstimate,
      verdict: 'may-not-fit',
      reasons: [`Needs about ${mb(memoryBytes)} of memory; this device reports ${test.deviceMemoryGb} GB in total.`],
    };
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
  return "This model didn't load on this device last time.";
}

function mb(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}
