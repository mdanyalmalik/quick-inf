import { describe, expect, it } from 'vitest';
import { artifactFingerprint, catalogue, findModel, type ModelSpec } from '../catalogue/models';
import type { DeviceTestResult } from '../device/device-test';
import { estimateModel } from './estimate';
import { pickArtifact, recommend } from './recommend';
import { MAX_RUNS_PER_KIND, runsToPrune, type RunRecord } from './run-records';

// Device test averages on the owner's Mac (Apple GPU, Chrome 155), the calibration machine.
const MAC: DeviceTestResult = {
  version: 1,
  ranAt: 0,
  durationMs: 250,
  gpu: { vendor: 'apple', architecture: 'metal-3', description: '' },
  features: ['shader-f16', 'timestamp-query'],
  timing: 'timestamp',
  bandwidthGBps: 92.7,
  computeGflops: 408,
  computePrecision: 'f16',
  dispatchUs: 1.64,
  roundTripMs: 0.25,
  memory: { status: 'ok', testedBytes: 2.3e9 },
  limits: { maxBufferSize: 4294967292, maxStorageBufferBindingSize: 4294967292 },
  deviceMemoryGb: 16,
};

// Measured on the same Mac, in the app (sampling as configured per model), for the
// version-1 (legacy) formula. Qwen2.5 0.5B has since moved to another export.
const MEASURED_TPS: Record<string, number> = {
  'smollm2-360m-instruct': 109.6,
  'qwen3-0.6b': 80.5,
  'smollm2-1.7b-instruct': 55.0,
};

// Version-2 device tests (median of 3 rounds) on the two calibration devices, and
// generation speed measured there by phone-suite.html (mean of 3 sampled runs).
const MAC_V2: DeviceTestResult = {
  ...MAC,
  version: 2,
  bandwidthGBps: 103,
  matvecGBps: 100,
  computeGflops: 809,
  dispatchUs: 0.85,
  stepUs: 2.9,
  roundTripMs: 0.25,
  rounds: 3,
};
const PHONE_V2: DeviceTestResult = {
  ...MAC_V2,
  gpu: { vendor: 'qualcomm', architecture: 'adreno-7xx', description: '' },
  bandwidthGBps: 58.5,
  matvecGBps: 25.6,
  computeGflops: 68.1,
  dispatchUs: 1.5,
  stepUs: 9.2,
  roundTripMs: 3.4,
  limits: { maxBufferSize: 2147483648, maxStorageBufferBindingSize: 134217728 },
  deviceMemoryGb: 8,
};
const MEASURED_V2: [string, DeviceTestResult, number][] = [
  ['smollm2-360m-instruct', MAC_V2, 117.1],
  ['qwen2.5-0.5b-instruct', MAC_V2, 74.7],
  ['qwen3-0.6b', MAC_V2, 78.5],
  ['smollm2-1.7b-instruct', MAC_V2, 54.7],
  ['smollm2-360m-instruct', PHONE_V2, 26.5],
  ['qwen2.5-0.5b-instruct', PHONE_V2, 15.3],
  ['qwen3-0.6b', PHONE_V2, 14.1],
  ['smollm2-1.7b-instruct', PHONE_V2, 12.2],
];

const onnxArtifact = (model: ModelSpec, test: DeviceTestResult) => pickArtifact(model, test.features)!;

const verdicts = (test: DeviceTestResult, runs: RunRecord[] = []) =>
  Object.fromEntries(recommend(catalogue, test, runs).map((r) => [r.model.id, r.verdict]));

describe('estimates', () => {
  it.each(Object.entries(MEASURED_TPS))('%s: version-1 test: generation speed within 25%% of measured on the calibration machine', (id, measured) => {
    const model = findModel(id)!;
    const estimate = estimateModel(model, onnxArtifact(model, MAC), MAC);
    expect(estimate.tokensPerSecond / measured).toBeGreaterThan(0.75);
    expect(estimate.tokensPerSecond / measured).toBeLessThan(1.25);
  });

  it.each(MEASURED_V2)('version-2 test, %s: generation speed near measured (Mac and phone)', (id, test, measured) => {
    const model = findModel(id)!;
    const ratio = estimateModel(model, onnxArtifact(model, test), test).tokensPerSecond / measured;
    // The fit's worst cases are the Qwen models on the phone (+33%).
    expect(ratio).toBeGreaterThan(0.8);
    expect(ratio).toBeLessThan(1.4);
  });

  it('a longer conversation is slower, and a long prompt takes longer than a short one', () => {
    const model = findModel('qwen3-0.6b')!;
    const e = estimateModel(model, onnxArtifact(model, MAC), MAC);
    expect(e.firstTokenLongMs).toBeGreaterThan(e.firstTokenShortMs * 5);
  });
});

describe('recommend', () => {
  it('on the calibration machine models up to 3.5B (active) run well, and the strongest of those is recommended', () => {
    // The device test checks as much memory as the largest (non-experimental) model needs.
    const list = recommend(catalogue, { ...MAC_V2, memory: { status: 'ok', testedBytes: 9e9 } }, []);
    const active = (m: ModelSpec) => m.architecture.activeParameterCount ?? m.architecture.parameterCount;
    const big = (r: { model: ModelSpec }) => active(r.model) > 3.5e9 && !r.model.experimental;
    expect(list.filter((r) => !big(r) && !r.model.experimental).every((r) => r.verdict === 'good')).toBe(true);
    // The 3.8-7B models: long prompts take over 8 s (the mixture-of-experts 8B, 1.5B active, is quick).
    // (or, for the 14B models, may not fit: they need more than the 9 GB checked).
    expect(list.filter(big).every((r) => r.verdict === 'slow' || (r.verdict === 'may-not-fit' && r.estimate!.memoryBytes > 9e9))).toBe(true);
    // LFM2 24B: more memory than was checked, and never recommended.
    const huge = list.find((r) => r.model.experimental)!;
    expect(huge.verdict).toBe('may-not-fit');
    expect(huge.reasons[0]).toMatch(/Untested/);
    expect(list.filter((r) => r.recommended).map((r) => r.model.id)).toEqual(['lfm2-8b-a1b']);
    // Within a verdict, ordered by quality, best first.
    for (const verdict of ['good', 'slow'] as const) {
      const ranks = list.filter((r) => r.verdict === verdict).map((r) => r.model.quality.rank);
      expect(ranks).toEqual([...ranks].sort((a, b) => b - a));
    }
  });

  it('without a device test, nothing is judged', () => {
    expect(new Set(Object.values(verdicts(null as unknown as DeviceTestResult)))).toEqual(new Set(['untested']));
  });

  it('without fp16 support, models fall back to their q4 files', () => {
    const list = recommend(catalogue, { ...MAC_V2, features: [] }, []);
    for (const r of list) {
      if (r.model.artifacts.some((a) => a.precision === 'q4')) expect(r.artifact?.precision).toBe('q4');
      else expect(r.verdict).toBe('unsupported');
    }
  });

  it('a model needing more memory than the device test checked may not fit, until it has run here', () => {
    const test = { ...MAC_V2, memory: { status: 'ok' as const, testedBytes: 3e9 } };
    expect(verdicts(test)['olmo-3-7b-instruct']).toBe('may-not-fit');
    expect(verdicts(test)['qwen3-0.6b']).toBe('good');
  });

  it('a small maximum buffer size rules out ONNX models whose embedding table is larger', () => {
    const v = verdicts({ ...MAC_V2, limits: { maxBufferSize: 256e6, maxStorageBufferBindingSize: 128 * 1024 * 1024 } });
    expect(v['smollm2-360m-instruct']).toBe('good'); // 94 MB embedding
    expect(v['qwen3-0.6b']).toBe('unsupported'); // 311 MB
    expect(v['smollm2-1.7b-instruct']).toBe('good'); // 201 MB
  });

  it('a 128 MB per-binding limit alone (Galaxy S24) rules nothing out', () => {
    const v = verdicts({ ...MAC_V2, limits: { maxBufferSize: 2147483648, maxStorageBufferBindingSize: 134217728 } });
    expect(Object.values(v).every((verdict) => verdict !== 'unsupported')).toBe(true);
  });

  it('a failed memory check below what a model needs marks it as may-not-fit', () => {
    const v = verdicts({ ...MAC_V2, memory: { status: 'failed', testedBytes: 2.3e9, failedAtBytes: 1.1e9 } });
    expect(v['smollm2-1.7b-instruct']).toBe('may-not-fit');
    expect(v['smollm2-360m-instruct']).toBe('good');
  });

  it('a slow device makes the bigger models slow', () => {
    const v = verdicts({ ...MAC_V2, bandwidthGBps: 12, matvecGBps: 12, computeGflops: 150, dispatchUs: 6, stepUs: 10 });
    expect(v['smollm2-1.7b-instruct']).toBe('slow');
    expect(v['smollm2-360m-instruct']).toBe('good');
  });

  it('measured speed replaces the estimate and corrects the others', () => {
    const runs: RunRecord[] = [
      reply('qwen3-0.6b', 40), // half the estimate: this device is slower than the formulas think
    ];
    const list = recommend(catalogue, MAC_V2, runs);
    const qwen = list.find((r) => r.model.id === 'qwen3-0.6b')!;
    const smol = list.find((r) => r.model.id === 'smollm2-360m-instruct')!;
    expect(qwen.measured.tokensPerSecond).toBe(40);
    const uncorrected = recommend(catalogue, MAC_V2, []).find((r) => r.model.id === 'smollm2-360m-instruct')!;
    expect(smol.estimate!.tokensPerSecond).toBeLessThan(uncorrected.estimate!.tokensPerSecond * 0.6);
  });

  it('measurements of a file the model no longer uses are ignored', () => {
    const old = recommend(catalogue, MAC_V2, [reply('qwen2.5-0.5b-instruct', 11.5, 'a-replaced-file')]);
    const legacy = recommend(catalogue, MAC_V2, [{ ...reply('qwen2.5-0.5b-instruct', 11.5), fileSha256: undefined }]);
    for (const list of [old, legacy]) {
      expect(list.find((r) => r.model.id === 'qwen2.5-0.5b-instruct')!.measured.tokensPerSecond).toBeUndefined();
    }
  });

  it('a failed load is remembered until a later load succeeds', () => {
    const failed = load('smollm2-1.7b-instruct', false, 1, 'device-lost');
    expect(verdicts(MAC_V2, [failed])['smollm2-1.7b-instruct']).toBe('failed');
    expect(verdicts(MAC_V2, [failed, load('smollm2-1.7b-instruct', true, 2)])['smollm2-1.7b-instruct']).toBe('good');
    // Network problems don't count against the model.
    expect(verdicts(MAC_V2, [load('smollm2-1.7b-instruct', false, 3, 'download-failed')])['smollm2-1.7b-instruct']).toBe('good');
  });
});

/** The file a model runs from here (its first artifact; the calibration Mac has fp16). */
const fileOf = (modelId: string) => artifactFingerprint(findModel(modelId)!.artifacts[0]!);

function reply(modelId: string, tokensPerSecond: number, fileSha256 = fileOf(modelId)): RunRecord {
  return {
    id: `${modelId}-reply`,
    at: 1,
    env: 'test',
    modelId,
    fileSha256,
    kind: 'reply',
    ok: true,
    promptTokens: 120,
    generatedTokens: 200,
    tokensPerSecond,
    firstTokenMs: 80,
  };
}

function load(modelId: string, ok: boolean, at: number, errorCode?: string): RunRecord {
  return { id: `${modelId}-load-${at}`, at, env: 'test', modelId, fileSha256: fileOf(modelId), kind: 'load', ok, errorCode };
}

describe('runsToPrune', () => {
  const files = new Set(['current']);
  const run = (id: string, at: number, extra: Partial<RunRecord> = {}): RunRecord => ({
    id,
    at,
    env: 'mac',
    modelId: 'm',
    fileSha256: 'current',
    kind: 'reply',
    ok: true,
    ...extra,
  });

  it('drops runs of files no longer in the catalogue, and runs without a file hash', () => {
    const drop = runsToPrune([run('keep', 1), run('old-file', 2, { fileSha256: 'gone' }), run('no-hash', 3, { fileSha256: undefined })], files);
    expect(drop.sort()).toEqual(['no-hash', 'old-file']);
  });

  it(`keeps the newest ${MAX_RUNS_PER_KIND} per device, model and kind`, () => {
    const many = Array.from({ length: MAX_RUNS_PER_KIND + 5 }, (_, i) => run(`r${i}`, i));
    const loads = Array.from({ length: 3 }, (_, i) => run(`l${i}`, i, { kind: 'load' }));
    const otherDevice = Array.from({ length: 3 }, (_, i) => run(`p${i}`, i, { env: 'phone' }));
    const drop = runsToPrune([...many, ...loads, ...otherDevice], files);
    expect(drop.sort()).toEqual(['r0', 'r1', 'r2', 'r3', 'r4']);
  });
});
