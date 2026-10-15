// Keeps the current recommendations up to date: reads the latest device test and
// run records for this environment, records every finished load and reply, and
// runs the device test on request. Framework-free (subscribe/getSnapshot).

import { isMobileDevice } from '../device/mobile';
import { currentBudget, memoryEvidence } from './memory-evidence';
import { artifactFingerprint, catalogue } from '../catalogue/models';
import type { DeviceTestResult, DeviceTestStep } from '../device/device-test';
import { getRuntimeClient, type Measurement } from '../runtime/client';
import { RuntimeError } from '../runtime/contract';
import { memoryNeeded } from './estimate';
import { pickArtifact, recommend, type Recommendation } from './recommend';
import {
  type RunRecord,
  environmentKey,
  pruneRuns,
  readDeviceTest,
  readRuns,
  writeDeviceTest,
  writeRun,
} from './run-records';

export interface RecommendationSnapshot {
  /** 'unavailable': WebGPU missing, so nothing can run or be tested. */
  status: 'loading' | 'ready' | 'unavailable';
  test: DeviceTestResult | null;
  recommendations: Recommendation[];
  /** The device test step in progress, if one is running. */
  running: DeviceTestStep | 'starting' | null;
  error: string | null;
  /** The saved test's memory check covered less than the test would check now (larger models were added since). */
  memoryCheckOutdated: boolean;
}

class RecommendationStore {
  private snapshot: RecommendationSnapshot = {
    status: 'loading',
    test: null,
    recommendations: recommend(catalogue, null, []),
    memoryCheckOutdated: false,
    running: null,
    error: null,
  };
  private readonly listeners = new Set<() => void>();
  private env: string | null = null;
  private features: string[] = [];
  private runs: RunRecord[] = [];
  private started: Promise<void> | null = null;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): RecommendationSnapshot => this.snapshot;

  /** Identifies this device, loads stored results and starts recording runs. Safe to call repeatedly. */
  init(): Promise<void> {
    this.started ??= this.start();
    return this.started;
  }

  /** Runs the quick device test and stores the result. */
  async runDeviceTest(): Promise<void> {
    await this.init();
    if (!this.env || this.snapshot.running) return;
    this.update({ running: 'starting', error: null });
    try {
      const result = await getRuntimeClient().runDeviceTest(this.memoryToTest(), (step) => this.update({ running: step }));
      await writeDeviceTest(this.env, result).catch(() => {});
      this.update({ running: null, test: result });
      this.recompute();
    } catch (err) {
      const message =
        err instanceof RuntimeError && err.code === 'busy'
          ? err.message
          : `The benchmark didn't finish: ${err instanceof Error ? err.message : String(err)}`;
      this.update({ running: null, error: message });
    }
  }

  private async start(): Promise<void> {
    const client = getRuntimeClient();
    let check;
    try {
      check = await client.check(catalogue[0]!.id);
    } catch {
      this.update({ status: 'unavailable' });
      return;
    }
    if (check.device.webgpu !== 'available' || !check.device.gpu) {
      this.update({ status: 'unavailable' });
      return;
    }
    const gpu = check.device.gpu;
    this.env = environmentKey(gpu, navigator.userAgent);
    this.features = gpu.features;
    await pruneRuns(currentFileHashes()).catch(() => 0);
    const [test, runs] = await Promise.all([
      readDeviceTest(this.env).catch(() => null),
      readRuns(this.env).catch(() => [] as RunRecord[]),
    ]);
    this.runs = runs;
    this.update({ status: 'ready', test });
    this.recompute();
    client.onMeasurement((m) => void this.record(m));
    // The memory budget moves with what runs and fails here.
    memoryEvidence.subscribe(() => this.recompute());
    // A load the browser killed the page during, last visit (see runtime/breadcrumbs.ts): it didn't fit here.
    for (const crash of client.getCrashes()) {
      if (crash.modelId && (crash.step === 'load' || crash.step === 'reply')) {
        void this.record({ kind: 'load', modelId: crash.modelId, ok: false, errorCode: 'page-killed' });
      }
    }
  }

  private async record(m: Measurement): Promise<void> {
    if (!this.env || (m.kind !== 'load' && m.kind !== 'reply')) return;
    const model = catalogue.find((c) => c.id === m.modelId);
    // The file measured: the one loaded, or (for a failed load) the one that would have been.
    const artifactId = m.kind === 'reply' ? m.artifactId : m.ok ? m.result.artifactId : undefined;
    const artifact =
      model && (artifactId ? model.artifacts.find((a) => a.id === artifactId) : pickArtifact(model, this.features));
    const base = { id: crypto.randomUUID(), at: Date.now(), env: this.env, modelId: m.modelId, fileSha256: artifact && artifactFingerprint(artifact) };
    let record: RunRecord;
    if (m.kind === 'load') {
      record = m.ok
        ? {
            ...base,
            kind: 'load',
            ok: true,
            artifactId: m.result.artifactId,
            downloadedBytes: m.result.downloadedBytes,
            loadMs: m.result.timings.sessionMs + m.result.timings.validateMs,
          }
        : { ...base, kind: 'load', ok: false, errorCode: m.errorCode };
    } else {
      const s = m.result.stats;
      record = {
        ...base,
        kind: 'reply',
        ok: true,
        artifactId: m.artifactId,
        promptTokens: s.promptTokens,
        generatedTokens: s.generatedTokens,
        tokensPerSecond: s.decodeTokensPerSecond,
        firstTokenMs: s.timeToFirstTokenMs,
      };
    }
    this.runs = [...this.runs, record];
    this.recompute();
    await writeRun(record).catch(() => {});
  }

  /** GPU memory the memory check should try to hold: what the most demanding runnable model needs. */
  /**
   * GPU memory the device test tries to hold: what the largest model would need
   * (experimental ones aside), at most MOBILE_MEMORY_CHECK_BYTES on phones, where
   * holding too much can crash the browser instead of failing cleanly.
   */
  private memoryToTest(): number {
    let most = 0;
    for (const model of catalogue) {
      if (model.experimental) continue;
      const artifact = pickArtifact(model, this.features);
      if (artifact) most = Math.max(most, memoryNeeded(model, artifact));
    }
    // Never more than the device's budget (holding too much can crash the browser instead of failing cleanly).
    const cap = Math.min(MEMORY_CHECK_CAP_BYTES, currentBudget()?.bytes ?? (isMobileDevice() ? MOBILE_MEMORY_CHECK_BYTES : MEMORY_CHECK_CAP_BYTES));
    return Math.min(most, cap);
  }

  private recompute(): void {
    const { test } = this.snapshot;
    this.update({
      recommendations: recommend(catalogue, test, this.runs, undefined, currentBudget() ?? undefined),
      memoryCheckOutdated: test?.memory.status === 'ok' && test.memory.testedBytes < this.memoryToTest(),
    });
  }

  private update(patch: Partial<RecommendationSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const l of this.listeners) l();
  }
}

export const recommendationStore = new RecommendationStore();

/** Fingerprints of every model file in the catalogue. */
function currentFileHashes(): Set<string> {
  return new Set(catalogue.flatMap((m) => m.artifacts.map(artifactFingerprint)));
}

/** Before the device is probed, the most the memory check holds on a phone (Safari's tab limit is about 1.5 GB). */
const MOBILE_MEMORY_CHECK_BYTES = 1.2e9;
/**
 * The most the memory check ever holds. On shared-memory computers GPU memory is main RAM, so holding the largest
 * model's need (18 GB+) made the benchmark itself swap or crash; bigger models are proven by loading them instead.
 */
const MEMORY_CHECK_CAP_BYTES = 4e9;
