// Keeps the current recommendations up to date: reads the latest device test and
// run records for this environment, records every finished load and reply, and
// runs the device test on request. Framework-free (subscribe/getSnapshot).

import { catalogue } from '../catalogue/models';
import type { DeviceTestResult, DeviceTestStep } from '../device/device-test';
import { getRuntimeClient, type Measurement } from '../runtime/client';
import { RuntimeError } from '../runtime/contract';
import { memoryNeeded } from './estimate';
import { pickArtifact, recommend, type Recommendation } from './recommend';
import {
  environmentKey,
  readDeviceTest,
  readRuns,
  writeDeviceTest,
  writeRun,
  type RunRecord,
} from './run-records';

export interface RecommendationSnapshot {
  /** 'unavailable': WebGPU missing, so nothing can run or be tested. */
  status: 'loading' | 'ready' | 'unavailable';
  test: DeviceTestResult | null;
  recommendations: Recommendation[];
  /** The device test step in progress, if one is running. */
  running: DeviceTestStep | 'starting' | null;
  error: string | null;
}

class RecommendationStore {
  private snapshot: RecommendationSnapshot = {
    status: 'loading',
    test: null,
    recommendations: recommend(catalogue, null, []),
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
          : `The device test didn't finish: ${err instanceof Error ? err.message : String(err)}`;
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
    const [test, runs] = await Promise.all([
      readDeviceTest(this.env).catch(() => null),
      readRuns(this.env).catch(() => [] as RunRecord[]),
    ]);
    this.runs = runs;
    this.update({ status: 'ready', test });
    this.recompute();
    client.onMeasurement((m) => void this.record(m));
  }

  private async record(m: Measurement): Promise<void> {
    if (!this.env) return;
    const base = { id: crypto.randomUUID(), at: Date.now(), env: this.env, modelId: m.modelId };
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
  private memoryToTest(): number {
    let most = 0;
    for (const model of catalogue) {
      const artifact = pickArtifact(model, this.features);
      if (artifact) most = Math.max(most, memoryNeeded(model, artifact));
    }
    return most;
  }

  private recompute(): void {
    this.update({ recommendations: recommend(catalogue, this.snapshot.test, this.runs) });
  }

  private update(patch: Partial<RecommendationSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const l of this.listeners) l();
  }
}

export const recommendationStore = new RecommendationStore();
