// What has really happened on this device: model loads (success or failure)
// and reply speeds, plus the latest device test. Kept per "environment" (GPU,
// browser version, runtime version), because any of those changing can change
// speed or whether a model runs at all.

import type { DeviceTestResult } from '../device/device-test';
import packageJson from '../../package.json';
import { dbDeleteMany, dbGet, dbGetAll, dbGetAllByIndex, dbPut } from '../store/db';

export interface RunRecord {
  id: string;
  at: number;
  env: string;
  modelId: string;
  artifactId?: string;
  /**
   * SHA-256 of the model file measured. Artifact ids are reused when the catalogue
   * moves a model to a new file, so only this says which file a measurement is
   * for; runs without it (recorded before it existed) aren't used.
   */
  fileSha256?: string;
  kind: 'load' | 'reply';
  ok: boolean;
  errorCode?: string;
  /** Loads. */
  loadMs?: number;
  downloadedBytes?: number;
  /** Replies. */
  promptTokens?: number;
  generatedTokens?: number;
  tokensPerSecond?: number;
  firstTokenMs?: number;
}

export interface DeviceTestRecord {
  env: string;
  result: DeviceTestResult;
}

export interface GpuIdentity {
  vendor: string;
  architecture: string;
  description: string;
}

/** Identifies the GPU + browser version + runtime version the records apply to. */
export function environmentKey(gpu: GpuIdentity, userAgent: string): string {
  const browser =
    /(Edg|Chrome|Firefox|Version)\/(\d+)/.exec(userAgent)?.slice(1, 3).join(' ').replace('Version', 'Safari') ?? 'unknown';
  // The pinned runtime version (read from package.json so the page doesn't load the runtime itself).
  const runtime = packageJson.dependencies['onnxruntime-web'];
  return [gpu.vendor, gpu.architecture, gpu.description, browser, `ort ${runtime}`].join(' | ');
}

export async function readRuns(env: string): Promise<RunRecord[]> {
  return dbGetAllByIndex<RunRecord>('runs', 'byEnv', IDBKeyRange.only(env));
}

export async function writeRun(record: RunRecord): Promise<void> {
  await dbPut('runs', record);
}

/** Measurements kept per device, model and kind (loads or replies); older ones are dropped. */
export const MAX_RUNS_PER_KIND = 50;

/**
 * Which runs to delete: those for files no longer in the catalogue (or recorded
 * without a file hash), and all but the newest MAX_RUNS_PER_KIND per
 * environment, model and kind.
 */
export function runsToPrune(runs: readonly RunRecord[], currentFiles: ReadonlySet<string>): string[] {
  const drop: string[] = [];
  const groups = new Map<string, RunRecord[]>();
  for (const run of runs) {
    if (!run.fileSha256 || !currentFiles.has(run.fileSha256)) {
      drop.push(run.id);
      continue;
    }
    const key = `${run.env}\n${run.modelId}\n${run.kind}`;
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => b.at - a.at);
    for (const run of group.slice(MAX_RUNS_PER_KIND)) drop.push(run.id);
  }
  return drop;
}

/** Deletes stale and surplus runs (see runsToPrune). Returns how many. */
export async function pruneRuns(currentFiles: ReadonlySet<string>): Promise<number> {
  const drop = runsToPrune(await dbGetAll<RunRecord>('runs'), currentFiles);
  await dbDeleteMany('runs', drop);
  return drop.length;
}

export async function readDeviceTest(env: string): Promise<DeviceTestResult | null> {
  return (await dbGet<DeviceTestRecord>('deviceTests', env))?.result ?? null;
}

export async function writeDeviceTest(env: string, result: DeviceTestResult): Promise<void> {
  await dbPut<DeviceTestRecord>('deviceTests', { env, result });
}
