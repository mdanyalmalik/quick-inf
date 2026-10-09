// What has really happened on this device: model loads (success or failure)
// and reply speeds, plus the latest device test. Kept per "environment" (GPU,
// browser version, runtime version), because any of those changing can change
// speed or whether a model runs at all.

import type { DeviceTestResult } from '../device/device-test';
import packageJson from '../../package.json';
import { dbGet, dbGetAllByIndex, dbPut } from '../store/db';

export interface RunRecord {
  id: string;
  at: number;
  env: string;
  modelId: string;
  artifactId?: string;
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

export async function readDeviceTest(env: string): Promise<DeviceTestResult | null> {
  return (await dbGet<DeviceTestRecord>('deviceTests', env))?.result ?? null;
}

export async function writeDeviceTest(env: string, result: DeviceTestResult): Promise<void> {
  await dbPut<DeviceTestRecord>('deviceTests', { env, result });
}
