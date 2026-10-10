// Remembers engines that failed to run a model on this device, so the next load
// uses the next artifact (WebLLM → the ONNX engine) instead of failing again.
// Keyed by GPU + browser version: a browser update tries again. Runs in the worker.

import type { EngineId } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import { environmentKey } from '../recommend/run-records';
import { dbDelete, dbGetAll, dbPut } from '../store/db';
import type { RuntimeErrorCode } from './contract';

interface EngineFailure {
  key: string;
  kind: 'engine-failure';
  env: string;
  engine: EngineId;
  modelId: string;
  code: RuntimeErrorCode;
  message: string;
  at: number;
}

/** A remembered failure is retried after this long (drivers and the browser change). */
const FAILURE_TTL_MS = 30 * 24 * 60 * 60 * 1000;

/**
 * Failures that say the engine can't run this model on this device's GPU, as
 * opposed to the network, storage, the user cancelling, or a bug in how the app
 * configured the engine (which a fallback would only hide).
 */
export function isEngineFailure(error: { code: RuntimeErrorCode; message: string }): boolean {
  if (error.code === 'validation-failed' || error.code === 'device-lost' || error.code === 'runtime-crashed') return true;
  return error.code === 'load-failed' && GPU_ERROR.test(error.message);
}

/** Driver and GPU errors as WebGPU, Dawn and WebLLM report them. */
const GPU_ERROR = /VK_ERROR|Vulkan|Dawn|GPUPipelineError|CreateComputePipelines?|shader module|out of memory|\bOOM\b|device (was )?lost|DeviceLost|GPUValidationError|GPUOutOfMemoryError|mapAsync|maxStorageBufferBindingSize|maxBufferSize/i;

export function deviceEnv(device: DeviceReport): string | null {
  return device.gpu ? environmentKey(device.gpu, device.userAgent) : null;
}

function keyOf(env: string, engine: EngineId, modelId: string): string {
  return `engine-failure:${env}:${engine}:${modelId}`;
}

/** Engines that failed for `modelId` on this device. */
export async function failedEngines(env: string | null, modelId: string): Promise<Set<EngineId>> {
  return (await allFailedEngines(env)).get(modelId) ?? new Set();
}

/** Engines that failed on this device, by model id. */
export async function allFailedEngines(env: string | null): Promise<Map<string, Set<EngineId>>> {
  const byModel = new Map<string, Set<EngineId>>();
  if (!env) return byModel;
  const all = await dbGetAll<{ kind?: string }>('engineState').catch(() => []);
  for (const r of all.filter((r) => r.kind === 'engine-failure') as EngineFailure[]) {
    if (r.env !== env || Date.now() - r.at > FAILURE_TTL_MS) continue;
    byModel.set(r.modelId, (byModel.get(r.modelId) ?? new Set()).add(r.engine));
  }
  return byModel;
}

export async function recordEngineFailure(
  env: string | null,
  engine: EngineId,
  modelId: string,
  error: { code: RuntimeErrorCode; message: string },
): Promise<void> {
  if (!env) return;
  const record: EngineFailure = { key: keyOf(env, engine, modelId), kind: 'engine-failure', env, engine, modelId, code: error.code, message: error.message, at: Date.now() };
  await dbPut('engineState', record).catch(() => {});
}

/** Forgets a failure (e.g. the engine later loaded the model fine through the dev override). */
export async function clearEngineFailure(env: string | null, engine: EngineId, modelId: string): Promise<void> {
  if (env) await dbDelete('engineState', keyOf(env, engine, modelId)).catch(() => {});
}
