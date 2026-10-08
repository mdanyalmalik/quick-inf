import { useSyncExternalStore } from 'react';
import { getRuntimeClient } from '../../runtime/client';
import type { ModelStorage, RuntimeStatus, StorageReport } from '../../runtime/worker/protocol';

export function useRuntimeStatus(): RuntimeStatus {
  const client = getRuntimeClient();
  return useSyncExternalStore(client.subscribe, client.getStatus);
}

export function useStorage(): StorageReport {
  const client = getRuntimeClient();
  return useSyncExternalStore(client.subscribe, client.getStorage);
}

/** What is saved on this device for one model, if anything. */
export function useModelStorage(modelId: string): ModelStorage | undefined {
  return useStorage().models[modelId];
}

const LOADING_STATES = new Set(['checking', 'waiting', 'downloading', 'reading', 'verifying', 'loading', 'validating']);

/** How the runtime relates to one particular model. */
export function modelRuntimeState(
  status: RuntimeStatus,
  modelId: string,
): 'not-loaded' | 'loading' | 'ready' | 'busy' | 'failed' {
  if (status.modelId !== modelId) return 'not-loaded';
  if (status.state === 'ready') return 'ready';
  if (status.state === 'generating' || status.state === 'stopping') return 'busy';
  if (status.state === 'failed') return 'failed';
  if (LOADING_STATES.has(status.state)) return 'loading';
  return 'not-loaded';
}
