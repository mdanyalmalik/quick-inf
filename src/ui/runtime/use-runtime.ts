import { useSyncExternalStore } from 'react';
import { getRuntimeClient } from '../../runtime/client';
import type { RuntimeStatus } from '../../runtime/worker/protocol';

export function useRuntimeStatus(): RuntimeStatus {
  const client = getRuntimeClient();
  return useSyncExternalStore(client.subscribe, client.getStatus);
}

const LOADING_STATES = new Set(['checking', 'downloading', 'verifying', 'loading', 'validating']);

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
