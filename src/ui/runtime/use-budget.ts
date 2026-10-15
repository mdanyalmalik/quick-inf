import { useSyncExternalStore } from 'react';
import type { Budget } from '../../recommend/budget';
import { currentBudget, memoryEvidence } from '../../recommend/memory-evidence';

/** This device's memory budget (see recommend/budget.ts), or null until the device has been probed. */
export function useBudget(): Budget | null {
  return useSyncExternalStore(memoryEvidence.subscribe, currentBudget);
}
