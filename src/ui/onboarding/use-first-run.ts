import { useCallback, useState } from 'react';

// Temporary: a localStorage flag until the settings store in IndexedDB exists.
const STORAGE_KEY = 'quick-inf:onboarding-complete';

function readComplete(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

export function useFirstRun() {
  const [complete, setComplete] = useState(readComplete);

  const markComplete = useCallback(() => {
    try {
      localStorage.setItem(STORAGE_KEY, '1');
    } catch {
      // Storage unavailable (private mode, blocked site data): prompt again next visit.
    }
    setComplete(true);
  }, []);

  return { isFirstRun: !complete, markComplete };
}
