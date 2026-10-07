import { useCallback, useState } from 'react';

export interface WidthBounds {
  initial: number;
  min: number;
  max: number;
}

// Panel widths are a per-browser convenience, so localStorage is enough here.
function readStored(key: string, bounds: WidthBounds): number {
  try {
    const value = Number(localStorage.getItem(key));
    return value ? clamp(value, bounds) : bounds.initial;
  } catch {
    return bounds.initial;
  }
}

function clamp(value: number, { min, max }: WidthBounds): number {
  return Math.min(max, Math.max(min, value));
}

export function useResizableWidth(storageKey: string, bounds: WidthBounds) {
  const [width, setWidthState] = useState(() => readStored(storageKey, bounds));

  const setWidth = useCallback(
    (next: number) => {
      const clamped = clamp(Math.round(next), bounds);
      setWidthState(clamped);
      try {
        localStorage.setItem(storageKey, String(clamped));
      } catch {
        // Storage unavailable: the width still applies for this visit.
      }
    },
    [storageKey, bounds],
  );

  return [width, setWidth] as const;
}
