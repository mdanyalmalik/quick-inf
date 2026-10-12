// An image tab's image: one per session (generating again replaces it). The record (prompt, settings, seed,
// status) is kept in localStorage; a real model's picture is a PNG in IndexedDB (`images`), while a preview
// model's placeholder is drawn again from the record. Previews of the steps so far live in memory only.

import { useEffect, useState, useSyncExternalStore } from 'react';
import type { ImageSettings } from '../../../catalogue/image-models';
import { dbGet, dbPut } from '../../../store/db';
import { imageRecordKey as key, readImageRecord, writeImageRecord, type ImageRow } from '../../../store/image-records';

export interface Generation {
  id: string;
  prompt: string;
  settings: ImageSettings;
  seed: number;
  createdAt: number;
  status: 'generating' | 'done' | 'cancelled' | 'failed';
  /** Made by a model (its PNG is saved), not a placeholder. */
  real?: boolean;
  /** While generating: steps done and total, and what the step is doing. */
  progress?: { step: number; steps: number; phase?: 'prompt' | 'denoising' | 'decoding' };
  /** Object URL of the latest step's preview (real models, several-step runs; not saved). */
  previewUrl?: string;
  error?: { code: string; message: string };
  timings?: { totalMs: number };
}

const cache = new Map<string, Generation | null>();
const listeners = new Set<() => void>();

function read(sessionId: string): Generation | null {
  if (cache.has(sessionId)) return cache.get(sessionId)!;
  let generation: Generation | null = null;
  try {
    generation = readImageRecord(sessionId) as Generation | null;
    // One left running when the page closed didn't finish.
    if (generation?.status === 'generating') generation = { ...generation, status: 'cancelled', progress: undefined };
  } catch {
    generation = null;
  }
  cache.set(sessionId, generation);
  return generation;
}

function notify(): void {
  for (const l of listeners) l();
}

export const imageStore = {
  get: read,
  set(sessionId: string, generation: Generation): void {
    const previous = cache.get(sessionId);
    if (previous?.previewUrl && previous.previewUrl !== generation.previewUrl) URL.revokeObjectURL(previous.previewUrl);
    cache.set(sessionId, generation);
    // Progress isn't saved: only the start and the end of a run are.
    if (generation.status !== 'generating' || generation.progress?.step === 0) {
      try {
        const { progress: _p, previewUrl: _u, ...saved } = generation;
        writeImageRecord(sessionId, saved);
      } catch {
        // Not saved; still shown for this visit.
      }
    }
    notify();
  },
  /** Saves a real model's finished picture. */
  async saveImage(sessionId: string, generationId: string, png: Blob): Promise<void> {
    const old = urls.get(sessionId);
    if (old) URL.revokeObjectURL(old.url);
    urls.set(sessionId, { generationId, url: URL.createObjectURL(png) });
    notify();
    await dbPut<ImageRow>('images', { sessionId, generationId, png });
  },
  forget(sessionId: string): void {
    cache.delete(sessionId);
    const old = urls.get(sessionId);
    if (old) URL.revokeObjectURL(old.url);
    urls.delete(sessionId);
    try {
      localStorage.removeItem(key(sessionId));
    } catch {
      // Nothing to remove.
    }
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

/** Object URLs of saved pictures, by session (loaded from IndexedDB on first use). */
const urls = new Map<string, { generationId: string; url: string }>();

export function useGeneration(sessionId: string): Generation | null {
  return useSyncExternalStore(imageStore.subscribe, () => read(sessionId));
}

/** The saved picture of a real model's finished generation, as an object URL (null while loading or if missing). */
export function useImageUrl(sessionId: string, generation: Generation | null): string | null {
  const wanted = generation?.real && generation.status === 'done' ? generation.id : null;
  const known = urls.get(sessionId);
  const [, rerender] = useState(0);
  useEffect(() => {
    if (!wanted || urls.get(sessionId)?.generationId === wanted) return;
    let alive = true;
    void dbGet<ImageRow>('images', sessionId).then((row) => {
      if (!alive || !row || row.generationId !== wanted) return;
      urls.set(sessionId, { generationId: wanted, url: URL.createObjectURL(row.png) });
      rerender((n) => n + 1);
    });
    return () => {
      alive = false;
    };
  }, [sessionId, wanted]);
  return wanted && known?.generationId === wanted ? known.url : null;
}
