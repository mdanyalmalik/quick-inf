// A speech tab's clip: one per session (speaking again replaces it). The record (text, settings, status) is kept
// in localStorage, the finished audio as a WAV in IndexedDB (`speech`).

import { useEffect, useState, useSyncExternalStore } from 'react';
import type { SpeechSettings } from '../../../catalogue/speech-models';
import { dbGet, dbPut } from '../../../store/db';
import { readSpeechRecord, speechRecordKey as key, writeSpeechRecord, type SpeechRow } from '../../../store/speech-records';

export interface Clip {
  id: string;
  text: string;
  settings: SpeechSettings;
  createdAt: number;
  status: 'generating' | 'done' | 'cancelled' | 'failed';
  /** While generating: chunks of text spoken, of the total, and seconds of audio made so far. */
  progress?: { chunk: number; chunks: number; seconds: number };
  /** Length of the finished audio. */
  seconds?: number;
  error?: { code: string; message: string };
  timings?: { firstAudioMs: number; totalMs: number };
}

const cache = new Map<string, Clip | null>();
const listeners = new Set<() => void>();
/** Object URLs of saved clips, by session (loaded from IndexedDB on first use). */
const urls = new Map<string, { clipId: string; url: string }>();

function read(sessionId: string): Clip | null {
  if (cache.has(sessionId)) return cache.get(sessionId)!;
  let clip = readSpeechRecord(sessionId) as Clip | null;
  // One left running when the page closed didn't finish.
  if (clip?.status === 'generating') clip = { ...clip, status: 'cancelled', progress: undefined };
  cache.set(sessionId, clip);
  return clip;
}

function notify(): void {
  for (const l of listeners) l();
}

export const speechStore = {
  get: read,
  set(sessionId: string, clip: Clip): void {
    cache.set(sessionId, clip);
    // Progress isn't saved: only the start and the end of a run are.
    if (clip.status !== 'generating' || clip.progress?.chunk === 0) {
      try {
        const { progress: _p, ...saved } = clip;
        writeSpeechRecord(sessionId, saved);
      } catch {
        // Not saved; still shown for this visit.
      }
    }
    notify();
  },
  async saveAudio(sessionId: string, clipId: string, wav: Blob): Promise<void> {
    const old = urls.get(sessionId);
    if (old) URL.revokeObjectURL(old.url);
    urls.set(sessionId, { clipId, url: URL.createObjectURL(wav) });
    notify();
    await dbPut<SpeechRow>('speech', { sessionId, clipId, wav });
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

export function useClip(sessionId: string): Clip | null {
  return useSyncExternalStore(speechStore.subscribe, () => read(sessionId));
}

/** The saved audio of a finished clip, as an object URL (null while loading or if missing). */
export function useClipUrl(sessionId: string, clip: Clip | null): string | null {
  const wanted = clip?.status === 'done' ? clip.id : null;
  const known = urls.get(sessionId);
  const [, rerender] = useState(0);
  useEffect(() => {
    if (!wanted || urls.get(sessionId)?.clipId === wanted) return;
    let alive = true;
    void dbGet<SpeechRow>('speech', sessionId).then((row) => {
      if (!alive || !row || row.clipId !== wanted) return;
      urls.set(sessionId, { clipId: wanted, url: URL.createObjectURL(row.wav) });
      rerender((n) => n + 1);
    });
    return () => {
      alive = false;
    };
  }, [sessionId, wanted]);
  return wanted && known?.clipId === wanted ? known.url : null;
}
