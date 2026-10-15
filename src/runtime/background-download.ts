// Downloads that carry on when the person leaves the page or closes the tab, on phones and tablets whose browser
// supports it (Background Fetch: Chrome, Edge and Samsung Internet on Android; not Safari or Firefox). The browser's
// download manager fetches a model's missing files and shows its own progress notification; the service worker
// (public/background-download-sw.js) keeps them in a cache, and the model store moves them into its storage on the
// next load, checking them as usual (src/store/fetch-artifact.ts). Elsewhere, downloads run in the page as before.

import { isMobileDevice } from '../device/mobile';
import { simulation } from '../device/simulation';
import { BACKGROUND_CACHE } from '../store/fetch-artifact';
import type { DownloadPlan } from './worker/protocol';

const SERVICE_WORKER = `${import.meta.env.BASE_URL}background-download-sw.js`;
/** Free space to leave beyond the files (each is later moved into the model store, briefly needing room twice). */
const SPACE_MARGIN_BYTES = 512 * 1024 * 1024;
const READY_TIMEOUT_MS = 10_000;

export type BackgroundOutcome = 'done' | 'failed' | 'aborted' | 'unavailable';

type BackgroundFetchRegistration = EventTarget & {
  id: string;
  downloaded: number;
  downloadTotal: number;
  result: '' | 'success' | 'failure';
  failureReason: string;
  abort(): Promise<boolean>;
};
type BackgroundFetchManager = {
  fetch(id: string, requests: string[], options: { title: string; downloadTotal: number; icons: object[] }): Promise<BackgroundFetchRegistration>;
  get(id: string): Promise<BackgroundFetchRegistration | undefined>;
};

/**
 * Phones and tablets only, where leaving the browser pauses the page: on computers a background tab keeps
 * downloading, and the page's own download resumes mid-file (the browser's starts each file again).
 */
export function backgroundDownloadsSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    isMobileDevice() &&
    window.isSecureContext &&
    'serviceWorker' in navigator &&
    'BackgroundFetchManager' in window &&
    !simulation()?.noBackgroundFetch
  );
}

let registration: Promise<ServiceWorkerRegistration | null> | null = null;

/** Registers the service worker (once); null where it can't be. */
export function backgroundRegistration(): Promise<ServiceWorkerRegistration | null> {
  if (!backgroundDownloadsSupported()) return Promise.resolve(null);
  registration ??= navigator.serviceWorker
    .register(SERVICE_WORKER, { scope: import.meta.env.BASE_URL })
    // `ready` never settles if the worker can't activate: give up after a while (downloads then run in the page).
    .then(() => Promise.race([navigator.serviceWorker.ready, new Promise<null>((resolve) => setTimeout(() => resolve(null), READY_TIMEOUT_MS))]))
    .catch(() => null);
  return registration;
}

const idFor = (modelId: string) => `model:${modelId}`;

// Each background download shows its own notification, which stays after it ends. One that failed isn't started
// again for the same model in this tab (it would add a notification per attempt): the page downloads the rest.
const FAILED_KEY = 'quick-inf:background-failed';
function failedBefore(modelId: string): boolean {
  try {
    return (JSON.parse(sessionStorage.getItem(FAILED_KEY) ?? '[]') as string[]).includes(modelId);
  } catch {
    return false;
  }
}
function noteFailed(modelId: string): void {
  try {
    const ids = JSON.parse(sessionStorage.getItem(FAILED_KEY) ?? '[]') as string[];
    if (!ids.includes(modelId)) sessionStorage.setItem(FAILED_KEY, JSON.stringify([...ids, modelId]));
  } catch {
    // Without storage it may be tried again on the next load; nothing else changes.
  }
}

/**
 * Downloads `plan`'s files with the browser's download manager, reporting bytes as they arrive. Resolves once the
 * files are in the cache ('done'), when the browser gave up ('failed'; what arrived whole is kept), when the
 * person or `signal` stopped it ('aborted'), or at once when it can't run here ('unavailable').
 */
export async function downloadInBackground(
  modelId: string,
  title: string,
  plan: DownloadPlan,
  onProgress: (loadedBytes: number, totalBytes: number) => void,
  signal: AbortSignal,
): Promise<BackgroundOutcome> {
  const reg = await backgroundRegistration();
  const manager = (reg as (ServiceWorkerRegistration & { backgroundFetch?: BackgroundFetchManager }) | null)?.backgroundFetch;
  if (!manager || !plan.canSave) return 'unavailable';
  const running = await manager.get(idFor(modelId)).catch(() => undefined);
  if (!running && failedBefore(modelId)) return 'unavailable';

  // Files already downloaded in the background (on an earlier visit) needn't be fetched again.
  const cache = await caches.open(BACKGROUND_CACHE);
  const files: DownloadPlan['files'] = [];
  for (const file of plan.files) if (!(await cache.match(file.url))) files.push(file);
  if (files.length === 0) return 'done';
  const total = files.reduce((n, f) => n + f.sizeBytes, 0);
  if (!(await hasSpaceFor(total + Math.max(...files.map((f) => f.sizeBytes))))) return 'unavailable';

  let bgFetch: BackgroundFetchRegistration;
  try {
    bgFetch =
      running ??
      (await manager.fetch(
        idFor(modelId),
        files.map((f) => f.url),
        { title, downloadTotal: total, icons: [{ src: `${import.meta.env.BASE_URL}apple-touch-icon.png`, sizes: '180x180', type: 'image/png' }] },
      ));
  } catch {
    // Refused (no space, notifications blocked by policy, …): the page downloads it instead.
    noteFailed(modelId);
    return 'unavailable';
  }

  return new Promise<BackgroundOutcome>((resolve) => {
    let settled = false;
    const finish = (outcome: BackgroundOutcome) => {
      if (settled) return;
      settled = true;
      if (outcome === 'failed') noteFailed(modelId);
      bgFetch.removeEventListener('progress', onBgProgress);
      navigator.serviceWorker.removeEventListener('message', onMessage);
      signal.removeEventListener('abort', onAbort);
      clearInterval(poll);
      resolve(outcome);
    };
    // The service worker says when the files are in the cache; until it does, also look for them there.
    const allCached = async () => {
      for (const f of files) if (!(await cache.match(f.url))) return false;
      return true;
    };
    const onBgProgress = () => {
      onProgress(bgFetch.downloaded, total);
      if (bgFetch.result === 'failure') finish(bgFetch.failureReason === 'aborted' ? 'aborted' : 'failed');
    };
    const onMessage = (e: MessageEvent<{ kind?: string; id?: string; state?: string }>) => {
      if (e.data?.kind !== 'background-download' || e.data.id !== bgFetch.id) return;
      if (e.data.state === 'done') finish('done');
      else if (e.data.state === 'aborted') finish('aborted');
      else finish('failed');
    };
    const onAbort = () => {
      void bgFetch.abort().catch(() => {});
      finish('aborted');
    };
    const poll = setInterval(() => {
      if (bgFetch.result === 'success') void allCached().then((all) => all && finish('done'));
    }, 1000);
    bgFetch.addEventListener('progress', onBgProgress);
    navigator.serviceWorker.addEventListener('message', onMessage);
    signal.addEventListener('abort', onAbort);
    onBgProgress();
  });
}

async function hasSpaceFor(bytes: number): Promise<boolean> {
  try {
    const { quota, usage } = await navigator.storage.estimate();
    return quota === undefined || usage === undefined || quota - usage >= bytes + SPACE_MARGIN_BYTES;
  } catch {
    return true;
  }
}
