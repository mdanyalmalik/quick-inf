// Artifact download primitives. `streamArtifact` streams one file to a caller-
// supplied writer, optionally resuming from a byte offset with an HTTP Range
// request. The OPFS model store writes to disk with it; `fetchArtifact` keeps
// a file in memory (tests, diagnostics page).

import type { ArtifactFile } from '../catalogue/models';

export class ArtifactError extends Error {
  constructor(
    readonly code: 'download-failed' | 'integrity-mismatch' | 'cancelled',
    message: string,
    /** A network hiccup, a stall or an overloaded server: trying again (resuming) may well work. */
    readonly transient = false,
  ) {
    super(message);
  }
}

/** No data for this long means the connection has stalled: give up on it and resume on a new one. */
export const STALL_MS = 60_000;

/** HTTP statuses that mean the file isn't there or isn't ours to read, so retrying can't help (as Transformers.js treats them). */
const PERMANENT: Record<number, string> = {
  401: 'access to the file was refused (401)',
  403: 'access to the file was refused (403)',
  404: 'the file was not found (404)',
  410: 'the file is gone (410)',
};

/** Statuses worth retrying: timeouts, rate limits, server errors. */
function isTransientStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

/** Wait before retry number `attempt` (1-based): growing, jittered (half to 1.5×), at most 20 s. */
export function retryDelayMs(attempt: number, random: () => number = Math.random): number {
  return Math.min(20_000, 1000 * 2 ** (attempt - 1)) * (0.5 + random());
}

export interface FetchProgress {
  file: string;
  loadedBytes: number;
  totalBytes: number;
}

export interface StreamOptions {
  /** Bytes already saved; asks the server for the rest (Range request). */
  offset?: number;
  /** Writes a chunk at an absolute file position. May be async (e.g. to checkpoint). */
  write: (chunk: Uint8Array, position: number) => void | Promise<void>;
  /** The server ignored the Range request and is sending the whole file: discard saved bytes. */
  onRestart?: () => void | Promise<void>;
  onProgress?: (p: FetchProgress) => void;
  signal?: AbortSignal;
}

/**
 * Streams `file` from `url` into `options.write`. Resolves once exactly
 * `file.sizeBytes` bytes have been written; checks size only (callers verify
 * the SHA-256). Returns the number of bytes transferred by this call.
 */
export async function streamArtifact(url: string, file: ArtifactFile, options: StreamOptions): Promise<number> {
  const { signal } = options;
  const offset = options.offset ?? 0;
  if (offset >= file.sizeBytes) return 0;

  // Our own controller, so a stalled transfer can be stopped as well as a cancelled one.
  const transfer = new AbortController();
  const onAbort = () => transfer.abort();
  if (signal?.aborted) transfer.abort();
  signal?.addEventListener('abort', onAbort);
  let stalled = false;
  let stallTimer: ReturnType<typeof setTimeout> | undefined;
  const watch = () => {
    clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      stalled = true;
      transfer.abort();
    }, STALL_MS);
  };
  watch();
  try {
    return await transferTo(url, file, options, offset, transfer.signal, watch, () => stalled);
  } finally {
    clearTimeout(stallTimer);
    signal?.removeEventListener('abort', onAbort);
  }
}

async function transferTo(
  url: string,
  file: ArtifactFile,
  options: StreamOptions,
  start: number,
  signal: AbortSignal,
  progressed: () => void,
  stalled: () => boolean,
): Promise<number> {
  const { write, onProgress } = options;
  let offset = start;
  const interrupted = (err: unknown) =>
    stalled()
      ? new ArtifactError('download-failed', `The download of ${file.path} stalled (no data for ${STALL_MS / 1000} s).`, true)
      : options.signal?.aborted
        ? new ArtifactError('cancelled', 'Download cancelled.')
        : new ArtifactError('download-failed', `Download of ${file.path} was interrupted: ${describe(err)}`, true);

  // A file the browser downloaded in the background (public/background-download-sw.js) is read from there.
  const background = await backgroundDownload(url);
  let response: Response;
  try {
    response = background ?? (await fetch(url, { signal, headers: offset > 0 ? { Range: `bytes=${offset}-` } : undefined }));
  } catch (err) {
    throw interrupted(err);
  }
  if (!response.ok || !response.body) {
    const permanent = PERMANENT[response.status];
    throw new ArtifactError(
      'download-failed',
      permanent ? `Could not download ${file.path}: ${permanent}.` : `Could not download ${file.path}: HTTP ${response.status}`,
      !permanent && isTransientStatus(response.status),
    );
  }
  if (offset > 0 && !(response.status === 206 && rangeStart(response) === offset)) {
    // No usable range support: start again from the beginning.
    offset = 0;
    await options.onRestart?.();
  }

  let position = offset;
  const reader = response.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (position + value.byteLength > file.sizeBytes) {
        throw new ArtifactError('integrity-mismatch', `${file.path} is larger than the expected ${file.sizeBytes} bytes`);
      }
      await write(value, position);
      position += value.byteLength;
      progressed();
      onProgress?.({ file: file.path, loadedBytes: position, totalBytes: file.sizeBytes });
    }
  } catch (err) {
    if (err instanceof ArtifactError) throw err;
    // Errors from `write` (e.g. disk full) are the caller's to classify.
    if (!(err instanceof TypeError) && !isAbortError(err)) throw err;
    throw interrupted(err);
  } finally {
    // Stops the network transfer if we bailed out early; a no-op once the body is fully read.
    reader.cancel().catch(() => {});
  }
  if (position !== file.sizeBytes) {
    throw new ArtifactError('integrity-mismatch', `${file.path}: expected ${file.sizeBytes} bytes, received ${position}`);
  }
  // Copied to the model store: the background copy is no longer needed (and takes as much space again).
  if (background) await forgetBackgroundDownload(url);
  return background ? 0 : position - offset;
}

/** Where public/background-download-sw.js keeps files the browser downloaded in the background. */
export const BACKGROUND_CACHE = 'quick-inf-background-downloads';

/** Whether the browser has already downloaded `url` in the background (see BACKGROUND_CACHE). */
export async function hasBackgroundDownload(url: string): Promise<boolean> {
  return (await backgroundDownload(url)) !== null;
}

async function backgroundDownload(url: string): Promise<Response | null> {
  try {
    if (typeof caches === 'undefined') return null;
    const cache = await caches.open(BACKGROUND_CACHE);
    const response = await cache.match(url);
    return response?.ok && response.body ? response : null;
  } catch {
    return null;
  }
}

async function forgetBackgroundDownload(url: string): Promise<void> {
  try {
    await (await caches.open(BACKGROUND_CACHE)).delete(url);
  } catch {
    // Left for the browser to evict; harmless.
  }
}

/** Downloads a whole file into memory (no resume). */
export async function fetchArtifact(
  url: string,
  file: ArtifactFile,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  // One buffer of the expected size, instead of concatenating chunks (which briefly needs twice the memory).
  const bytes = new Uint8Array(file.sizeBytes);
  await streamArtifact(url, file, { write: (chunk, at) => bytes.set(chunk, at), onProgress, signal });
  return bytes;
}

export async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

export async function verifySha256(bytes: Uint8Array<ArrayBuffer>, file: ArtifactFile): Promise<void> {
  const hex = await sha256Hex(bytes);
  if (hex !== file.sha256) {
    throw new ArtifactError('integrity-mismatch', `${file.path} failed its integrity check (sha256 ${hex})`);
  }
}

/** Start offset from a `Content-Range: bytes start-end/total` header, or null. */
function rangeStart(response: Response): number | null {
  const match = /^bytes (\d+)-\d+\/(\d+|\*)$/.exec(response.headers.get('content-range') ?? '');
  return match ? Number(match[1]) : null;
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
