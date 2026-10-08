// Artifact download primitives. `streamArtifact` streams one file to a caller-
// supplied writer, optionally resuming from a byte offset with an HTTP Range
// request. The OPFS model store writes to disk with it; `fetchArtifact` keeps
// a file in memory (tests, diagnostics page).

import type { ArtifactFile } from '../catalogue/models';

export class ArtifactError extends Error {
  constructor(
    readonly code: 'download-failed' | 'integrity-mismatch' | 'cancelled',
    message: string,
  ) {
    super(message);
  }
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
  const { write, onProgress, signal } = options;
  let offset = options.offset ?? 0;
  if (offset >= file.sizeBytes) return 0;

  let response: Response;
  try {
    response = await fetch(url, { signal, headers: offset > 0 ? { Range: `bytes=${offset}-` } : undefined });
  } catch (err) {
    if (signal?.aborted) throw new ArtifactError('cancelled', 'Download cancelled.');
    throw new ArtifactError('download-failed', `Could not download ${file.path}: ${describe(err)}`);
  }
  if (!response.ok || !response.body) {
    throw new ArtifactError('download-failed', `Could not download ${file.path}: HTTP ${response.status}`);
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
      onProgress?.({ file: file.path, loadedBytes: position, totalBytes: file.sizeBytes });
    }
  } catch (err) {
    if (err instanceof ArtifactError) throw err;
    if (signal?.aborted) throw new ArtifactError('cancelled', 'Download cancelled.');
    // Errors from `write` (e.g. disk full) are the caller's to classify.
    if (!(err instanceof TypeError) && !isAbortError(err)) throw err;
    throw new ArtifactError('download-failed', `Download of ${file.path} was interrupted: ${describe(err)}`);
  } finally {
    // Stops the network transfer if we bailed out early; a no-op once the body is fully read.
    reader.cancel().catch(() => {});
  }
  if (position !== file.sizeBytes) {
    throw new ArtifactError('integrity-mismatch', `${file.path}: expected ${file.sizeBytes} bytes, received ${position}`);
  }
  return position - offset;
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
