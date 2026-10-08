// Temporary artifact loader for milestone 1: downloads into memory and verifies
// size and SHA-256. Milestone 3 replaces this with the OPFS model store
// (streamed writes, checkpoints, reuse), keeping the same verify-before-ready rule.

import type { ArtifactFile } from '../catalogue/models';

export class ArtifactError extends Error {
  constructor(
    readonly code: 'download-failed' | 'integrity-mismatch',
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

export async function fetchArtifact(
  url: string,
  file: ArtifactFile,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  let response: Response;
  try {
    response = await fetch(url, { signal });
  } catch (err) {
    throw new ArtifactError('download-failed', `Could not download ${file.path}: ${describe(err)}`);
  }
  if (!response.ok || !response.body) {
    throw new ArtifactError('download-failed', `Could not download ${file.path}: HTTP ${response.status}`);
  }

  // Write straight into one buffer of the expected size instead of collecting
  // chunks and concatenating (which would briefly need twice the memory).
  const bytes = new Uint8Array(file.sizeBytes);
  let loaded = 0;
  const reader = response.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > bytes.byteLength) {
        throw new ArtifactError('integrity-mismatch', `${file.path} is larger than the expected ${file.sizeBytes} bytes`);
      }
      bytes.set(value, loaded);
      loaded += value.byteLength;
      onProgress({ file: file.path, loadedBytes: loaded, totalBytes: file.sizeBytes });
    }
  } catch (err) {
    if (err instanceof ArtifactError) throw err;
    throw new ArtifactError('download-failed', `Download of ${file.path} was interrupted: ${describe(err)}`);
  }
  if (loaded !== file.sizeBytes) {
    throw new ArtifactError('integrity-mismatch', `${file.path}: expected ${file.sizeBytes} bytes, received ${loaded}`);
  }
  return bytes;
}

export async function verifySha256(bytes: Uint8Array<ArrayBuffer>, file: ArtifactFile): Promise<void> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
  if (hex !== file.sha256) {
    throw new ArtifactError('integrity-mismatch', `${file.path} failed its integrity check (sha256 ${hex})`);
  }
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
