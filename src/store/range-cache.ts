// A device's share of a model split across devices (info/distributed-plan.md): byte ranges of the model's files,
// fetched from Hugging Face (the dev server in development) and kept in OPFS under ranges/<repo>/<revision>/<sha256>.
// Each is checked against the SHA-256 the host sent (from its own verified copy) before it's kept, so, as with whole
// files in model-store.ts, nothing unverified is ever loaded. A range kept from an earlier session is read from disk
// (a kept file of the wrong size is fetched again). Runs in the inference worker (OPFS sync access handles).

import { artifactUrl, type ModelSpec } from '../catalogue/models';
import type { ByteSource } from '../runtime/onnx/external-weights';
import { retryDelayMs } from './fetch-artifact';

const ROOT_DIR = 'ranges';
const ATTEMPTS = 4;

type Model = Pick<ModelSpec, 'source'>;

interface SyncAccessHandle {
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  getSize(): number;
  flush(): void;
  close(): void;
}
type WritableFileHandle = FileSystemFileHandle & { createSyncAccessHandle(): Promise<SyncAccessHandle> };

export interface RangeRequest {
  /** The file's path in the model's repository. */
  path: string;
  offset: number;
  length: number;
  sha256: string;
}

/** The range, from disk (fetched and checked first if it isn't kept yet); close it once read. */
export async function keptRange(model: Model, range: RangeRequest, onBytes: (n: number) => void): Promise<ByteSource & { close(): void }> {
  const dir = await rangeDir(model, true);
  const name = range.sha256;
  const existing = await dir.getFileHandle(name).catch(() => null);
  if (existing) {
    const reader = await open(existing as WritableFileHandle, range);
    if (reader) {
      onBytes(range.length);
      return reader;
    }
    await dir.removeEntry(name).catch(() => {});
  }
  const bytes = await fetchChecked(model, range, onBytes);
  const handle = (await dir.getFileHandle(name, { create: true })) as WritableFileHandle;
  const access = await handle.createSyncAccessHandle();
  try {
    access.write(bytes, { at: 0 });
    access.flush();
  } finally {
    access.close();
  }
  const reader = await open(handle, range);
  if (!reader) throw new Error(`Could not keep part of ${range.path}`);
  return reader;
}

/** Bytes of the ranges kept for this model revision. */
export async function keptRangeBytes(model: Model): Promise<number> {
  const [repo, revision] = parts(model);
  try {
    const root = await (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT_DIR);
    const dir = await (await root.getDirectoryHandle(repo)).getDirectoryHandle(revision);
    let bytes = 0;
    for await (const handle of (dir as unknown as { values(): AsyncIterable<FileSystemHandle> }).values()) {
      if (handle.kind === 'file') bytes += (await (handle as FileSystemFileHandle).getFile()).size;
    }
    return bytes;
  } catch {
    return 0; // nothing kept
  }
}

/** Deletes the ranges kept for this model revision other than `keep` (by hash): only the latest share is kept. */
export async function pruneRanges(model: Model, keep: ReadonlySet<string>): Promise<void> {
  const dir = await rangeDir(model, false).catch(() => null);
  if (!dir) return;
  const names: string[] = [];
  for await (const handle of (dir as unknown as { values(): AsyncIterable<FileSystemHandle> }).values()) {
    if (handle.kind === 'file' && !keep.has(handle.name)) names.push(handle.name);
  }
  for (const name of names) await dir.removeEntry(name).catch(() => {});
}

/** Deletes the ranges kept for this model revision. */
export async function deleteRanges(model: Model): Promise<void> {
  const [repo, revision] = parts(model);
  const root = await (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT_DIR).catch(() => null);
  const repoDir = await root?.getDirectoryHandle(repo).catch(() => null);
  await repoDir?.removeEntry(revision, { recursive: true }).catch(() => {});
}

async function open(handle: WritableFileHandle, range: RangeRequest): Promise<(ByteSource & { close(): void }) | null> {
  const access = await handle.createSyncAccessHandle();
  if (access.getSize() !== range.length) {
    access.close();
    return null;
  }
  return {
    size: range.length,
    read(offset, length) {
      const buffer = new Uint8Array(length);
      if (access.read(buffer, { at: offset }) !== length) throw new Error(`Short read from a kept part of ${range.path}`);
      return buffer;
    },
    readInto(offset, target) {
      if (access.read(target, { at: offset }) !== target.byteLength) throw new Error(`Short read from a kept part of ${range.path}`);
    },
    close: () => access.close(),
  };
}

/** The range from the model's host (Hugging Face), checked against its hash; retried with backoff. */
async function fetchChecked(model: Model, range: RangeRequest, onBytes: (n: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  const url = artifactUrl(model, { path: range.path, sizeBytes: 0, sha256: range.sha256 });
  for (let attempt = 1; ; attempt++) {
    let counted = 0;
    try {
      const response = await fetch(url, { headers: { Range: `bytes=${range.offset}-${range.offset + range.length - 1}` } });
      if (response.status !== 206 || !response.body) throw new Error(`HTTP ${response.status}`);
      const bytes = new Uint8Array(range.length);
      let at = 0;
      for (const reader = response.body.getReader(); ; ) {
        const { done, value } = await reader.read();
        if (done) break;
        if (at + value.byteLength > range.length) throw new Error('more bytes than asked for');
        bytes.set(value, at);
        at += value.byteLength;
        counted += value.byteLength;
        onBytes(value.byteLength);
      }
      if (at !== range.length) throw new Error(`${at} of ${range.length} bytes`);
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
      if (hash !== range.sha256) throw new Error('its contents differ from the host’s copy');
      return bytes;
    } catch (err) {
      onBytes(-counted);
      if (attempt >= ATTEMPTS) throw new Error(`Could not download part of ${range.path}: ${err instanceof Error ? err.message : String(err)}`);
      await new Promise((r) => setTimeout(r, retryDelayMs(attempt)));
    }
  }
}

function parts(model: Model): [string, string] {
  return [model.source.repo.replace('/', '__'), model.source.revision];
}

async function rangeDir(model: Model, create: boolean): Promise<FileSystemDirectoryHandle> {
  let dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT_DIR, { create });
  for (const part of parts(model)) dir = await dir.getDirectoryHandle(part, { create });
  return dir;
}
