// The on-device model store: artifact files live in OPFS, their status in
// IndexedDB (`artifacts` store). Runs in the inference worker, which is the
// only place OPFS sync access handles (fast, exclusive file writes) exist.
//
// Rules:
// - A file counts as saved only when its record says `complete` AND the file
//   on disk has the expected size. `complete` is set only after the SHA-256
//   matched, so nothing unverified is ever loaded from disk.
// - Downloads write a record before creating the file, and checkpoint
//   (flush to disk, then record `bytesWritten`) every few MB. An interrupted
//   download resumes from its last checkpoint with an HTTP Range request.
// - OPFS and IndexedDB aren't one transaction, so `reconcile` repairs the
//   leftovers of a crash (missing files, damaged files, files with no record).
// - Files the catalogue no longer lists (a model moved to a new revision or
//   file, or was removed) are deleted at start-up by `removeUnlisted`; nothing
//   else would ever reach them.
// - One Web Lock per model revision keeps tabs from writing or deleting the
//   same files at once. A tab that wants files another tab is downloading
//   waits for it, then uses the result.

import type { ArtifactFile, ModelSpec } from '../catalogue/models';
import { RuntimeError } from '../runtime/contract';
import { dbDelete, dbGet, dbGetAll, dbPut } from './db';
import type { ByteSource } from '../runtime/onnx/external-weights';
import { ArtifactError, sha256Hex, streamArtifact } from './fetch-artifact';
import { Sha256 } from './sha256';

export interface ArtifactRecord {
  /** OPFS path below `models/`, e.g. `HuggingFaceTB__SmolLM2-360M-Instruct/<revision>/onnx/model_q4f16.onnx`. */
  key: string;
  repo: string;
  revision: string;
  path: string;
  sizeBytes: number;
  sha256: string;
  status: 'partial' | 'complete';
  /** Durable checkpoint: bytes known to be flushed to disk. */
  bytesWritten: number;
  updatedAt: number;
}

export interface FileState {
  file: ArtifactFile;
  complete: boolean;
  /** Bytes already on disk that a download can resume from (sizeBytes when complete). */
  savedBytes: number;
}

type Model = Pick<ModelSpec, 'source'>;

/** OPFS's worker-only synchronous file handle (not in the DOM type library). */
interface SyncAccessHandle {
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  truncate(size: number): void;
  getSize(): number;
  flush(): void;
  close(): void;
}
type WritableFileHandle = FileSystemFileHandle & { createSyncAccessHandle(): Promise<SyncAccessHandle> };

const ROOT_DIR = 'models';
/** Flush and record progress this often, bounding the work lost to a crash. */
const CHECKPOINT_BYTES = 16 * 1024 * 1024;
/** Free space to leave beyond the download itself. */
const MIN_HEADROOM_BYTES = 64 * 1024 * 1024;

export class ModelStore {
  private ready: Promise<boolean> | null = null;

  /**
   * @param listed The files to keep: record key (see artifactKey) → expected
   *   SHA-256. Anything else saved is deleted at start-up. Omit to keep everything.
   */
  constructor(private readonly listed?: ReadonlyMap<string, string>) {}

  /** Checks the browser supports the store, then repairs leftovers from earlier crashes. False if unusable. */
  init(): Promise<boolean> {
    this.ready ??= (async () => {
      if (!supported()) return false;
      try {
        await this.dir([], true);
        await dbGetAll('artifacts');
      } catch {
        return false; // e.g. storage blocked in a private window
      }
      await this.reconcile().catch(() => {});
      await this.removeUnlisted().catch(() => {});
      return true;
    })();
    return this.ready;
  }

  async inspect(model: Model, files: ArtifactFile[]): Promise<FileState[]> {
    return Promise.all(
      files.map(async (file) => {
        const { complete, savedBytes } = await this.state(model, file);
        return { file, complete, savedBytes };
      }),
    );
  }

  /**
   * Runs `fn` holding this model's lock. If another tab holds it, calls
   * `onWaiting` and waits (abortable via `signal`).
   */
  async withLock<T>(
    model: Model,
    fn: () => Promise<T>,
    options: { signal?: AbortSignal; onWaiting?: () => void } = {},
  ): Promise<T> {
    const name = lockName(modelDir(model));
    let result: { value: T } | null = null;
    await navigator.locks.request(name, { ifAvailable: true }, async (lock) => {
      if (lock) result = { value: await fn() };
    });
    if (result) return (result as { value: T }).value;
    options.onWaiting?.();
    try {
      return await navigator.locks.request(name, { signal: options.signal }, fn);
    } catch (err) {
      if (options.signal?.aborted) throw new RuntimeError('cancelled', 'Loading was cancelled.');
      throw err;
    }
  }

  /** Runs `fn` holding this model's lock if no other tab holds it; null if one does. */
  async ifUnlocked<T>(model: Model, fn: () => Promise<T>): Promise<T | null> {
    let result: { value: T } | null = null;
    await navigator.locks.request(lockName(modelDir(model)), { ifAvailable: true }, async (lock) => {
      if (lock) result = { value: await fn() };
    });
    return result ? (result as { value: T }).value : null;
  }

  /** Free space check before downloading `bytes`. */
  async ensureSpace(bytes: number): Promise<void> {
    // Development: test browsers can be given a larger quota than navigator.storage.estimate() reports.
    if (import.meta.env.DEV && (globalThis as { __quickInfSkipSpaceCheck?: boolean }).__quickInfSkipSpaceCheck) return;
    const { quota, usage } = await navigator.storage.estimate();
    if (quota === undefined || usage === undefined) return;
    const free = quota - usage;
    const needed = bytes + Math.max(MIN_HEADROOM_BYTES, bytes * 0.05);
    if (free < needed) {
      throw new RuntimeError(
        'storage-full',
        `The download needs ${mb(needed)} MB of browser storage but only ${mb(free)} MB is available.`,
        true,
      );
    }
  }

  /**
   * Downloads `file` to disk, resuming from the last checkpoint. Call inside
   * `withLock`. Resolves when every byte is on disk (not yet verified).
   * Returns the bytes transferred over the network.
   */
  async download(
    model: Model,
    file: ArtifactFile,
    url: string,
    options: { onProgress?: (savedBytes: number) => void; signal?: AbortSignal } = {},
  ): Promise<number> {
    const current = await this.state(model, file);
    if (current.complete) return 0;
    let record = current.record;
    if (!record) {
      record = newRecord(model, file);
      await dbPut('artifacts', record); // record first: a file on disk always has one
    }

    const handle = await this.fileHandle(model, file, true);
    let access: SyncAccessHandle;
    try {
      access = await (handle as WritableFileHandle).createSyncAccessHandle();
    } catch (err) {
      throw storageError(err, `Could not open ${file.path} for writing`);
    }

    let position = Math.min(current.savedBytes, access.getSize());
    const checkpoint = async (bytes: number) => {
      access.flush();
      record = { ...record!, bytesWritten: bytes, updatedAt: Date.now() };
      await dbPut('artifacts', record);
    };
    try {
      access.truncate(position); // drop anything written after the last checkpoint
      let sinceCheckpoint = 0;
      const transferred = await streamArtifact(url, file, {
        offset: position,
        write: async (chunk, at) => {
          access.write(chunk, { at });
          position = at + chunk.byteLength;
          sinceCheckpoint += chunk.byteLength;
          if (sinceCheckpoint >= CHECKPOINT_BYTES) {
            sinceCheckpoint = 0;
            await checkpoint(position);
          }
        },
        onRestart: async () => {
          access.truncate(0);
          position = 0;
          await checkpoint(0);
        },
        onProgress: (p) => options.onProgress?.(p.loadedBytes),
        signal: options.signal,
      });
      await checkpoint(file.sizeBytes);
      return transferred;
    } catch (err) {
      if (err instanceof ArtifactError && err.code === 'integrity-mismatch') {
        // The server sent something else (wrong size): start over next time.
        access.truncate(0);
        await checkpoint(0).catch(() => {});
        throw err;
      }
      // Keep what arrived so the next attempt resumes from here.
      await checkpoint(position).catch(() => {});
      if (err instanceof ArtifactError || err instanceof RuntimeError) throw err;
      throw storageError(err, `Could not save ${file.path}`);
    } finally {
      access.close();
    }
  }

  /** Reads a fully downloaded file, checks its SHA-256 and marks it complete. Call inside `withLock`. */
  async verifyAndCommit(model: Model, file: ArtifactFile, { keepBytes = true } = {}): Promise<Uint8Array<ArrayBuffer>> {
    let bytes = new Uint8Array(0);
    let hex: string;
    if (keepBytes) {
      bytes = await this.readBytes(model, file);
      hex = bytes.byteLength === file.sizeBytes ? await sha256Hex(bytes) : 'wrong size';
    } else {
      hex = await this.hashOnDisk(model, file);
    }
    if (hex !== file.sha256) {
      await this.discard(model, file);
      throw new ArtifactError('integrity-mismatch', `${file.path} failed its integrity check (sha256 ${hex})`);
    }
    const record = newRecord(model, file);
    await dbPut('artifacts', { ...record, status: 'complete', bytesWritten: file.sizeBytes });
    return bytes;
  }

  /** Reads a saved (complete) file. */
  async read(model: Model, file: ArtifactFile): Promise<Uint8Array<ArrayBuffer>> {
    const bytes = await this.readBytes(model, file);
    if (bytes.byteLength !== file.sizeBytes) {
      await this.discard(model, file);
      throw new RuntimeError('integrity-mismatch', `The saved copy of ${file.path} is damaged. Load again to download it.`, true);
    }
    return bytes;
  }

  /**
   * Opens a saved (complete) file for reading slices on demand, without
   * reading it whole. Call inside `withLock`, and close it when done.
   */
  async openReader(model: Model, file: ArtifactFile): Promise<ByteSource & { close(): void }> {
    let access: SyncAccessHandle;
    try {
      access = await ((await this.fileHandle(model, file, false)) as WritableFileHandle).createSyncAccessHandle();
    } catch (err) {
      throw storageError(err, `Could not open the saved copy of ${file.path}`);
    }
    if (access.getSize() !== file.sizeBytes) {
      access.close();
      await this.discard(model, file);
      throw new RuntimeError('integrity-mismatch', `The saved copy of ${file.path} is damaged. Load again to download it.`, true);
    }
    return {
      size: file.sizeBytes,
      read(offset, length) {
        const buffer = new Uint8Array(length);
        const got = access.read(buffer, { at: offset });
        if (got !== length) throw new Error(`Short read from ${file.path}: ${got} of ${length} bytes at ${offset}`);
        return buffer;
      },
      readInto(offset, target) {
        const got = access.read(target, { at: offset });
        if (got !== target.byteLength) throw new Error(`Short read from ${file.path}: ${got} of ${target.byteLength} bytes at ${offset}`);
      },
      close: () => access.close(),
    };
  }

  /** Deletes every saved file of this model revision. Fails if another tab is using them. */
  async deleteModel(model: Model): Promise<void> {
    const dir = modelDir(model);
    const deleted = await navigator.locks.request(lockName(dir), { ifAvailable: true }, async (lock) => {
      if (!lock) return false;
      // Records first: if this is interrupted, leftover files have no record and reconcile removes them.
      const records = await dbGetAll<ArtifactRecord>('artifacts');
      for (const r of records) if (r.key.startsWith(`${dir}/`)) await dbDelete('artifacts', r.key);
      const repoDir = await this.dir([dir.split('/')[0]!], false);
      await repoDir?.removeEntry(model.source.revision, { recursive: true }).catch(ignoreNotFound);
      return true;
    });
    if (!deleted) {
      throw new RuntimeError('in-use-elsewhere', 'Another Quick Inference tab is downloading or loading this model.', true);
    }
  }

  /** All records, for storage reporting. */
  records(): Promise<ArtifactRecord[]> {
    return dbGetAll<ArtifactRecord>('artifacts');
  }

  /** Exact bytes of every file under `models/` (saved, partial and leftover files alike). */
  async diskBytes(): Promise<number> {
    let total = 0;
    for (const size of (await this.listFiles()).values()) total += size;
    return total;
  }

  /** Repairs leftovers of crashes and interrupted deletes. Skips models another tab is using. */
  private async reconcile(): Promise<void> {
    const records = await dbGetAll<ArtifactRecord>('artifacts');
    const byKey = new Map(records.map((r) => [r.key, r]));
    const files = await this.listFiles();
    const dirs = new Set([...records.map((r) => dirOfKey(r.key)), ...files.keys()].map(dirOfKey));

    for (const dir of dirs) {
      await navigator.locks.request(lockName(dir), { ifAvailable: true }, async (lock) => {
        if (!lock) return;
        for (const record of records.filter((r) => dirOfKey(r.key) === dir)) {
          const size = files.get(record.key);
          if (size === undefined) {
            await dbDelete('artifacts', record.key); // file gone
          } else if (record.status === 'complete' && size !== record.sizeBytes) {
            await dbDelete('artifacts', record.key); // damaged; the file is removed below
            byKey.delete(record.key);
          } else if (record.status === 'partial' && size < record.bytesWritten) {
            await dbPut('artifacts', { ...record, bytesWritten: 0, updatedAt: Date.now() }); // checkpoint is wrong
          }
        }
        for (const key of files.keys()) {
          if (dirOfKey(key) === dir && !byKey.has(key)) await this.removeFile(key); // no record: abandoned
        }
      });
    }
  }

  /**
   * Deletes saved files the catalogue doesn't list (or lists with another hash):
   * whole old model revisions and single replaced files. Skips models another
   * tab is using; they're cleaned up on a later start.
   */
  private async removeUnlisted(): Promise<void> {
    const listed = this.listed;
    if (!listed) return;
    const records = await dbGetAll<ArtifactRecord>('artifacts');
    const files = await this.listFiles();
    const dirs = new Set([...records.map((r) => r.key), ...files.keys()].map(dirOfKey));
    const listedDirs = new Set([...listed.keys()].map(dirOfKey));

    for (const dir of dirs) {
      await navigator.locks.request(lockName(dir), { ifAvailable: true }, async (lock) => {
        if (!lock) return;
        if (!listedDirs.has(dir)) {
          // A whole revision no longer used: records first (see deleteModel), then the folder.
          for (const r of records) if (dirOfKey(r.key) === dir) await dbDelete('artifacts', r.key);
          const [repo, revision] = dir.split('/') as [string, string];
          const repoDir = await this.dir([repo], false);
          await repoDir?.removeEntry(revision, { recursive: true }).catch(ignoreNotFound);
          // Remove the repo folder too if that was its last revision.
          let empty = true;
          if (repoDir) for await (const _ of repoDir.keys()) empty = false;
          if (repoDir && empty) await (await this.dir([], false))?.removeEntry(repo).catch(ignoreNotFound);
          return;
        }
        for (const r of records) {
          if (dirOfKey(r.key) === dir && listed.get(r.key) !== r.sha256) {
            await dbDelete('artifacts', r.key);
            await this.removeFile(r.key);
          }
        }
      });
    }
  }

  private async state(
    model: Model,
    file: ArtifactFile,
  ): Promise<{ record: ArtifactRecord | undefined; complete: boolean; savedBytes: number }> {
    const key = artifactKey(model, file);
    let record = await dbGet<ArtifactRecord>('artifacts', key);
    if (record && (record.sha256 !== file.sha256 || record.sizeBytes !== file.sizeBytes)) {
      // The catalogue changed this file's hash: treat the saved copy as stale.
      await this.discard(model, file);
      record = undefined;
    }
    if (!record) return { record, complete: false, savedBytes: 0 };
    const size = (await this.fileSize(key)) ?? 0;
    if (record.status === 'complete') {
      if (size === file.sizeBytes) return { record, complete: true, savedBytes: size };
      record = { ...record, status: 'partial', bytesWritten: 0 }; // damaged: download again
      await dbPut('artifacts', record);
      return { record, complete: false, savedBytes: 0 };
    }
    return { record, complete: false, savedBytes: Math.min(record.bytesWritten, size) };
  }

  /** SHA-256 of a saved file, read in slices (never whole), or 'wrong size'. */
  private async hashOnDisk(model: Model, file: ArtifactFile): Promise<string> {
    let blob: Blob;
    try {
      blob = await (await this.fileHandle(model, file, false)).getFile();
    } catch (err) {
      throw storageError(err, `Could not read the saved copy of ${file.path}`);
    }
    if (blob.size !== file.sizeBytes) return 'wrong size';
    const sha = new Sha256();
    const step = 8 * 1024 * 1024;
    for (let at = 0; at < blob.size; at += step) sha.update(new Uint8Array(await blob.slice(at, at + step).arrayBuffer()));
    return sha.hex();
  }

  private async readBytes(model: Model, file: ArtifactFile): Promise<Uint8Array<ArrayBuffer>> {
    try {
      const handle = await this.fileHandle(model, file, false);
      return new Uint8Array(await (await handle.getFile()).arrayBuffer());
    } catch (err) {
      throw storageError(err, `Could not read the saved copy of ${file.path}`);
    }
  }

  private async discard(model: Model, file: ArtifactFile): Promise<void> {
    const key = artifactKey(model, file);
    await dbDelete('artifacts', key);
    await this.removeFile(key);
  }

  private async fileHandle(model: Model, file: ArtifactFile, create: boolean): Promise<FileSystemFileHandle> {
    const parts = artifactKey(model, file).split('/');
    const name = parts.pop()!;
    const dir = await this.dir(parts, create);
    if (!dir) throw new DOMException(`${file.path} is not saved`, 'NotFoundError');
    return dir.getFileHandle(name, { create });
  }

  /** The directory at `models/<parts…>`, or null if it doesn't exist and `create` is false. */
  private async dir(parts: string[], create: boolean): Promise<FileSystemDirectoryHandle | null> {
    let dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT_DIR, { create: true });
    for (const part of parts) {
      try {
        dir = await dir.getDirectoryHandle(part, { create });
      } catch (err) {
        if (!create && isNotFound(err)) return null;
        throw err;
      }
    }
    return dir;
  }

  private async fileSize(key: string): Promise<number | null> {
    const parts = key.split('/');
    const name = parts.pop()!;
    try {
      const dir = await this.dir(parts, false);
      if (!dir) return null;
      return (await (await dir.getFileHandle(name)).getFile()).size;
    } catch (err) {
      if (isNotFound(err)) return null;
      throw err;
    }
  }

  private async removeFile(key: string): Promise<void> {
    const parts = key.split('/');
    const name = parts.pop()!;
    const dir = await this.dir(parts, false);
    await dir?.removeEntry(name).catch(ignoreNotFound);
  }

  /** Every file under `models/`, keyed like records, with its size. */
  private async listFiles(): Promise<Map<string, number>> {
    const out = new Map<string, number>();
    const walk = async (dir: FileSystemDirectoryHandle, prefix: string) => {
      for await (const [name, entry] of dir.entries()) {
        const path = prefix ? `${prefix}/${name}` : name;
        if (entry.kind === 'directory') await walk(entry as FileSystemDirectoryHandle, path);
        else out.set(path, (await (entry as FileSystemFileHandle).getFile()).size);
      }
    };
    const root = await this.dir([], true);
    if (root) await walk(root, '');
    return out;
  }
}

function supported(): boolean {
  return (
    typeof indexedDB !== 'undefined' &&
    typeof navigator.storage?.getDirectory === 'function' &&
    typeof navigator.locks?.request === 'function' &&
    typeof FileSystemFileHandle !== 'undefined' &&
    'createSyncAccessHandle' in FileSystemFileHandle.prototype
  );
}

/** `<repo with / as __>/<revision>`: one directory (and one lock) per model revision. */
function modelDir(model: Model): string {
  return `${model.source.repo.replace(/\//g, '__')}/${model.source.revision}`;
}

export function artifactKey(model: Model, file: ArtifactFile): string {
  return `${modelDir(model)}/${file.path}`;
}

/** The model directory part (first two segments) of a record key. */
function dirOfKey(key: string): string {
  return key.split('/').slice(0, 2).join('/');
}

function lockName(dir: string): string {
  return `quick-inf:model:${dir}`;
}

function newRecord(model: Model, file: ArtifactFile): ArtifactRecord {
  return {
    key: artifactKey(model, file),
    repo: model.source.repo,
    revision: model.source.revision,
    path: file.path,
    sizeBytes: file.sizeBytes,
    sha256: file.sha256,
    status: 'partial',
    bytesWritten: 0,
    updatedAt: Date.now(),
  };
}

function storageError(err: unknown, context: string): RuntimeError {
  const name = err instanceof DOMException ? err.name : '';
  const detail = err instanceof Error ? err.message : String(err);
  if (name === 'QuotaExceededError') return new RuntimeError('storage-full', `${context}: browser storage is full.`, true);
  if (name === 'NoModificationAllowedError') {
    return new RuntimeError('in-use-elsewhere', `${context}: the file is in use in another tab.`, true);
  }
  return new RuntimeError('storage-unavailable', `${context}: ${detail}`, true);
}

function isNotFound(err: unknown): boolean {
  return err instanceof DOMException && (err.name === 'NotFoundError' || err.name === 'TypeMismatchError');
}

function ignoreNotFound(err: unknown): void {
  if (!isNotFound(err)) throw err;
}

function mb(bytes: number): number {
  return Math.ceil(bytes / 1e6);
}
