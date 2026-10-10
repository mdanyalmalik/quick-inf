// What WebLLM has saved on this device, for the storage panel, deletion,
// start-up cleanup and the post-download weight check. Runs in the worker.
//
// WebLLM downloads and saves its own files (we load it with its OPFS backend).
// Its layout, as of @mlc-ai/web-llm 0.2.85: `tvmjs-opfs-store/<scope>/` holds,
// per downloaded URL, `<sha256(url) hex>.bin` (the bytes) and
// `<sha256(url) hex>.record.json` ({ url, nbytes }), the record written only
// after the bytes are complete. Scopes are webllm/model, webllm/config and
// webllm/wasm. This module reads that layout itself rather than importing
// WebLLM (6.6 MB), and finds files by URL hash, whatever the scope.
//
// Recheck the layout when upgrading WebLLM.

import type { WebLlmArtifact } from '../catalogue/models';
import { WEBLLM_FILES } from '../catalogue/webllm-files';
import { dbDelete, dbGet, dbPut } from './db';
import { Md5 } from './md5';

const ROOT_DIR = 'tvmjs-opfs-store';

/** One saved file (complete when its record exists and the sizes agree). */
export interface WebLlmSavedFile {
  /** Hex SHA-256 of the URL, WebLLM's file name. */
  hash: string;
  /** From the record; absent while the download is unfinished. */
  url?: string;
  recordBytes?: number;
  /** Bytes on disk. */
  payloadBytes: number;
  dir: FileSystemDirectoryHandle;
}

export interface WebLlmArtifactState {
  /** Bytes on disk, including unfinished files. */
  savedBytes: number;
  /** Bytes in finished files: what a resumed download keeps (unfinished files start over). */
  resumableBytes: number;
  complete: boolean;
  /** Some files are saved, not all. */
  partial: boolean;
}

/** Every URL WebLLM fetches to load `artifact`, with the size it should have. */
export function webllmUrls(artifact: WebLlmArtifact): { url: string; sizeBytes: number; md5?: string }[] {
  const { build } = artifact;
  const files = WEBLLM_FILES[artifact.modelId] ?? [];
  // WebLLM's own form of the base URL (cleanModelUrl), so the hashes match its file names.
  const base = new URL(`https://huggingface.co/${build.repo}/resolve/${build.revision}/`).href;
  return [
    ...files.map((f) => ({ url: new URL(f.path, base).href, sizeBytes: f.sizeBytes, md5: f.md5 })),
    { url: build.modelLib.url, sizeBytes: build.modelLib.sizeBytes },
  ];
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export class WebLlmStore {
  /** Every saved file, complete or not, by URL hash. */
  async list(): Promise<Map<string, WebLlmSavedFile>> {
    const files = new Map<string, WebLlmSavedFile>();
    let root: FileSystemDirectoryHandle;
    try {
      root = await (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT_DIR);
    } catch {
      return files; // nothing saved yet
    }
    const walk = async (dir: FileSystemDirectoryHandle) => {
      for await (const [name, handle] of entries(dir)) {
        if (handle.kind === 'directory') {
          await walk(handle as FileSystemDirectoryHandle);
          continue;
        }
        const match = /^([0-9a-f]{64})\.(bin|record\.json)$/.exec(name);
        if (!match) continue;
        const hash = match[1]!;
        const entry = files.get(hash) ?? { hash, payloadBytes: 0, dir };
        const file = await (handle as FileSystemFileHandle).getFile();
        if (match[2] === 'bin') entry.payloadBytes = file.size;
        else {
          try {
            const record = JSON.parse(await file.text()) as { url?: unknown; nbytes?: unknown };
            if (typeof record.url === 'string' && typeof record.nbytes === 'number') {
              entry.url = record.url;
              entry.recordBytes = record.nbytes;
            }
          } catch {
            // A damaged record: treated as an unfinished file.
          }
        }
        files.set(hash, entry);
      }
    };
    await walk(root);
    return files;
  }

  /** How much of `artifact` is saved, given `saved` from `list()`. */
  async state(artifact: WebLlmArtifact, saved: Map<string, WebLlmSavedFile>): Promise<WebLlmArtifactState> {
    let savedBytes = 0;
    let resumableBytes = 0;
    let completeFiles = 0;
    const urls = webllmUrls(artifact);
    for (const { url, sizeBytes } of urls) {
      const file = saved.get(await sha256Hex(url));
      if (!file) continue;
      savedBytes += Math.min(file.payloadBytes, sizeBytes);
      if (file.url === url && file.recordBytes === sizeBytes && file.payloadBytes === sizeBytes) {
        completeFiles++;
        resumableBytes += sizeBytes;
      }
    }
    const complete = completeFiles === urls.length;
    return { savedBytes, resumableBytes, complete, partial: !complete && savedBytes > 0 };
  }

  /** Deletes every saved file of `artifact` and its check record. */
  async delete(artifact: WebLlmArtifact): Promise<void> {
    const saved = await this.list();
    for (const { url } of webllmUrls(artifact)) {
      const file = saved.get(await sha256Hex(url));
      if (file) await removeFile(file);
    }
    await dbDelete('engineState', verifiedKey(artifact));
  }

  /**
   * Deletes saved files no listed URL needs: old revisions, models no longer in
   * the catalogue, and leftovers of unfinished downloads of those. Unfinished
   * files of listed URLs stay, so a download in another tab isn't disturbed.
   */
  async removeUnlisted(listedUrls: Iterable<string>): Promise<number> {
    const listed = new Set<string>();
    for (const url of listedUrls) listed.add(await sha256Hex(url));
    let removed = 0;
    for (const file of (await this.list()).values()) {
      if (listed.has(file.hash)) continue;
      await removeFile(file);
      removed++;
    }
    return removed;
  }

  /** Whether `artifact`'s weights were checked since they were downloaded. */
  async isVerified(artifact: WebLlmArtifact): Promise<boolean> {
    return (await dbGet('engineState', verifiedKey(artifact))) !== undefined;
  }

  /**
   * Checks each saved weight shard against the MD5 its build lists. Returns the
   * paths that don't match (empty: all good, and recorded as checked).
   */
  async verify(artifact: WebLlmArtifact, onProgress?: (doneBytes: number, totalBytes: number) => void): Promise<string[]> {
    const saved = await this.list();
    const shards = (WEBLLM_FILES[artifact.modelId] ?? []).filter((f) => f.md5);
    const urls = webllmUrls(artifact);
    const total = shards.reduce((n, f) => n + f.sizeBytes, 0);
    let done = 0;
    const bad: string[] = [];
    for (const shard of shards) {
      const url = urls.find((u) => u.md5 === shard.md5 && u.sizeBytes === shard.sizeBytes && u.url.endsWith(`/${shard.path}`))!.url;
      const file = saved.get(await sha256Hex(url));
      if (!file || file.payloadBytes !== shard.sizeBytes || (await md5OfFile(file)) !== shard.md5) bad.push(shard.path);
      done += shard.sizeBytes;
      onProgress?.(done, total);
    }
    if (bad.length === 0) await dbPut('engineState', { key: verifiedKey(artifact), at: Date.now() });
    return bad;
  }
}

function verifiedKey(artifact: WebLlmArtifact): string {
  return `webllm-verified:${artifact.build.repo}@${artifact.build.revision}`;
}

async function md5OfFile(file: WebLlmSavedFile): Promise<string> {
  const blob = await (await file.dir.getFileHandle(`${file.hash}.bin`)).getFile();
  const md5 = new Md5();
  const step = 8 * 1024 * 1024;
  for (let at = 0; at < blob.size; at += step) md5.update(new Uint8Array(await blob.slice(at, at + step).arrayBuffer()));
  return md5.hex();
}

async function removeFile(file: WebLlmSavedFile): Promise<void> {
  for (const name of [`${file.hash}.record.json`, `${file.hash}.bin`]) {
    await file.dir.removeEntry(name).catch((err: unknown) => {
      if (!(err instanceof DOMException && err.name === 'NotFoundError')) throw err;
    });
  }
}

/** `dir.entries()`, which the DOM type library doesn't declare yet. */
function entries(dir: FileSystemDirectoryHandle): AsyncIterable<[string, FileSystemHandle]> {
  return (dir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries();
}
