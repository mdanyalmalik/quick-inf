// Development pages' model files, from the dev server's local model cache (tools/reference/.cache).

import { artifactUrl, type ArtifactFile } from '../catalogue/models';
import type { ByteSource } from '../runtime/onnx/external-weights';

const CHUNK = 64 * 1024 * 1024;
/** Prefetch requests in flight at once, and the gap between ranges still fetched as one request. */
const PREFETCH_REQUESTS = 4;
const MERGE_GAP = 1024 * 1024;
const inWorker = typeof document === 'undefined';

/** Bytes read from the dev server so far (by every file of this page or worker). */
export const devFileStats = { bytesRead: 0 };

/**
 * A file from the dev server read by byte range when asked (synchronous requests, as ByteSource reads are): for
 * models bigger than this page's memory. Binary through the x-user-defined charset (no arraybuffer for synchronous
 * requests on a page). Ranges known ahead (a segment's weights) can be prefetched in parallel; their reads are then
 * answered from memory.
 */
export function rangedFile(model: Parameters<typeof artifactUrl>[0], file: ArtifactFile): ByteSource {
  const url = artifactUrl(model, file);
  // Prefetched blocks, each dropped once every byte of it has been read.
  const blocks: { offset: number; bytes: Uint8Array<ArrayBuffer>; left: number }[] = [];
  return {
    size: file.sizeBytes,
    async prefetch(ranges) {
      // Neighbouring ranges merged into requests of up to CHUNK bytes, PREFETCH_REQUESTS at a time.
      const sorted = [...ranges].sort((a, b) => a.offset - b.offset);
      const requests: { offset: number; end: number }[] = [];
      for (const r of sorted) {
        const last = requests.at(-1);
        if (last && r.offset - last.end <= MERGE_GAP && r.offset + r.length - last.offset <= CHUNK) last.end = Math.max(last.end, r.offset + r.length);
        else requests.push({ offset: r.offset, end: r.offset + r.length });
      }
      let next = 0;
      const worker = async () => {
        while (next < requests.length) {
          const { offset, end } = requests[next++]!;
          const response = await fetch(url, { headers: { Range: `bytes=${offset}-${end - 1}` } });
          if (response.status !== 206) throw new Error(`${file.path}: range read failed (${response.status})`);
          const bytes = new Uint8Array(await response.arrayBuffer());
          if (bytes.byteLength !== end - offset) throw new Error(`${file.path}: range read returned ${bytes.byteLength} bytes`);
          devFileStats.bytesRead += bytes.byteLength;
          blocks.push({ offset, bytes, left: bytes.byteLength });
        }
      };
      await Promise.all(Array.from({ length: PREFETCH_REQUESTS }, worker));
    },
    read(offset, length) {
      const block = blocks.find((b) => offset >= b.offset && offset + length <= b.offset + b.bytes.byteLength);
      if (block) {
        const bytes = block.bytes.slice(offset - block.offset, offset - block.offset + length);
        block.left -= length;
        if (block.left <= 0) blocks.splice(blocks.indexOf(block), 1);
        return bytes;
      }
      devFileStats.bytesRead += length;
      const bytes = new Uint8Array(length);
      for (let done = 0; done < length; ) {
        const take = Math.min(length - done, CHUNK);
        const xhr = new XMLHttpRequest();
        xhr.open('GET', url, false);
        // Workers can take binary synchronously; a page only as text.
        if (inWorker) xhr.responseType = 'arraybuffer';
        else xhr.overrideMimeType('text/plain; charset=x-user-defined');
        xhr.setRequestHeader('Range', `bytes=${offset + done}-${offset + done + take - 1}`);
        xhr.send();
        if (xhr.status !== 206) throw new Error(`${file.path}: range read failed (${xhr.status})`);
        if (inWorker) {
          const got = new Uint8Array(xhr.response as ArrayBuffer);
          if (got.byteLength !== take) throw new Error(`${file.path}: range read returned ${got.byteLength} bytes`);
          bytes.set(got, done);
        } else {
          const text = xhr.responseText;
          if (text.length !== take) throw new Error(`${file.path}: range read returned ${text.length} bytes`);
          for (let i = 0; i < take; i++) bytes[done + i] = text.charCodeAt(i) & 0xff;
        }
        done += take;
      }
      return bytes;
    },
  };
}

