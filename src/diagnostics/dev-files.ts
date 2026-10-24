// Development pages' model files, from the dev server's local model cache (tools/reference/.cache).

import { artifactUrl, type ArtifactFile } from '../catalogue/models';
import type { ByteSource } from '../runtime/onnx/external-weights';

const CHUNK = 64 * 1024 * 1024;
const inWorker = typeof document === 'undefined';

/**
 * A file from the dev server read by byte range when asked (synchronous requests, as ByteSource reads are): for
 * models bigger than this page's memory. Binary through the x-user-defined charset (no arraybuffer for synchronous
 * requests on a page).
 */
export function rangedFile(model: Parameters<typeof artifactUrl>[0], file: ArtifactFile): ByteSource {
  const url = artifactUrl(model, file);
  return {
    size: file.sizeBytes,
    read(offset, length) {
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

