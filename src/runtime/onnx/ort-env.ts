// The only module that configures ONNX Runtime Web globally. The WASM binary is
// served by the app itself (not a CDN) so the runtime can later work offline.

import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';

let configured = false;

export function getOrt(): typeof ort {
  if (!configured) {
    ort.env.wasm.wasmPaths = { wasm: wasmUrl };
    // Multi-threaded WASM needs cross-origin isolation; with WebGPU doing the
    // heavy work, one CPU thread is enough.
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'warning';
    configured = true;
  }
  return ort;
}

export const ORT_VERSION = ort.env.versions.web ?? 'unknown';
