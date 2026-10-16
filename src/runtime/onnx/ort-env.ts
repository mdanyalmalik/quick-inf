// The only module that configures ONNX Runtime Web globally. The WASM binaries are
// served by the app itself (not a CDN) so the runtime can later work offline.
//
// Which build runs is chosen once per worker, by what the browser supports (as wllama does):
//   jspi      WebGPU + CPU, using WebAssembly's JS Promise Integration (Chrome 137+, Safari 27+):
//             no Asyncify, so a smaller binary.
//   asyncify  WebGPU + CPU, for browsers without JSPI.
//   cpu       CPU only, the plain build: for WebKit without JSPI, where compiling the
//             Asyncify build makes Safari's WebAssembly compiler use gigabytes and the tab is
//             killed (WebKit bug 304810; onnxruntime issue 26827), and where WebGPU is missing.
// Multi-threaded CPU inference needs the page to be cross-origin isolated (COOP + COEP headers).

import type * as OrtModule from 'onnxruntime-web/webgpu';
import asyncifyMjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import asyncifyWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import jspiMjs from 'onnxruntime-web/ort-wasm-simd-threaded.jspi.mjs?url';
import jspiWasm from 'onnxruntime-web/ort-wasm-simd-threaded.jspi.wasm?url';
import plainMjs from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import plainWasm from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { simulation } from '../../device/simulation';

export type Ort = typeof OrtModule;
export type OrtBuild = 'jspi' | 'asyncify' | 'cpu';

export interface BuildSignals {
  /** `'Suspending' in WebAssembly`. */
  jspi: boolean;
  /** Apple's WebKit engine: Safari, and every browser on iPhone and iPad. */
  webkit: boolean;
  /** `navigator.gpu` exists. */
  webgpu: boolean;
}

/** The build for these browser capabilities (see the top of this file). */
export function chooseBuild(signals: BuildSignals): OrtBuild {
  if (!signals.webgpu) return 'cpu';
  if (signals.jspi) return 'jspi';
  return signals.webkit ? 'cpu' : 'asyncify';
}

/** WebKit, not Blink or Gecko: every iOS browser (CriOS, FxiOS…) and Safari, but not desktop Chrome or Edge. */
export function isWebKit(userAgent: string): boolean {
  return /AppleWebKit/.test(userAgent) && !/Chrome\/|Chromium\/|Edg\/|OPR\//.test(userAgent);
}

export function browserSignals(): BuildSignals {
  const nav = globalThis.navigator as (Navigator & { gpu?: unknown }) | undefined;
  return {
    jspi: typeof WebAssembly !== 'undefined' && 'Suspending' in WebAssembly,
    webkit: isWebKit(nav?.userAgent ?? ''),
    webgpu: !!nav?.gpu && !simulation()?.noWebgpu,
  };
}

let loaded: { ort: Ort; build: OrtBuild } | null = null;
let loading: Promise<Ort> | null = null;

/** Loads (once) the ONNX Runtime build for this browser (`?sim=ort=…` forces one in development). */
export function loadOrt(): Promise<Ort> {
  loading ??= (async () => {
    const build = simulation()?.ort ?? chooseBuild(browserSignals());
    const ort = (
      build === 'jspi' ? await import('onnxruntime-web/jspi') : build === 'cpu' ? await import('onnxruntime-web/wasm') : await import('onnxruntime-web/webgpu')
    ) as Ort;
    // The glue and the binary it was built with, always as a pair (vite.config.ts: the builds without glue inside).
    ort.env.wasm.wasmPaths =
      build === 'jspi' ? { mjs: jspiMjs, wasm: jspiWasm } : build === 'cpu' ? { mjs: plainMjs, wasm: plainWasm } : { mjs: asyncifyMjs, wasm: asyncifyWasm };
    // Threads need SharedArrayBuffer, i.e. a cross-origin isolated page; 0 lets ONNX Runtime pick (half the cores, at most 4).
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? 0 : 1;
    ort.env.logLevel = 'warning';
    loaded = { ort, build };
    return ort;
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** The loaded runtime (after {@link loadOrt}). */
export function getOrt(): Ort {
  if (!loaded) throw new Error('ONNX Runtime is not loaded yet.');
  return loaded.ort;
}

/** The loaded build, or null before {@link loadOrt}. */
export function ortBuild(): OrtBuild | null {
  return loaded?.build ?? null;
}

/** Whether the loaded build can use WebGPU. */
export function ortHasWebGpu(): boolean {
  return loaded !== null && loaded.build !== 'cpu';
}
