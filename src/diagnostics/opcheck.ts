// Development op check: runs single-op graphs (made by a script into the dev server's model cache,
// tools/reference/.cache/micro/0/<name>.onnx, with <name>.json listing inputs and outputs and <name>.<tensor>.bin
// holding their data, expected outputs as float32) through ONNX Runtime WebGPU and reports how far each output is from
// what was expected: the largest error and the largest error relative to the largest expected value.
//
//   /opcheck.html?cases=matmul_iq2s_m1,matmul_iq2s_m40
// Results are exposed as `window.__bench` (as bench.html's, for tools/bench/bench.mjs --page opcheck).

import type { InferenceSession, Tensor } from 'onnxruntime-web/webgpu';
import { getGpuDevice } from '../device/gpu';
import { loadOrt } from '../runtime/onnx/ort-env';

interface TensorSpec {
  name: string;
  type: 'float16' | 'float32' | 'int32' | 'int64' | 'uint8';
  dims: number[];
}

const params = new URLSearchParams(location.search);
const cases = (params.get('cases') ?? '').split(',').filter(Boolean);
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

const ARRAYS = { float16: Uint16Array, float32: Float32Array, int32: Int32Array, int64: BigInt64Array, uint8: Uint8Array };

function halfToFloat(h: number): number {
  const exponent = (h >> 10) & 31;
  const mantissa = h & 1023;
  const value = exponent === 0 ? mantissa * 2 ** -24 : exponent === 31 ? (mantissa ? NaN : Infinity) : (1 + mantissa / 1024) * 2 ** (exponent - 15);
  return h & 0x8000 ? -value : value;
}

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = await getGpuDevice();
  for (const name of cases) {
    const base = `/__models/micro/0/${name}`;
    // `model`: a graph elsewhere in the cache (with its external data files), instead of <name>.onnx.
    const spec = (await (await fetch(`${base}.json`)).json()) as { inputs: TensorSpec[]; outputs: TensorSpec[]; model?: string; externalData?: string[] };
    const read = async (tensor: string) => (await (await fetch(`${base}.${tensor}.bin`)).arrayBuffer());
    const feeds: Record<string, Tensor> = {};
    for (const input of spec.inputs) {
      const empty = input.dims.includes(0);
      feeds[input.name] = new ort.Tensor(input.type, new ARRAYS[input.type](empty ? new ArrayBuffer(0) : await read(input.name)), input.dims);
    }
    const graph = spec.model ? `/__models/micro/0/${spec.model}` : `${base}.onnx`;
    const model = new Uint8Array(await (await fetch(graph)).arrayBuffer());
    const folder = graph.slice(0, graph.lastIndexOf('/') + 1);
    const session: InferenceSession = await ort.InferenceSession.create(model, {
      executionProviders: [{ name: 'webgpu', device } as InferenceSession.WebGpuExecutionProviderOption],
      graphOptimizationLevel: 'all',
      externalData: (spec.externalData ?? []).map((path) => ({ path, data: folder + path })),
    });
    const results = await session.run(feeds);
    for (const output of spec.outputs) {
      const want = new Float32Array(await read(output.name));
      const data = results[output.name]!.data as Uint16Array | Float32Array;
      const got = data instanceof Uint16Array ? Float32Array.from(data, halfToFloat) : data;
      let maxError = 0;
      let maxValue = 0;
      let bad = 0;
      for (let i = 0; i < want.length; i++) {
        const value = got[i]!;
        const error = Math.abs(value - want[i]!);
        if (!(error <= maxError)) maxError = Number.isNaN(error) ? Infinity : Math.max(maxError, error);
        if (!Number.isFinite(value)) bad++;
        maxValue = Math.max(maxValue, Math.abs(want[i]!));
      }
      const key = `${name}.${output.name}`;
      state.results[key] = { maxError, relative: maxError / maxValue, nonFinite: bad, size: want.length === got.length };
      log(`${key}: max error ${maxError.toPrecision(3)} (${(maxError / maxValue).toExponential(2)} of the largest value), ${bad} non-finite${want.length === got.length ? '' : `, size ${got.length} vs ${want.length}`}`);
    }
    await session.release();
  }
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
