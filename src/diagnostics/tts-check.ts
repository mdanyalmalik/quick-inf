// Development check of speech-model graphs on ONNX Runtime WebGPU: replays inputs recorded from each model's own
// pipeline on the CPU (tools/export/.cache/tts-eval/tts_capture.py, into the dev server's model cache under
// tts-check/0/<model>/<graph>/) and reports, per graph, the nodes ONNX Runtime left on the CPU, the run time,
// and how close the outputs are to the CPU's.
//
//   /tts-check.html?models=kitten-nano,soprano[&runs=5]
// Results are exposed as `window.__bench` (as bench.html's, for tools/bench/bench.mjs --page tts-check).

import type { InferenceSession, Tensor } from 'onnxruntime-web/webgpu';
import { getGpuDevice } from '../device/gpu';
import { loadOrt } from '../runtime/onnx/ort-env';

interface Entry {
  dtype: string;
  dims: number[];
  offset: number;
  length: number;
}
interface Call {
  feeds: Record<string, Entry>;
  outputs: Record<string, Entry>;
  feedBytes: number;
}

const params = new URLSearchParams(location.search);
const models = (params.get('models') ?? '').split(',').filter(Boolean);
const runs = Number(params.get('runs') ?? 5);
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

// ONNX Runtime lists node placements at verbose log level: a "Node(s) placed on [<EP>]" header, then one
// "  <OpType> (<name>)" line per node. The op types of the nodes under the CPU header are counted per session.
let placement: Record<string, number> = {};
let onCpu = false;
for (const level of ['log', 'info', 'warn', 'error', 'debug'] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    const text = args.map(String).join(' ');
    if (text.includes('VerifyEachNodeIsAssignedToAnEp')) {
      const header = /Node\(s\) placed on \[(\w+)\]/.exec(text);
      const node = /AnEp\]\s{3}(\w+) \(/.exec(text);
      if (header) onCpu = header[1] === 'CPUExecutionProvider';
      else if (node && onCpu) placement[node[1]!] = (placement[node[1]!] ?? 0) + 1;
      return;
    }
    original(...args);
  };
}

const ARRAYS = {
  float32: Float32Array,
  int64: BigInt64Array,
  int32: Int32Array,
  bool: Uint8Array,
  uint8: Uint8Array,
  int8: Int8Array,
  float16: Uint16Array,
} as const;

function view(buffer: ArrayBuffer, base: number, e: Entry): InstanceType<(typeof ARRAYS)[keyof typeof ARRAYS]> {
  const Type = ARRAYS[e.dtype as keyof typeof ARRAYS];
  if (!Type) throw new Error(`unsupported dtype ${e.dtype}`);
  return new Type(buffer.slice(base + e.offset, base + e.offset + e.length));
}

function half(bits: number): number {
  const exponent = (bits >> 10) & 0x1f;
  const value = exponent ? (1 + (bits & 0x3ff) / 1024) * 2 ** (exponent - 15) : ((bits & 0x3ff) / 1024) * 2 ** -14;
  return bits & 0x8000 ? -value : value;
}

/** A float16 tensor's values as numbers (its data is raw bits in a Uint16Array). */
function numbers(t: Tensor): number[] {
  const data = t.data as ArrayLike<number | bigint>;
  return t.type === 'float16' && data instanceof Uint16Array ? Array.from(data, half) : Array.from(data, Number);
}

function compare(got: Tensor, want: ArrayLike<number | bigint>): { cosine: number; maxAbs: number } {
  const a = numbers(got);
  const b = Array.from(want, Number);
  if (a.length !== b.length) return { cosine: NaN, maxAbs: NaN };
  let dot = 0;
  let na = 0;
  let nb = 0;
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
    maxAbs = Math.max(maxAbs, Math.abs(a[i]! - b[i]!));
  }
  // Two all-zero outputs (e.g. an empty or silent tensor) count as equal.
  return { cosine: na === 0 && nb === 0 ? 1 : dot / Math.sqrt(na * nb), maxAbs };
}

async function run(): Promise<void> {
  const ort = await loadOrt();
  ort.env.logLevel = 'verbose';
  const device = await getGpuDevice();
  const index = (await (await fetch('/__models/tts-check/0/index.json')).json()) as Record<string, Record<string, number>>;
  for (const model of models.length ? models : Object.keys(index)) {
    for (const [graph, calls] of Object.entries(index[model] ?? {})) {
      const base = `/__models/tts-check/0/${model}/${graph}`;
      const bytes = new Uint8Array(await (await fetch(`${base}/model.onnx`)).arrayBuffer());
      // Weights in a separate file (the GenAI builder's exports); the dev server redirects missing files to Hugging Face.
      const external = (await fetch(`${base}/model.onnx.data`, { method: 'HEAD', redirect: 'manual' })).ok;
      placement = {};
      onCpu = false;
      const key = `${model}/${graph}`;
      let session: InferenceSession;
      try {
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: [{ name: 'webgpu', device } as InferenceSession.WebGpuExecutionProviderOption],
          graphOptimizationLevel: 'all',
          ...(external ? { externalData: [{ path: 'model.onnx.data', data: `${base}/model.onnx.data` }] } : {}),
          logSeverityLevel: 0,
          logVerbosityLevel: 0,
        });
      } catch (err) {
        state.results[key] = { error: String(err) };
        log(`${key}: session failed: ${String(err)}`);
        continue;
      }
      const result: Record<string, unknown> = { cpuNodes: { ...placement }, calls: [] };
      for (let n = 0; n < calls; n++) {
        const meta = (await (await fetch(`${base}/call-${n}.json`)).json()) as Call;
        const data = await (await fetch(`${base}/call-${n}.bin`)).arrayBuffer();
        const feeds: Record<string, Tensor> = {};
        for (const [name, e] of Object.entries(meta.feeds)) {
          feeds[name] = new ort.Tensor(e.dtype as Tensor.Type, view(data, 0, e) as Tensor.DataType, e.dims);
        }
        try {
          let outputs = await session.run(feeds);
          const start = performance.now();
          for (let i = 0; i < runs; i++) outputs = await session.run(feeds);
          const ms = (performance.now() - start) / runs;
          const scores: Record<string, unknown> = {};
          for (const [name, e] of Object.entries(meta.outputs)) {
            const got = outputs[name];
            const s = got ? compare(got, view(data, meta.feedBytes, e) as ArrayLike<number | bigint>) : { missing: true };
            scores[name] = s;
          }
          (result.calls as unknown[]).push({ ms: +ms.toFixed(2), outputs: scores });
          log(`${key} call ${n}: ${ms.toFixed(2)} ms; ${Object.entries(scores).map(([k, v]) => `${k} ${JSON.stringify(v)}`).join('; ')}`);
        } catch (err) {
          (result.calls as unknown[]).push({ error: String(err) });
          log(`${key} call ${n}: run failed: ${String(err)}`);
        }
      }
      log(`${key}: on the CPU: ${JSON.stringify(result.cpuNodes)}`);
      state.results[key] = result;
      await session.release();
    }
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
