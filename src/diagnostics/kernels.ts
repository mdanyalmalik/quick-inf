// Development kernel timing: runs synthetic graphs that chain one op many times (made by a script into the dev
// server's model cache, tools/reference/.cache/micro/0/<name>.onnx) through ONNX Runtime WebGPU on this app's device,
// with graph capture, and reports the time per op. Input x and output y are [1, M, K] float16 (M = 1 by default).
//
//   /kernels.html?graphs=sigmoid200:200,q2b128_x40:40[&k=2048][&m=1][&runs=50]
// Results are exposed as `window.__bench` (as bench.html's, for tools/bench/bench.mjs --page kernels).

import type { InferenceSession } from 'onnxruntime-web/webgpu';
import { getGpuDevice } from '../device/gpu';
import { loadOrt } from '../runtime/onnx/ort-env';

const params = new URLSearchParams(location.search);
const graphs = (params.get('graphs') ?? '').split(',').filter(Boolean);
const k = Number(params.get('k') ?? 2048);
const m = Number(params.get('m') ?? 1);
const runs = Number(params.get('runs') ?? 50);
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = await getGpuDevice();
  const bytes = m * k * 2;
  const make = () => device.createBuffer({ size: bytes, usage: 0x80 | 0x4 | 0x8 });
  const xBuffer = make();
  const yBuffer = make();
  device.queue.writeBuffer(xBuffer, 0, new Uint16Array(m * k).fill(0x3c00));
  const x = ort.Tensor.fromGpuBuffer(xBuffer, { dataType: 'float16', dims: [1, m, k] });
  const y = ort.Tensor.fromGpuBuffer(yBuffer, { dataType: 'float16', dims: [1, m, k] });
  const staging = device.createBuffer({ size: bytes, usage: 0x1 | 0x8 });
  const sync = async () => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(yBuffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(0x1);
    staging.unmap();
  };
  for (const spec of graphs) {
    const [name, countText] = spec.split(':');
    const count = Number(countText ?? 1);
    const model = new Uint8Array(await (await fetch(`/__models/micro/0/${name}.onnx`)).arrayBuffer());
    for (const capture of [true, false]) {
      const session: InferenceSession = await ort.InferenceSession.create(model, {
        executionProviders: [
          { name: 'webgpu', device, qiOptions: capture ? { enableGraphCapture: '1', enableInt64: '1' } : {} } as InferenceSession.WebGpuExecutionProviderOption,
        ],
        preferredOutputLocation: 'gpu-buffer',
        graphOptimizationLevel: 'all',
      });
      const options = { extra: { gpu_graph_id: capture ? '0' : '-1' } };
      for (let i = 0; i < 5; i++) await session.run({ x }, { y }, options);
      await sync();
      const start = performance.now();
      for (let i = 0; i < runs; i++) await session.run({ x }, { y }, options);
      await sync();
      const perRun = (performance.now() - start) / runs;
      const key = `${name}${capture ? '' : ' (uncaptured)'}`;
      state.results[key] = { msPerRun: +perRun.toFixed(3), usPerOp: +((perRun * 1000) / count).toFixed(1) };
      log(`${key}: ${perRun.toFixed(3)} ms per run, ${((perRun * 1000) / count).toFixed(1)} µs per op`);
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
