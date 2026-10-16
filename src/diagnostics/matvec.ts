// Development: hand-written WebGPU matrix-vector kernels for MatMulNBits' packed layouts, timed on this device, to
// see how close to memory bandwidth the browser can get (ONNX Runtime's kernels reach 40-70 GB/s here).
//
//   /matvec.html[?n=2048&k=2048&runs=200]
// Results are exposed as `window.__bench`.

import { getGpuDevice } from '../device/gpu';

const params = new URLSearchParams(location.search);
const N = Number(params.get('n') ?? 2048);
const K = Number(params.get('k') ?? 2048);
const RUNS = Number(params.get('runs') ?? 200);
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const MAP_READ = 0x1;

/**
 * y[n] = Σ_k x[k] · (q[n, k] − zero) · scale[n, k / block]. One subgroup per row: each lane reads 32-bit words of
 * packed weights (32 / bits of them per word, low bits first, as MatMulNBits packs them), accumulates in float32,
 * and the subgroup sums. ROWS rows per workgroup.
 */
function kernel(bits: 2 | 4, block: number, rows: number): string {
  const perWord = 32 / bits;
  const mask = (1 << bits) - 1;
  const zero = 1 << (bits - 1);
  return /* wgsl */ `
enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
const K: u32 = ${K}u;
const WORDS: u32 = K / ${perWord}u;
const BLOCKS: u32 = K / ${block}u;
const WORDS_PER_BLOCK: u32 = ${block / perWord}u;
@compute @workgroup_size(32, ${rows})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
  let wanted = wid.x * ${rows}u + lid.y;
  let row = min(wanted, ${N - 1}u);
  var acc = 0.0;
  let base = row * WORDS;
  for (var i = lane; i < WORDS; i += width) {
    let word = w[base + i];
    let scale = f32(scales[row * BLOCKS + i / WORDS_PER_BLOCK]);
    var part = 0.0;
    // ${perWord} weights: x[i * ${perWord} .. +${perWord}], read as vec4s.
    for (var v = 0u; v < ${perWord / 4}u; v++) {
      let xs = vec4<f32>(x[i * ${perWord / 4}u + v]);
      let s = word >> (${bits * 4}u * v);
      let q = vec4<f32>(f32(s & ${mask}u), f32((s >> ${bits}u) & ${mask}u), f32((s >> ${2 * bits}u) & ${mask}u), f32((s >> ${3 * bits}u) & ${mask}u)) - ${zero}.0;
      part += dot(xs, q);
    }
    acc += part * scale;
  }
  acc = subgroupAdd(acc);
  if (lane == 0u && wanted < ${N}u) { y[row] = f16(acc); }
}`;
}

/**
 * Version 2: each lane reads a vec4<u32> (4 words) per step, x staged in workgroup memory as vec4<f16>; a 4-word chunk
 * (within one scale block) is summed in `chunk` precision (f16 vectors run at twice the rate), then scaled into f32.
 */
function kernel2(bits: 2 | 4, block: number, rows: number, chunk: 'f16' | 'f32'): string {
  const perWord = 32 / bits;
  const mask = (1 << bits) - 1;
  const zero = 1 << (bits - 1);
  const vecsPerWord = perWord / 4;
  return /* wgsl */ `
enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
const K: u32 = ${K}u;
const CHUNKS: u32 = K / ${perWord * 4}u;
const BLOCKS: u32 = K / ${block}u;
const CHUNKS_PER_BLOCK: u32 = ${Math.max(1, block / (perWord * 4))}u;
var<workgroup> xs: array<vec4<f16>, ${K / 4}>;
fn unpack(word: u32, v: u32) -> vec4<${chunk}> {
  let s = word >> (${bits * 4}u * v);
  return vec4<${chunk}>(vec4<u32>(s & ${mask}u, (s >> ${bits}u) & ${mask}u, (s >> ${2 * bits}u) & ${mask}u, (s >> ${3 * bits}u) & ${mask}u)) - ${zero}.0;
}
@compute @workgroup_size(32, ${rows})
fn main(@builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
  for (var i = li; i < K / 4u; i += ${32 * rows}u) { xs[i] = x[i]; }
  workgroupBarrier();
  let wanted = wid.x * ${rows}u + lid.y;
  let row = min(wanted, ${N - 1}u);
  var acc = 0.0;
  for (var c = lane; c < CHUNKS; c += width) {
    let words = w[row * CHUNKS + c];
    var part = vec4<${chunk}>(0.0);
    let xbase = c * ${vecsPerWord * 4}u;
    for (var j = 0u; j < 4u; j++) {
      let word = words[j];
      for (var v = 0u; v < ${vecsPerWord}u; v++) {
        part += vec4<${chunk}>(xs[xbase + j * ${vecsPerWord}u + v]) * unpack(word, v);
      }
    }
    let scale = f32(scales[row * BLOCKS + c / CHUNKS_PER_BLOCK]);
    acc += f32(part.x + part.y + part.z + part.w) * scale;
  }
  acc = subgroupAdd(acc);
  if (lane == 0u && wanted < ${N}u) { y[row] = f16(acc); }
}`;
}

/**
 * Version 3: one subgroup computes R rows together (llama.cpp's Metal kernels do this): each step every lane issues
 * R independent vec4<u32> loads, so more memory requests are in flight; x (in workgroup memory) is read once for all R.
 * G subgroups per workgroup.
 */
function kernel3(bits: 2 | 4, block: number, r: number, g: number): string {
  const perWord = 32 / bits;
  const mask = (1 << bits) - 1;
  const zero = 1 << (bits - 1);
  const vecsPerWord = perWord / 4;
  const rows = r * g;
  return /* wgsl */ `
enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
const K: u32 = ${K}u;
const CHUNKS: u32 = K / ${perWord * 4}u;
const BLOCKS: u32 = K / ${block}u;
const CHUNKS_PER_BLOCK: u32 = ${Math.max(1, block / (perWord * 4))}u;
var<workgroup> xs: array<vec4<f16>, ${K / 4}>;
fn unpack(word: u32, v: u32) -> vec4<f16> {
  let s = word >> (${bits * 4}u * v);
  return vec4<f16>(vec4<u32>(s & ${mask}u, (s >> ${bits}u) & ${mask}u, (s >> ${2 * bits}u) & ${mask}u, (s >> ${3 * bits}u) & ${mask}u)) - ${zero}.0;
}
@compute @workgroup_size(32, ${g})
fn main(@builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
  for (var i = li; i < K / 4u; i += ${32 * g}u) { xs[i] = x[i]; }
  workgroupBarrier();
  let first = wid.x * ${rows}u + lid.y * ${r}u;
  var acc: array<f32, ${r}>;
  for (var c = lane; c < CHUNKS; c += width) {
    var words: array<vec4<u32>, ${r}>;
    for (var t = 0u; t < ${r}u; t++) { words[t] = w[min(first + t, ${N - 1}u) * CHUNKS + c]; }
    let xbase = c * ${vecsPerWord * 4}u;
    for (var t = 0u; t < ${r}u; t++) {
      var part = vec4<f16>(0.0);
      for (var j = 0u; j < 4u; j++) {
        for (var v = 0u; v < ${vecsPerWord}u; v++) {
          part += xs[xbase + j * ${vecsPerWord}u + v] * unpack(words[t][j], v);
        }
      }
      let scale = f32(scales[min(first + t, ${N - 1}u) * BLOCKS + c / CHUNKS_PER_BLOCK]);
      acc[t] += f32(part.x + part.y + part.z + part.w) * scale;
    }
  }
  for (var t = 0u; t < ${r}u; t++) {
    let total = subgroupAdd(acc[t]);
    if (lane == 0u && first + t < ${N}u) { y[first + t] = f16(total); }
  }
}`;
}

/**
 * Version 4: unpacks 4 weights at once. (word >> bits·k) & 0x03030303 (2-bit; 0x0F0F0F0F for 4-bit) leaves weights
 * k, k + 4, k + 8, k + 12 (2-bit; k, k + 2, k + 4, k + 6 for 4-bit) in its four bytes, which unpack4x8unorm turns into
 * a vec4 (×1/255); x is staged in workgroup memory in that order, so one dot product takes 4 weights. The zero point
 * comes out of the sum: Σ x·(q − z) = Σ x·q − z·Σ x. One subgroup per row, ROWS rows per workgroup.
 */
function kernel4(bits: 2 | 4, block: number, rows: number): string {
  const perWord = 32 / bits;
  const shifts = perWord / 4; // vec4s per word
  const zero = 1 << (bits - 1);
  const mask = bits === 2 ? '0x03030303u' : '0x0F0F0F0Fu';
  const chunkWeights = perWord * 4;
  return /* wgsl */ `
enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
const K: u32 = ${K}u;
const CHUNKS: u32 = K / ${chunkWeights}u;
const BLOCKS: u32 = K / ${block}u;
const CHUNKS_PER_BLOCK: u32 = ${Math.max(1, block / chunkWeights)}u;
// xs[word * ${shifts} + k] = the 4 x values that (word >> ${bits}·k) & mask pairs with.
var<workgroup> xs: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(32, ${rows})
fn main(@builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
  for (var i = li; i < K / 4u; i += ${32 * rows}u) {
    let word = i / ${shifts}u;
    let k = i % ${shifts}u;
    let base = word * ${perWord}u + k;
    xs[i] = vec4<f32>(f32(x[base]), f32(x[base + ${shifts}u]), f32(x[base + ${2 * shifts}u]), f32(x[base + ${3 * shifts}u]));
  }
  workgroupBarrier();
  let wanted = wid.x * ${rows}u + lid.y;
  let row = min(wanted, ${N - 1}u);
  var acc = 0.0;
  for (var c = lane; c < CHUNKS; c += width) {
    let words = w[row * CHUNKS + c];
    var dotq = 0.0;
    var sumx = 0.0;
    let xbase = c * ${4 * shifts}u;
    for (var j = 0u; j < 4u; j++) {
      for (var k = 0u; k < ${shifts}u; k++) {
        let xv = xs[xbase + j * ${shifts}u + k];
        dotq += dot(xv, unpack4x8unorm((words[j] >> (${bits}u * k)) & ${mask}));
        sumx += xv.x + xv.y + xv.z + xv.w;
      }
    }
    let scale = f32(scales[row * BLOCKS + c / CHUNKS_PER_BLOCK]);
    acc += scale * (255.0 * dotq - ${zero}.0 * sumx);
  }
  acc = subgroupAdd(acc);
  if (lane == 0u && wanted < ${N}u) { y[row] = f16(acc); }
}`;
}

async function run(): Promise<void> {
  const device = await getGpuDevice();
  if (!device.features.has('subgroups')) throw new Error('no subgroups');
  const results: Record<string, unknown> = {};
  for (const [bits, block] of [
    [2, 128],
    [4, 128],
    [4, 32],
  ] as const) {
    const wordsPerRow = (K * bits) / 32;
    const w = new Uint32Array(N * wordsPerRow);
    for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 ** 32) >>> 0;
    // Enough distinct copies (128 MB or more) that the reads come from memory, not the GPU's caches.
    const copies = Math.max(8, Math.ceil(128e6 / w.byteLength));
    const scales = new Uint16Array((N * K) / block).fill(0x2000);
    const x = new Uint16Array(K).fill(0x3c00);
    const buffer = (data: ArrayBufferView, usage = STORAGE | COPY_DST) => {
      const b = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage });
      device.queue.writeBuffer(b, 0, data as BufferSource);
      return b;
    };
    const xb = buffer(x);
    const wbs = Array.from({ length: copies }, () => buffer(w));
    const sbs = Array.from({ length: copies }, () => buffer(scales));
    const yb = device.createBuffer({ size: Math.ceil((N * 2) / 16) * 16, usage: STORAGE | COPY_SRC });
    const staging = device.createBuffer({ size: Math.ceil((N * 2) / 16) * 16, usage: MAP_READ | COPY_DST });
    // A 4-word chunk must stay within one scale block (4-bit block 32 has 32 weights in 4 words: fine; 2-bit: 64 in 4).
    const variants: [string, string][] = [
      ['v1 8 rows', kernel(bits, block, 8)],
      ...(block >= (32 / bits) * 4 ? ([
        ['v2 f32 4 rows', kernel2(bits, block, 4, 'f32')],
        ['v2 f16 4 rows', kernel2(bits, block, 4, 'f16')],
        ['v2 f16 8 rows', kernel2(bits, block, 8, 'f16')],
        ['v3 r2 g4 8 rows', kernel3(bits, block, 2, 4)],
        ['v4 4 rows', kernel4(bits, block, 4)],
        ['v4 8 rows', kernel4(bits, block, 8)],
        ['v4 16 rows', kernel4(bits, block, 16)],
      ] as [string, string][]) : []),
    ];
    for (const [variant, code] of variants) {
      const rows = Number(/(\d+) rows/.exec(variant)![1]);
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      for (const m of info.messages) log(`${m.type}: ${m.message} (line ${m.lineNum})`);
      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const error = await device.popErrorScope();
      if (error) throw new Error(`pipeline: ${error.message}`);
      const binds = wbs.map((wb, c) =>
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [xb, wb, sbs[c]!, yb].map((b, i) => ({ binding: i, resource: { buffer: b } })),
        }),
      );
      const time = async (count: number) => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < count; i++) {
          pass.setBindGroup(0, binds[i % binds.length]!);
          pass.dispatchWorkgroups(Math.ceil(N / rows));
        }
        pass.end();
        encoder.copyBufferToBuffer(yb, 0, staging, 0, staging.size);
        const start = performance.now();
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(MAP_READ);
        const ms = performance.now() - start;
        state.results.y0 = new Uint16Array(staging.getMappedRange().slice(0, 4))[0];
        staging.unmap();
        return ms;
      };
      await time(20);
      const ms = await time(RUNS);
      const us = (ms * 1000) / RUNS;
      const bytes = w.byteLength + scales.byteLength;
      const key = `${bits}-bit block ${block}, ${variant}`;
      results[key] = { usPerOp: +us.toFixed(1), GBps: +(bytes / us / 1000).toFixed(1) };
      log(`${key}: ${us.toFixed(1)} µs (${(bytes / us / 1000).toFixed(1)} GB/s)`);
    }
  }
  state.results = results;
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
