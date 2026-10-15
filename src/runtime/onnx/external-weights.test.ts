// Rewrites each cached catalogue model with externalizeWeights and checks the
// result. tools/reference/check_external.py then loads the rewritten structure
// with ONNX Runtime (CPU) and confirms its outputs equal the original model's.

import { closeSync, existsSync, fstatSync, openSync, readSync, writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { catalogue } from '../../catalogue/models';
import { bytesSource, externalizeWeights, lazyBytes, type ByteSource } from './external-weights';

function fileSource(path: string): ByteSource & { close(): void } {
  const fd = openSync(path, 'r');
  return {
    size: fstatSync(fd).size,
    read(offset, length) {
      const buf = new Uint8Array(length);
      readSync(fd, buf, 0, length, offset);
      return buf;
    },
    close: () => closeSync(fd),
  };
}

for (const model of catalogue) {
  for (const artifact of model.artifacts) {
    // Single-file graphs only: split exports keep their weights outside already.
    if (artifact.engine !== 'onnx-webgpu' || artifact.externalData || artifact.embeddings) continue;
    const dir = new URL(`../../../tools/reference/.cache/${model.source.repo.replace('/', '__')}/${model.source.revision}/`, import.meta.url);
    const path = new URL(artifact.graph.path, dir);
    describe.skipIf(!existsSync(path))(`${model.name} ${artifact.precision}`, () => {
      it('moves the weights out and leaves a small structure', () => {
        const source = fileSource(path.pathname);
        const location = artifact.graph.path.split('/').pop()!;
        const result = externalizeWeights(source, location);
        source.close();
        expect(result.tensors).toBeGreaterThan(100);
        expect(result.bytes).toBeGreaterThan(artifact.graph.sizeBytes * 0.95);
        expect(result.graph.byteLength).toBeLessThan(artifact.graph.sizeBytes * 0.05);
        // The output layer's MatMul by the transposed embedding table, in the exports that have it.
        expect(result.fusedTransposes).toBeLessThanOrEqual(1);
        // Every export's output layer is found and fed the last position only.
        expect(result.lastTokenLogits).toBe(true);
        // transformers.js's fused-attention exports have the causal-mask subgraph to replace.
        if (model.source.repo.endsWith('-MHA')) expect(result.attentionBias).toBe(true);
        // Saved beside the original, which serves as its external data file.
        writeFileSync(new URL(`${artifact.graph.path}.external-graph`, dir), result.graph);
      });
    });
  }
}

const SMOLLM = new URL('../../../tools/reference/.cache/HuggingFaceTB__SmolLM2-360M-Instruct/a10cc1512eabd3dde888204e902eca88bddb4951/onnx/model_q4f16.onnx', import.meta.url).pathname;

describe.skipIf(!existsSync(SMOLLM))('quantized output layer (SmolLM2 360M, fp16 table)', () => {
  it('replaces the table with an 8-bit copy that decodes to within half a step', () => {
    const source = fileSource(SMOLLM);
    const result = externalizeWeights(source, 'model_q4f16.onnx', { quantizeOutput: 8 });
    expect(result.fusedTransposes).toBe(1);
    expect(result.extraFiles).toHaveLength(1);
    const [rows, cols] = [49152, 960];
    const blocks = cols / 32;
    const file = result.extraFiles[0]!.source;
    expect(file.size).toBe(rows * cols + rows * blocks * 2);
    const q = file.read(0, rows * cols);
    const scaleBytes = file.read(rows * cols, rows * blocks * 2);
    const scales = new Uint16Array(scaleBytes.buffer, scaleBytes.byteOffset, rows * blocks);
    // The table's bytes in the original file: the first tensor of exactly rows × cols fp16 values.
    const table = findRawTensor(source, rows * cols * 2);
    const half = (h: number) => {
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      return (h & 0x8000 ? -1 : 1) * (e === 0 ? 2 ** -14 * (f / 1024) : 2 ** (e - 15) * (1 + f / 1024));
    };
    for (const row of [0, 1, 12345, rows - 1]) {
      const original = source.read(table + row * cols * 2, cols * 2);
      const values = new Uint16Array(original.buffer, original.byteOffset, cols);
      for (let c = 0; c < cols; c++) {
        const scale = half(scales[row * blocks + Math.floor(c / 32)]!);
        const decoded = (q[row * cols + c]! - 128) * scale;
        expect(Math.abs(decoded - half(values[c]!))).toBeLessThanOrEqual(scale / 2 + 1e-6);
      }
    }
    source.close();
  });
});

/** Offset of the first raw_data field of exactly `length` bytes (protobuf: tag 0x4a, then the length varint). */
function findRawTensor(source: ByteSource, length: number): number {
  const varint: number[] = [];
  for (let v = length; ; v = Math.floor(v / 128)) {
    if (v < 128) {
      varint.push(v);
      break;
    }
    varint.push((v % 128) | 0x80);
  }
  const pattern = [0x4a, ...varint];
  const chunk = 1 << 20;
  for (let at = 0; at < source.size; at += chunk - pattern.length) {
    const bytes = source.read(at, Math.min(chunk, source.size - at));
    outer: for (let i = 0; i + pattern.length <= bytes.length; i++) {
      for (let k = 0; k < pattern.length; k++) if (bytes[i + k] !== pattern[k]) continue outer;
      return at + i + pattern.length;
    }
  }
  throw new Error('table not found');
}

describe('lazyBytes', () => {
  it('answers byteLength and subarray from the source, like the external data loader uses them', () => {
    const data = new Uint8Array([10, 11, 12, 13, 14, 15]);
    const reads: [number, number][] = [];
    const lazy = lazyBytes({ size: 6, read: (o, l) => (reads.push([o, l]), data.slice(o, o + l)) });
    expect(lazy).toBeInstanceOf(Uint8Array);
    expect(lazy.byteLength).toBe(6);
    expect([...lazy.subarray(2, 5)]).toEqual([12, 13, 14]);
    expect(reads).toEqual([[2, 3]]);
  });

  it('bytesSource reads without copying', () => {
    const data = new Uint8Array([1, 2, 3, 4]);
    const view = bytesSource(data).read(1, 2);
    expect(view.buffer).toBe(data.buffer);
  });
});

// Tiny graphs for the rewrites of ops ONNX Runtime WebGPU lacks (made by a script with the onnx package; their
// outputs were checked against ONNX Runtime's CPU backend or numpy in the browser when the rewrites were written).
const fixture = (name: string) => fileSource(new URL(`./__fixtures__/${name}.onnx`, import.meta.url).pathname);
const ops = (graph: Uint8Array) => new TextDecoder('latin1').decode(graph);

describe('rewrites for ops WebGPU lacks', () => {
  it('GroupNorm becomes InstanceNormalization (in float32) and the affine and SiLU steps', () => {
    const source = fixture('rewrite-groupnorm');
    expect(ops(source.read(0, source.size))).toContain('GroupNorm');
    const text = ops(externalizeWeights(source, 'g.onnx', { externalizeOnly: true }).graph);
    source.close();
    expect(text).not.toContain('GroupNorm');
    for (const op of ['InstanceNormalization', 'Cast', 'Sigmoid', 'Reshape']) expect(text).toContain(op);
  });

  it('int4 DequantizeLinear → MatMul becomes MatMulNBits over the same values, repacked', () => {
    const source = fixture('rewrite-dq-int4');
    expect(ops(source.read(0, source.size))).toContain('DequantizeLinear');
    const result = externalizeWeights(source, 'dq.onnx', { externalizeOnly: true });
    const text = ops(result.graph);
    expect(text).toContain('MatMulNBits');
    expect(text).not.toContain('DequantizeLinear');
    const [k, n, block] = [64, 6, 32];
    const blocks = k / block;
    const file = result.extraFiles[0]!.source;
    expect(file.size).toBe((n * k) / 2 + n * blocks * 2);
    const packed = file.read(0, (n * k) / 2);
    const scales = file.read((n * k) / 2, n * blocks * 2);
    const original = source.read(findRawTensor(source, (k * n) / 2), (k * n) / 2);
    const originalScales = source.read(findRawTensor(source, blocks * n * 2), blocks * n * 2);
    for (let row = 0; row < k; row++) {
      for (let col = 0; col < n; col++) {
        const i = row * n + col;
        const nibble = (original[i >> 1]! >> ((i & 1) * 4)) & 0x0f;
        const value = nibble >= 8 ? nibble - 16 : nibble;
        const j = col * k + row;
        const repacked = ((packed[j >> 1]! >> ((j & 1) * 4)) & 0x0f) - 8;
        expect(repacked).toBe(value);
      }
    }
    // Scales [K / block, N] → [N, K / block].
    for (let b = 0; b < blocks; b++)
      for (let col = 0; col < n; col++)
        expect([...scales.subarray((col * blocks + b) * 2, (col * blocks + b) * 2 + 2)]).toEqual([
          ...originalScales.subarray((b * n + col) * 2, (b * n + col) * 2 + 2),
        ]);
    source.close();
  });

  it('ONNX Attention with a mask and a past cache becomes standard ops', () => {
    const source = fixture('rewrite-attention');
    expect(ops(source.read(0, source.size))).toContain('Attention');
    const text = ops(externalizeWeights(source, 'a.onnx', { externalizeOnly: true }).graph);
    source.close();
    expect(text).not.toContain('Attention');
    for (const op of ['MatMul', 'Softmax', 'Concat', 'Transpose']) expect(text).toContain(op);
  });

  it('quantizeWeights: a float16 MatMul becomes MatMulNBits over an 8-bit copy within half a step of each value', () => {
    const source = fixture('rewrite-matmul-fp16');
    const result = externalizeWeights(source, 'mm.onnx', { externalizeOnly: true, quantizeWeights: 8 });
    expect(ops(result.graph)).toContain('MatMulNBits');
    const [k, n] = [64, 6];
    const blocks = k / 32;
    const file = result.extraFiles[0]!.source;
    const q = file.read(0, n * k);
    const scaleBytes = file.read(n * k, n * blocks * 2);
    const scales = new Uint16Array(scaleBytes.buffer.slice(scaleBytes.byteOffset, scaleBytes.byteOffset + n * blocks * 2));
    const raw = source.read(findRawTensor(source, k * n * 2), k * n * 2);
    const w = new Uint16Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + k * n * 2));
    const half = (h: number) => {
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      return (h & 0x8000 ? -1 : 1) * (e === 0 ? 2 ** -14 * (f / 1024) : 2 ** (e - 15) * (1 + f / 1024));
    };
    for (let row = 0; row < k; row++)
      for (let col = 0; col < n; col++) {
        const scale = half(scales[col * blocks + Math.floor(row / 32)]!);
        const decoded = (q[col * k + row]! - 128) * scale;
        expect(Math.abs(decoded - half(w[row * n + col]!))).toBeLessThanOrEqual(scale / 2 + 1e-6);
      }
    source.close();
  });

  it('packed MultiHeadAttention inputs are separated', () => {
    const source = fixture('rewrite-packed-mha');
    const text = ops(externalizeWeights(source, 'm.onnx', { externalizeOnly: true }).graph);
    source.close();
    // Three parts of the packed QKV, two of the packed KV.
    expect(text.match(/Gather/g)).toHaveLength(5);
    expect(text).toContain('MultiHeadAttention');
  });
});

// The same rewrites run by ONNX Runtime Web's CPU (WebAssembly) backend, against the original graph where the CPU
// can run it, else a plain implementation here. (Their WebGPU numerics were checked in the browser.)
describe('rewrites compute the same results (ONNX Runtime Web, CPU)', () => {
  ort.env.wasm.numThreads = 1;
  const run = async (graph: Uint8Array, feeds: Record<string, ort.Tensor>, externalData?: { path: string; data: Uint8Array }[]) => {
    const session = await ort.InferenceSession.create(graph, { executionProviders: ['wasm'], externalData });
    const out = await session.run(feeds);
    await session.release();
    return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, numbers(t)]));
  };
  const rewrite = (name: string, options: Parameters<typeof externalizeWeights>[2] = {}) => {
    const source = fixture(name);
    const original = source.read(0, source.size);
    const result = externalizeWeights(source, 'original.onnx', { externalizeOnly: true, ...options });
    return {
      original,
      graph: result.graph,
      data: [{ path: 'original.onnx', data: lazyBytes(bytesSource(original)) }, ...result.extraFiles.map((f) => ({ path: f.path, data: lazyBytes(f.source) }))],
    };
  };
  const random = (n: number, seed: number, scale = 1) => {
    let s = seed;
    return Array.from({ length: n }, () => ((s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31 - 0.5) * 2 * scale);
  };
  const f16 = (values: number[], dims: number[]) => new ort.Tensor('float16', toHalfBits(values), dims);
  const close = (a: number[], b: number[], tolerance: number) => {
    expect(a.length).toBe(b.length);
    const scale = Math.max(1e-6, ...b.map(Math.abs));
    for (let i = 0; i < a.length; i++) expect(Math.abs(a[i]! - b[i]!) / scale).toBeLessThan(tolerance);
  };

  it('int4 DequantizeLinear → MatMul and its MatMulNBits rewrite agree', async () => {
    const r = rewrite('rewrite-dq-int4');
    const feeds = () => ({ x: f16(random(128, 1), [1, 2, 64]) });
    close((await run(r.graph, feeds(), r.data)).y!, (await run(r.original, feeds())).y!, 2e-3);
  });

  it('ONNX Attention and its decomposition agree (grouped heads, mask, past cache)', async () => {
    const r = rewrite('rewrite-attention');
    const mask = [0, 0, 0, 0, -65504, 0, 0, 0, 0, 0];
    const feeds = () => ({
      q: f16(random(128, 2), [1, 2, 64]),
      k: f16(random(64, 3), [1, 2, 32]),
      v: f16(random(64, 4), [1, 2, 32]),
      mask: f16(mask, [1, 1, 2, 5]),
      pk: f16(random(96, 5), [1, 2, 3, 16]),
      pv: f16(random(96, 6), [1, 2, 3, 16]),
    });
    const ours = await run(r.graph, feeds(), r.data);
    const theirs = await run(r.original, feeds());
    for (const name of ['y', 'ok', 'ov']) close(ours[name]!, theirs[name]!, 3e-3);
  });

  it('the float16 MatMul quantized to 8 bits stays within 2% of the original', async () => {
    const r = rewrite('rewrite-matmul-fp16', { quantizeWeights: 8 });
    const feeds = () => ({ x: f16(random(128, 7), [1, 2, 64]) });
    close((await run(r.graph, feeds(), r.data)).y!, (await run(r.original, feeds())).y!, 2e-2);
  });

  it('GroupNorm (NCHW, SiLU) matches a plain implementation', async () => {
    const r = rewrite('rewrite-groupnorm');
    const x = random(128, 8, 40);
    const out = (await run(r.graph, { x: f16(x, [1, 8, 4, 4]) }, r.data)).y!;
    // gamma and beta from the fixture: the two float16 initializers of 8 values.
    const gamma = initializerValues(r.original, 'g');
    const beta = initializerValues(r.original, 'b');
    const xs = numbers(f16(x, [128]));
    const expected: number[] = [];
    for (let c = 0; c < 8; c++) {
      const group = Math.floor(c / 2);
      const members = xs.slice(group * 32, group * 32 + 32);
      const mean = members.reduce((a, b) => a + b, 0) / 32;
      const variance = members.reduce((a, b) => a + (b - mean) ** 2, 0) / 32;
      for (let i = 0; i < 16; i++) {
        const y = ((xs[c * 16 + i]! - mean) / Math.sqrt(variance + 1e-5)) * gamma[c]! + beta[c]!;
        expected.push(y / (1 + Math.exp(-y)));
      }
    }
    close(out, expected, 5e-3);
  });

  it('IsInf after a Cast to float64 runs, as float32', async () => {
    const r = rewrite('rewrite-isinf');
    const y = (await run(r.graph, { x: new ort.Tensor('float16', Uint16Array.of(0x3c00, 0x7c00, 0xfc00, 0x7bff), [1, 4]) }, r.data)).y!;
    expect(y).toEqual([0, 1, 1, 0]);
  });

  it('computeFloat32: the float16 graph computes the same in float32, its inputs and outputs float32', async () => {
    const r = rewrite('rewrite-upcast', { computeFloat32: true });
    const x = random(192, 9);
    const ours = await run(r.graph, { x: new ort.Tensor('float32', Float32Array.from(numbers(f16(x, [192]))), [1, 3, 64]) }, r.data);
    const theirs = await run(r.original, { x: f16(x, [1, 3, 64]) });
    close(ours.y!, theirs.y!, 3e-3);
  });
});

function numbers(t: ort.Tensor): number[] {
  if (t.type === 'bool') return Array.from(t.data as Uint8Array, Number);
  if (t.type !== 'float16') return Array.from(t.data as Float32Array);
  const data = t.data as Uint16Array;
  return Array.from(data instanceof Uint16Array ? data : toHalfBits(Array.from(data as ArrayLike<number>)), halfToNumber);
}

function halfToNumber(h: number): number {
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  const sign = h & 0x8000 ? -1 : 1;
  return e === 0 ? sign * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : sign * Infinity) : sign * 2 ** (e - 15) * (1 + f / 1024);
}

function toHalfBits(values: number[]): Uint16Array {
  const view = new DataView(new ArrayBuffer(4));
  return Uint16Array.from(values, (v) => {
    view.setFloat32(0, v);
    const x = view.getUint32(0);
    const sign = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 112;
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) return sign;
    return sign | (e << 10) | ((x >>> 13) & 0x3ff);
  });
}

/** A small float16 initializer's values, found in a fixture's bytes by its name (raw_data follows the name). */
function initializerValues(model: Uint8Array, name: string): number[] {
  const bytes = Buffer.from(model);
  const at = bytes.indexOf(Buffer.from([0x42, name.length, ...Buffer.from(name)]));
  expect(at).toBeGreaterThan(0);
  const raw = bytes.indexOf(0x4a, at + 2 + name.length);
  const length = bytes[raw + 1]!;
  return Array.from(new Uint16Array(model.slice(raw + 2, raw + 2 + length).buffer), halfToNumber);
}
