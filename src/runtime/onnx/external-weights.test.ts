// Rewrites each cached catalogue model with externalizeWeights and checks the
// result. tools/reference/check_external.py then loads the rewritten structure
// with ONNX Runtime (CPU) and confirms its outputs equal the original model's.

import { closeSync, existsSync, fstatSync, openSync, readSync, writeFileSync } from 'node:fs';
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
