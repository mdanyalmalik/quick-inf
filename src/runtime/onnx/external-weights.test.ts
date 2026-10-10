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
        // transformers.js's fused-attention exports have the causal-mask subgraph to replace.
        if (model.source.repo.endsWith('-MHA')) expect(result.attentionBias).toBe(true);
        // Saved beside the original, which serves as its external data file.
        writeFileSync(new URL(`${artifact.graph.path}.external-graph`, dir), result.graph);
      });
    });
  }
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
