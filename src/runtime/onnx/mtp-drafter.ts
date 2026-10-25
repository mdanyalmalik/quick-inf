// Multi-token prediction (MTP) drafting for speculative decoding: the draft layer a Qwen3.5-family checkpoint ships
// (tools/export/export_mtp.py, OnnxArtifact.draft with kind 'mtp') guesses the next tokens from the model's final hidden
// states, which the model's graph outputs as HIDDEN_OUTPUT (external-weights.ts: exposeHidden).
//
// A draft row at position t pairs the model's hidden state at t with the token at t + 1 (the graph looks it up in a
// copy of the model's embedding table) and guesses the token at t + 2; later guesses in a chain feed the layer's own
// guess and output hidden state (graph outputs, so a chain stays on the GPU). The layer keeps its own static KV cache at
// the model's positions: rows written for guesses the model rejected are overwritten before anything attends to them,
// as in the model's cache.

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import type { DraftHead } from './eagle-drafter';

type Ort = typeof import('onnxruntime-web');

const BUFFER_MAP_READ = 0x1;
const BUFFER_COPY_SRC = 0x4;
const BUFFER_COPY_DST = 0x8;
const BUFFER_STORAGE = 0x80;
const MAP_MODE_READ = 0x1;

export class MtpDrafter {
  private readonly buffers: GPUBuffer[] = [];
  /** Per guess in a chain: its token and the layer's output hidden state. */
  private readonly tokens: Tensor[];
  private readonly hidden: Tensor[];
  private readonly staging: GPUBuffer;
  private cache: { key: Tensor; value: Tensor; buffers: GPUBuffer[] } | null = null;
  /** Positions the layer's cache holds (0 until allocated). */
  maxLength = 0;

  constructor(
    private readonly ort: Ort,
    private readonly device: GPUDevice,
    private readonly session: InferenceSession,
    private readonly head: DraftHead,
    /** Most guesses per chain. */
    private readonly count: number,
  ) {
    const H = head.hiddenSize;
    const tensor = (dataType: 'float16' | 'int64', dims: number[], bytes: number) => {
      const buffer = this.buffer(bytes);
      this.buffers.push(buffer);
      return ort.Tensor.fromGpuBuffer(buffer, { dataType, dims });
    };
    this.tokens = Array.from({ length: count }, () => tensor('int64', [1, 1], 8));
    this.hidden = Array.from({ length: count }, () => tensor('float16', [1, 1, H], H * 2));
    this.staging = device.createBuffer({ size: Math.max(16, count * 8), usage: BUFFER_MAP_READ | BUFFER_COPY_DST });
  }

  /** A cache for `maxLength` positions (its contents are lost). */
  allocate(maxLength: number): void {
    this.freeCache();
    const dims = [1, this.head.kvHeads, maxLength, this.head.headDim];
    const bytes = dims.reduce((a, b) => a * b, 1) * 2;
    const key = this.buffer(bytes);
    const value = this.buffer(bytes);
    this.cache = {
      key: this.ort.Tensor.fromGpuBuffer(key, { dataType: 'float16', dims }),
      value: this.ort.Tensor.fromGpuBuffer(value, { dataType: 'float16', dims }),
      buffers: [key, value],
    };
    this.maxLength = maxLength;
  }

  /**
   * `count` guesses (0: only write the rows to the cache, for a prompt's earlier chunks) after a model pass whose rows
   * (from position `start`) gave the hidden states `hidden`; `ids[i]` is the token at start + i + 1 (rows past the
   * sequence repeat its last token), and row `last` pairs with the newest token.
   */
  async guess(hidden: Tensor, ids: number[], start: number, last: number, count: number): Promise<number[]> {
    const ort = this.ort;
    const rows = ids.length;
    if (start + rows > this.maxLength) return [];
    count = Math.min(count, this.count, this.maxLength - start - last);
    const lengths = (row: number, from: number, n: number) => ({
      last: new ort.Tensor('int64', BigInt64Array.of(BigInt(row)), [1]),
      position_ids: new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, i) => BigInt(from + i)), [1, n]),
      seqlens_k: new ort.Tensor('int32', Int32Array.of(from + n - 1), [1]),
      total_sequence_length: new ort.Tensor('int32', Int32Array.of(from + n), []),
    });
    const input_ids = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, rows]);
    await this.run({ input_ids, hidden, ...lengths(last, start, rows) }, 0);
    if (count <= 0) return [];
    for (let j = 1; j < count; j++) {
      await this.run({ input_ids: this.tokens[j - 1]!, hidden: this.hidden[j - 1]!, ...lengths(0, start + last + j, 1) }, j);
    }
    const encoder = this.device.createCommandEncoder();
    for (let j = 0; j < count; j++) encoder.copyBufferToBuffer(this.tokens[j]!.gpuBuffer as GPUBuffer, 0, this.staging, j * 8, 8);
    this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(MAP_MODE_READ, 0, count * 8);
    const values = Array.from(new BigInt64Array(this.staging.getMappedRange(0, count * 8).slice(0)), Number);
    this.staging.unmap();
    return values;
  }

  private async run(feeds: Record<string, Tensor>, slot: number): Promise<void> {
    const { key, value } = this.cache!;
    await this.session.run(
      { ...feeds, 'past_key_values.0.key': key, 'past_key_values.0.value': value },
      { token: this.tokens[slot]!, hidden_out: this.hidden[slot]!, 'present.0.key': key, 'present.0.value': value },
    );
  }

  private buffer(bytes: number): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: BUFFER_STORAGE | BUFFER_COPY_SRC | BUFFER_COPY_DST });
  }

  private freeCache(): void {
    for (const buffer of this.cache?.buffers ?? []) buffer.destroy();
    this.cache = null;
    this.maxLength = 0;
  }

  /** Frees the GPU buffers (the session belongs to the caller). */
  destroy(): void {
    this.freeCache();
    for (const buffer of this.buffers) buffer.destroy();
    this.staging.destroy();
  }
}
