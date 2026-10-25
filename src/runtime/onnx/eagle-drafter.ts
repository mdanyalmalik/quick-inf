// EAGLE-3 drafting for speculative decoding: a one-layer draft head (tools/export/export_eagle3.py, OnnxArtifact.draft)
// guesses the next tokens from the target model's hidden states, which the target graph outputs as `eagle_taps` (the
// residual stream entering three of its layers, export_dense.py).
//
// A draft row at position t pairs the target's taps at t with the token at t + 1 and guesses the token at t + 2; later
// guesses feed the head's own output hidden state instead of taps. The head keeps its own static KV cache (one layer)
// at the target's positions: rows written for guesses the target rejected are overwritten before anything attends
// to them, as in the target's cache. Guesses stay on the GPU until a chain is done. (Chains aren't graph-captured:
// replayed chains guessed wrongly after their first token, cause not found.)

import type { InferenceSession, Tensor } from 'onnxruntime-web';

type Ort = typeof import('onnxruntime-web');

/** The head's sizes (OnnxArtifact.draft). */
export interface DraftHead {
  hiddenSize: number;
  kvHeads: number;
  headDim: number;
}

const BUFFER_MAP_READ = 0x1;
const BUFFER_COPY_SRC = 0x4;
const BUFFER_COPY_DST = 0x8;
const BUFFER_STORAGE = 0x80;
const MAP_MODE_READ = 0x1;

/**
 * The head's session options from the model's: int64 kernels on the GPU (the guess's argmax and vocabulary mapping
 * would otherwise run on the CPU, a GPU round trip in every run) and size-bucketed buffer reuse (its temporaries are
 * small).
 */
export function draftSessionOptions(options: InferenceSession.SessionOptions): InferenceSession.SessionOptions {
  const providers = options.executionProviders?.map((p) =>
    typeof p === 'object' && p.name === 'webgpu'
      ? { ...p, storageBufferCacheMode: 'bucket', qiOptions: { ...(p as { qiOptions?: Record<string, string> }).qiOptions, enableInt64: '1' } }
      : p,
  );
  return { ...options, ...(providers && { executionProviders: providers as InferenceSession.ExecutionProviderConfig[] }) };
}

export class EagleDrafter {
  private readonly buffers: GPUBuffer[] = [];
  /** Per guess in a chain: its token and output hidden state. */
  private readonly tokens: Tensor[];
  private readonly hidden: Tensor[];
  private readonly noTaps: Tensor;
  private readonly noHidden: Tensor;
  private readonly staging: GPUBuffer;
  private cache: { key: Tensor; value: Tensor; buffers: GPUBuffer[] } | null = null;
  /** Positions the head's cache holds (0 until allocated). */
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
    this.noTaps = tensor('float16', [1, 0, 3 * H], 0);
    this.noHidden = tensor('float16', [1, 0, H], 0);
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
   * `count` guesses (as made: 0 writes the rows to the cache only, for a prompt's earlier chunks) after a target pass
   * whose rows (from position `start`) gave `taps`; `ids[i]` is the token at start + i + 1 (rows past the sequence
   * repeat its last token), and row `last` pairs with the newest token.
   */
  async guess(taps: Tensor, ids: number[], start: number, last: number, count: number): Promise<number[]> {
    const ort = this.ort;
    const rows = ids.length;
    if (start + rows > this.maxLength) return [];
    count = Math.min(count, this.count, this.maxLength - start - last);
    // Lengths and rows come from the CPU, uploaded in order with ONNX Runtime's own work.
    const inputs = (row: number, seqlen: number) => ({
      last: new ort.Tensor('int64', BigInt64Array.of(BigInt(row)), [1]),
      seqlens_k: new ort.Tensor('int32', Int32Array.of(seqlen), [1]),
      total_sequence_length: new ort.Tensor('int32', Int32Array.of(seqlen + 1), []),
    });
    const input_ids = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, rows]);
    let t = performance.now();
    await this.run({ input_ids, taps, hidden: this.noHidden, ...inputs(last, start + rows - 1) }, 0);
    t = mark('draftFirst', t);
    if (count <= 0) return [];
    // Chained guesses: the row after `last` pairs the head's own output with its guess.
    for (let j = 1; j < count; j++) {
      await this.run({ input_ids: this.tokens[j - 1]!, taps: this.noTaps, hidden: this.hidden[j - 1]!, ...inputs(0, start + last + j) }, j);
      t = mark('draftNext', t);
    }
    const encoder = this.device.createCommandEncoder();
    for (let j = 0; j < count; j++) encoder.copyBufferToBuffer(this.tokens[j]!.gpuBuffer as GPUBuffer, 0, this.staging, j * 8, 8);
    this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(MAP_MODE_READ, 0, count * 8);
    const values = Array.from(new BigInt64Array(this.staging.getMappedRange(0, count * 8).slice(0)), Number);
    this.staging.unmap();
    mark('draftRead', t);
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

/** Development timing (read by the bench page, as the adapter's phases). */
function mark(phase: string, start: number): number {
  if (!import.meta.env.DEV) return 0;
  const now = performance.now();
  const profile = (globalThis as { __quickInfProfile?: Record<string, [number, number]> }).__quickInfProfile;
  if (profile) {
    const entry = (profile[phase] ??= [0, 0]);
    entry[0]++;
    entry[1] += now - start;
  }
  return now;
}
