// Runs a model split at layer boundaries (moe-split.ts: splitLayers) as one session per segment, in order, on this
// device: the first step towards running the segments on several devices (info/distributed-plan.md). Each segment
// keeps its own layers' KV inputs and outputs; values crossing a cut (the residual stream) stay on the GPU here and are
// what would travel between devices.
//
// To the adapter this looks like one InferenceSession (inputMetadata, run, release), as moe-stream.ts's does.

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { elemTypeOf, type GraphSplit, type ValueType } from './moe-split';

export interface SegmentStats {
  runs: number;
  /** Time in each segment's runs, milliseconds. */
  segmentMs: number[];
  /** Bytes passed into each segment from earlier ones, over all runs (what would cross the network). */
  passedBytes: number[];
}

/** Creates the segments' sessions in order (each one's output types declare the next one's inputs). */
export async function createSegmentedSession(
  split: GraphSplit,
  createSession: (model: Uint8Array<ArrayBuffer>) => Promise<InferenceSession>,
): Promise<SegmentedSession> {
  const types = new Map<string, ValueType>();
  const sessions: InferenceSession[] = [];
  try {
    for (const segment of split.segments) {
      const session = await createSession(segment.build(types));
      sessions.push(session);
      for (const meta of session.outputMetadata) if (meta.isTensor) types.set(meta.name, { elem: elemTypeOf(meta.type), shape: meta.shape });
    }
  } catch (err) {
    for (const session of sessions) await session.release().catch(() => {});
    throw err;
  }
  const session = new SegmentedSession(split, sessions);
  // Development: the statistics, for tests (read in the worker).
  if (import.meta.env.DEV) (globalThis as { __quickInfSegments?: SegmentedSession }).__quickInfSegments = session;
  return session;
}

export class SegmentedSession {
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];
  readonly stats: SegmentStats;
  /** For each value passed between segments: the last segment that reads it. */
  private readonly lastUse = new Map<string, number>();

  constructor(
    private readonly split: GraphSplit,
    private readonly sessions: InferenceSession[],
  ) {
    const seen = new Set<string>();
    this.inputMetadata = sessions.flatMap((s) => s.inputMetadata.filter((m) => split.modelInputs.includes(m.name) && !seen.has(m.name) && seen.add(m.name)));
    split.segments.forEach((segment, s) => segment.passedIn.forEach((n) => this.lastUse.set(n, Math.max(this.lastUse.get(n) ?? 0, s))));
    this.stats = { runs: 0, segmentMs: sessions.map(() => 0), passedBytes: sessions.map(() => 0) };
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const values = new Map<string, Tensor>(Object.entries(feeds));
    const passed = new Set<string>();
    const outputs: Record<string, Tensor> = {};
    try {
      for (const [s, segment] of this.split.segments.entries()) {
        for (const n of segment.passedIn) this.stats.passedBytes[s]! += byteSize(values.get(n)!);
        const start = performance.now();
        const result = await this.sessions[s]!.run(Object.fromEntries(segment.inputs.map((n) => [n, values.get(n)!])));
        this.stats.segmentMs[s]! += performance.now() - start;
        for (const [name, value] of Object.entries(result)) {
          if (this.lastUse.has(name)) {
            values.set(name, value as Tensor);
            passed.add(name);
          } else outputs[name] = value as Tensor;
        }
        for (const name of passed) {
          if (this.lastUse.get(name)! <= s) {
            values.get(name)?.dispose();
            values.delete(name);
            passed.delete(name);
          }
        }
      }
    } catch (err) {
      for (const name of passed) values.get(name)?.dispose();
      for (const value of Object.values(outputs)) value.dispose();
      throw err;
    }
    this.stats.runs++;
    return outputs;
  }

  async release(): Promise<void> {
    for (const session of this.sessions) await session.release().catch(() => {});
  }
}

const BYTES: Record<string, number> = { float32: 4, float16: 2, int64: 8, int32: 4, uint8: 1, int8: 1, bool: 1 };

function byteSize(tensor: Tensor): number {
  return tensor.size * (BYTES[tensor.type] ?? 4);
}
