// Runs a model split at layer boundaries (moe-split.ts: splitLayers) as one stage per segment, in order: an ONNX
// Runtime session on this device, or another device running that segment (src/distributed/, info/distributed-plan.md).
// Values crossing a cut (the residual stream) are what travels between devices.
//
// A remote stage keeps its layers' KV cache on its own device. The adapter still sees every layer's state: a remote
// stage lists its past inputs in its inputMetadata and returns its present outputs, as small stand-ins for the state it
// holds (src/distributed/remote-stage.ts), so the adapter's cache handling (reuse, cut-back, disposal) stays unchanged.
//
// To the adapter this looks like one InferenceSession (inputMetadata, run, release), as moe-stream.ts's does.

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { elemTypeOf, type GraphSplit, type Segment, type ValueType } from './moe-split';

/** One segment's runner. */
export interface Stage {
  /** The model inputs it takes (its graph inputs: cache states and the like), as ONNX Runtime describes them. */
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];
  /** Its outputs' types, for declaring later segments' inputs. */
  readonly outputTypes: ReadonlyMap<string, ValueType>;
  run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  release(): Promise<void>;
}

/** Makes segment `index`'s stage; `types` holds the types of every value made before it. */
export type StageFactory = (segment: Segment, index: number, types: ReadonlyMap<string, ValueType>) => Promise<Stage>;

/** A stage running a session on this device. */
export function localStage(session: InferenceSession): Stage {
  const outputTypes = new Map<string, ValueType>();
  for (const meta of session.outputMetadata) if (meta.isTensor) outputTypes.set(meta.name, { elem: elemTypeOf(meta.type), shape: meta.shape });
  return {
    inputMetadata: session.inputMetadata,
    outputTypes,
    run: (feeds) => session.run(feeds) as Promise<Record<string, Tensor>>,
    release: () => session.release(),
  };
}

export interface SegmentStats {
  runs: number;
  /** Time in each segment's runs, milliseconds. */
  segmentMs: number[];
  /** Bytes passed into each segment from earlier ones, over all runs (what would cross the network). */
  passedBytes: number[];
}

/** Creates the stages in order (each one's output types declare the next one's inputs). */
export async function createSegmentedSession(split: GraphSplit, createStage: StageFactory): Promise<SegmentedSession> {
  const types = new Map<string, ValueType>();
  const stages: Stage[] = [];
  try {
    for (const [s, segment] of split.segments.entries()) {
      const stage = await createStage(segment, s, types);
      stages.push(stage);
      for (const [name, type] of stage.outputTypes) types.set(name, type);
    }
  } catch (err) {
    for (const stage of stages) await stage.release().catch(() => {});
    throw err;
  }
  const session = new SegmentedSession(split, stages);
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
    private readonly stages: Stage[],
  ) {
    const seen = new Set<string>();
    this.inputMetadata = stages.flatMap((s) => s.inputMetadata.filter((m) => split.modelInputs.includes(m.name) && !seen.has(m.name) && seen.add(m.name)));
    split.segments.forEach((segment, s) => segment.passedIn.forEach((n) => this.lastUse.set(n, Math.max(this.lastUse.get(n) ?? 0, s))));
    this.stats = { runs: 0, segmentMs: stages.map(() => 0), passedBytes: stages.map(() => 0) };
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const values = new Map<string, Tensor>(Object.entries(feeds));
    const passed = new Set<string>();
    const outputs: Record<string, Tensor> = {};
    try {
      for (const [s, segment] of this.split.segments.entries()) {
        for (const n of segment.passedIn) this.stats.passedBytes[s]! += byteSize(values.get(n)!);
        const start = performance.now();
        const result = await this.stages[s]!.run(Object.fromEntries(segment.inputs.map((n) => [n, values.get(n)!])));
        this.stats.segmentMs[s]! += performance.now() - start;
        for (const [name, value] of Object.entries(result)) {
          if (this.lastUse.has(name)) {
            values.set(name, value);
            passed.add(name);
          } else outputs[name] = value;
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
    for (const stage of this.stages) await stage.release().catch(() => {});
  }
}

const BYTES: Record<string, number> = { float32: 4, float16: 2, int64: 8, int32: 4, uint8: 1, int8: 1, bool: 1 };

function byteSize(tensor: Tensor): number {
  return tensor.size * (BYTES[tensor.type] ?? 4);
}
