// Runs a model split at layer boundaries (moe-split.ts: splitLayers) as stages, in order: an ONNX Runtime session on
// this device for a segment, or other devices running one or more consecutive segments, passing values from one to
// the next themselves (src/distributed/, info/distributed-plan.md). Values crossing a cut (the residual stream) are
// what travels between devices.
//
// A remote stage keeps its layers' KV cache on its own device. The adapter still sees every layer's state: a remote
// stage lists its past inputs in its inputMetadata and returns its present outputs, as small stand-ins for the state it
// holds (src/distributed/remote-stage.ts), so the adapter's cache handling (reuse, cut-back, disposal) stays unchanged.
//
// To the adapter this looks like one InferenceSession (inputMetadata, run, release), as moe-stream.ts's does.

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { elemTypeOf, type GraphSplit, type ValueType } from './moe-split';

/** The runner of one segment, or of several consecutive ones. */
export interface Stage {
  /** The model inputs it takes (its graph inputs: cache states and the like), as ONNX Runtime describes them. */
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];
  /** Its outputs' types, for declaring later segments' inputs. */
  readonly outputTypes: ReadonlyMap<string, ValueType>;
  run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  release(): Promise<void>;
}

/** A stage and how many segments, from the one it was made for, it runs. */
export interface StagePlan {
  stage: Stage;
  segments: number;
}

/** Makes the stage starting at segment `index`; `types` holds the types of every value made before it. */
export type StageFactory = (index: number, types: ReadonlyMap<string, ValueType>) => Promise<StagePlan>;

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
  /** Time in each stage's runs, milliseconds. */
  segmentMs: number[];
  /** Bytes passed into each stage from earlier ones, over all runs (what would cross the network). */
  passedBytes: number[];
}

/** A stage with the segments [from, to) it runs, the values it takes, and those it hands on. */
interface Group {
  stage: Stage;
  from: number;
  to: number;
  inputs: string[];
  passedIn: string[];
}

/** Creates the stages in order (each one's output types declare the next one's inputs). */
export async function createSegmentedSession(split: GraphSplit, createStage: StageFactory): Promise<SegmentedSession> {
  const types = new Map<string, ValueType>();
  const groups: Group[] = [];
  try {
    for (let from = 0; from < split.segments.length; ) {
      const { stage, segments } = await createStage(from, types);
      const to = Math.min(split.segments.length, from + Math.max(1, segments));
      // Its inputs: what its segments take that none of them makes before.
      const made = new Set<string>();
      const inputs = new Set<string>();
      for (const segment of split.segments.slice(from, to)) {
        for (const n of segment.inputs) if (!made.has(n)) inputs.add(n);
        for (const n of segment.outputs) made.add(n);
      }
      groups.push({ stage, from, to, inputs: [...inputs], passedIn: [...inputs].filter((n) => !split.modelInputs.includes(n)) });
      for (const [name, type] of stage.outputTypes) types.set(name, type);
      from = to;
    }
  } catch (err) {
    for (const { stage } of groups) await stage.release().catch(() => {});
    throw err;
  }
  const session = new SegmentedSession(split, groups);
  // Development: the statistics, for tests (read in the worker).
  if (import.meta.env.DEV) (globalThis as { __quickInfSegments?: SegmentedSession }).__quickInfSegments = session;
  return session;
}

export class SegmentedSession {
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];
  readonly stats: SegmentStats;
  /** For each value passed between stages: the last stage that reads it. */
  private readonly lastUse = new Map<string, number>();

  constructor(
    split: GraphSplit,
    private readonly groups: Group[],
  ) {
    const seen = new Set<string>();
    this.inputMetadata = groups.flatMap((g) => g.stage.inputMetadata.filter((m) => split.modelInputs.includes(m.name) && !seen.has(m.name) && seen.add(m.name)));
    groups.forEach((g, i) => g.passedIn.forEach((n) => this.lastUse.set(n, Math.max(this.lastUse.get(n) ?? 0, i))));
    this.stats = { runs: 0, segmentMs: groups.map(() => 0), passedBytes: groups.map(() => 0) };
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const values = new Map<string, Tensor>(Object.entries(feeds));
    const passed = new Set<string>();
    const outputs: Record<string, Tensor> = {};
    try {
      for (const [s, group] of this.groups.entries()) {
        for (const n of group.passedIn) this.stats.passedBytes[s]! += byteSize(values.get(n)!);
        const start = performance.now();
        const result = await group.stage.run(Object.fromEntries(group.inputs.map((n) => [n, values.get(n)!])));
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
    for (const { stage } of this.groups) await stage.release().catch(() => {});
  }
}

const BYTES: Record<string, number> = { float32: 4, float16: 2, int64: 8, int32: 4, uint8: 1, int8: 1, bool: 1 };

function byteSize(tensor: Tensor): number {
  return tensor.size * (BYTES[tensor.type] ?? 4);
}
