// The host's side of segments run on other devices (segmented-session.ts's Stage): each step's values go to the first
// device, which runs its segment and passes the values on to the next device itself, and so on; the last sends back
// what the host needs. Each device keeps its layers' KV cache; the adapter gets a stand-in for each present output
// (an empty tensor) that names the state the device holds, and passes it back as the past input of the next step, so
// the device continues that exact state. Stand-ins the adapter drops are freed on the device once they're garbage
// collected (it also keeps only a few states: see peer-stage.ts).

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import type { ModelSpec } from '../catalogue/models';
import type { GraphSplit, Segment, ValueType } from '../runtime/onnx/moe-split';
import type { Stage, StagePlan } from '../runtime/onnx/segmented-session';
import type { ByteSource } from '../runtime/onnx/external-weights';
import { fromWire, toWire, type AssignFile, type AssignPart, type Channel, type Hop, type PipelineMessage, type WireInput, type WireTensor } from './protocol';

type TensorConstructor = new (type: Tensor.Type, data: Tensor.DataType, dims: readonly number[]) => Tensor;

/** The host's link to its devices: requests and their replies, by device. */
export class PipelineLink {
  private nextId = 1;
  private readonly runs = new Map<number, { resolve: (m: Extract<PipelineMessage, { type: 'result' }>) => void; reject: (e: Error) => void }>();
  private readonly loads = new Map<string, { resolve: (m: Extract<PipelineMessage, { type: 'loaded' }>) => void; reject: (e: Error) => void }>();
  private readonly links = new Map<string, (direct: boolean) => void>();

  constructor(readonly channel: Channel) {
    channel.onMessage((m) => {
      if (m.type === 'result' || m.type === 'run-error') {
        const pending = this.runs.get(m.id);
        if (!pending) return;
        this.runs.delete(m.id);
        if (m.type === 'result') pending.resolve(m);
        else pending.reject(new Error(`${m.peer}: ${m.error}`));
      } else if (m.type === 'linked') {
        this.links.get(`${m.peer}>${m.to}`)?.(m.direct);
        this.links.delete(`${m.peer}>${m.to}`);
      } else if (m.type === 'loaded' || m.type === 'failed') {
        const pending = this.loads.get(m.peer);
        if (!pending) return;
        this.loads.delete(m.peer);
        if (m.type === 'loaded') pending.resolve(m);
        else pending.reject(new Error(`${m.peer} could not load its layers: ${m.error}`));
      }
    });
  }

  load(message: Extract<PipelineMessage, { type: 'assign' }>): Promise<Extract<PipelineMessage, { type: 'loaded' }>> {
    return new Promise((resolve, reject) => {
      this.loads.set(message.peer, { resolve, reject });
      this.channel.send(message);
    });
  }

  /** Sends a run to its first hop's device; resolves with the last one's result. */
  run(hops: Hop[], feeds: Record<string, WireTensor>): Promise<Extract<PipelineMessage, { type: 'result' }>> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.runs.set(id, { resolve, reject });
      this.channel.send({ type: 'run', peer: hops[0]!.peer, id, hops, feeds });
    });
  }

  /** Asks device `from` to link to device `to` (which it passes runs on to); resolves once it has. */
  connect(from: string, to: string): Promise<boolean> {
    return new Promise((resolve) => {
      this.links.set(`${from}>${to}`, resolve);
      this.channel.send({ type: 'link', peer: from, to });
    });
  }
}

export interface RemoteChainOptions {
  link: PipelineLink;
  /** The device running each segment, in order: a run goes through them one after the other. */
  peers: string[];
  model: ModelSpec;
  split: GraphSplit;
  /** The files the split graph names, with where in the model's files their bytes are. */
  sources: Record<string, ByteSource>;
  /** The first segment's number. */
  index: number;
  types: ReadonlyMap<string, ValueType>;
  Tensor: TensorConstructor;
}

/**
 * Consecutive segments on other devices (`peers[i]` runs segment `index + i`) as one stage: assigns each its segment
 * (in order: each one's output types declare the next one's inputs) and links each to the next. A run goes to the
 * first device with every device's part; each passes the values on itself, and the last sends the result back.
 */
export async function createRemoteChain(options: RemoteChainOptions): Promise<StagePlan> {
  const { link, peers, model, split, Tensor } = options;
  const types = new Map(options.types);
  const hops: { peer: string; segment: Segment; pastNames: string[]; presentNames: string[]; inputs: WireInput[] }[] = [];
  const inputMetadata: InferenceSession.ValueMetadata[] = [];
  const outputTypes = new Map<string, ValueType>();
  for (const [i, peer] of peers.entries()) {
    const segment = split.segments[options.index + i]!;
    const loaded = await link.load({
      type: 'assign',
      peer,
      modelId: model.id,
      segment: options.index + i,
      graph: segment.build(types).buffer,
      files: await segmentFiles(segment, options.sources),
    });
    for (const [name, type] of loaded.outputTypes) {
      types.set(name, type);
      outputTypes.set(name, type);
    }
    inputMetadata.push(...loaded.inputs.map((i) => ({ name: i.name, isTensor: true, type: i.type, shape: i.shape }) as InferenceSession.ValueMetadata));
    hops.push({
      peer,
      segment,
      pastNames: loaded.inputs.filter((i) => isState(i.name)).map((i) => i.name),
      presentNames: loaded.outputTypes.map(([n]) => n).filter((n) => n.startsWith('present')),
      inputs: loaded.inputs,
    });
  }
  for (const [i, hop] of hops.entries()) if (hops[i + 1]) await link.connect(hop.peer, hops[i + 1]!.peer);

  // What each hop takes from the bundle, and what the bundle keeps after it: values later hops take, and what the
  // chain hands back: what segments after it take, and the model's outputs (made, and taken by no segment of it).
  const later = new Set(split.segments.slice(options.index + peers.length).flatMap((s) => s.inputs));
  const inside = new Set(hops.flatMap((h) => h.segment.inputs));
  const returned = hops.flatMap((h) => h.segment.outputs.filter((n) => !n.startsWith('present') && (later.has(n) || !inside.has(n))));
  const feed = hops.map((h) => h.segment.inputs.filter((n) => !h.pastNames.includes(n)));
  const carry = hops.map((_, i) => [...new Set([...feed.slice(i + 1).flat(), ...returned])]);
  const holders = new WeakMap<Tensor, string>();
  const freed = new FinalizationRegistry<{ peer: string; state: string }>(({ peer, state }) => link.channel.send({ type: 'free', peer, states: [state] }));
  let runs = 0;
  const stage: Stage = {
    inputMetadata,
    outputTypes,
    async run(feeds) {
      const run = ++runs;
      const plan: Hop[] = hops.map((h, i) => {
        const past = h.pastNames.map((n) => feeds[n]).find((t) => t && holders.has(t));
        // The adapter's own empty cache starts a new state.
        return { peer: h.peer, state: past ? holders.get(past)! : null, next: `${options.index + i}:${run}`, feed: feed[i]!, carry: carry[i]! };
      });
      const sent: Record<string, WireTensor> = {};
      const pasts = new Set(hops.flatMap((h) => h.pastNames));
      for (const [name, tensor] of Object.entries(feeds)) if (!pasts.has(name)) sent[name] = await toWire(tensor);
      const result = await link.run(plan, sent);
      const outputs: Record<string, Tensor> = {};
      for (const [name, wire] of Object.entries(result.outputs)) if (!(name in feeds)) outputs[name] = fromWire(Tensor, wire);
      for (const [i, h] of hops.entries()) {
        for (const name of h.presentNames) {
          const meta = h.inputs.find((input) => input.name === name.replace('present', 'past_key_values'));
          const holder = emptyState(Tensor, meta?.type ?? 'float16', model);
          holders.set(holder, plan[i]!.next);
          outputs[name] = holder;
        }
        if (h.presentNames[0]) freed.register(outputs[h.presentNames[0]]!, { peer: h.peer, state: plan[i]!.next });
      }
      return outputs;
    },
    async release() {
      for (const peer of peers) link.channel.send({ type: 'release', peer });
    },
  };
  return { stage, segments: peers.length };
}
const isState = (name: string) => name.startsWith('past');

/** Neighbouring weights closer than this are sent as one part; a part is at most PART_BYTES. */
const MERGE_GAP = 1024 * 1024;
const PART_BYTES = 64 * 1024 * 1024;

/**
 * The bytes a segment reads, by the files its graph names: ranges of the model's files, each with the SHA-256 of
 * this (verified) copy for the device to check what it fetches against, or the bytes themselves for files made here.
 */
async function segmentFiles(segment: Segment, sources: Record<string, ByteSource>): Promise<AssignFile[]> {
  const byLocation = new Map<string, { offset: number; end: number }[]>();
  for (const e of [...segment.externals].sort((a, b) => a.offset - b.offset)) {
    const ranges = byLocation.get(e.location) ?? [];
    const last = ranges.at(-1);
    if (last && e.offset - last.end <= MERGE_GAP && e.offset + e.length - last.offset <= PART_BYTES) last.end = Math.max(last.end, e.offset + e.length);
    else ranges.push({ offset: e.offset, end: e.offset + e.length });
    byLocation.set(e.location, ranges);
  }
  const files: AssignFile[] = [];
  for (const [location, ranges] of byLocation) {
    const source = sources[location];
    if (!source) throw new Error(`The weights file ${location} isn't open`);
    const parts: AssignPart[] = [];
    for (const { offset, end } of ranges) {
      const bytes = source.read(offset, end - offset);
      parts.push(
        source.origin
          ? { offset, length: bytes.length, from: { path: source.origin.path, offset: source.origin.offset + offset, sha256: await sha256(bytes) } }
          : { offset, length: bytes.length, data: bytes.slice().buffer },
      );
    }
    files.push({ location, size: source.size, parts });
  }
  return files;
}

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** An empty KV tensor, [1, KV heads, 0, head size], standing for a state held on the device. */
function emptyState(Tensor: TensorConstructor, type: string, model: ModelSpec): Tensor {
  const { kvHeads, headDim } = model.architecture;
  return type === 'float16'
    ? new Tensor('float16', new Uint16Array(0), [1, kvHeads, 0, headDim])
    : new Tensor('float32', new Float32Array(0), [1, kvHeads, 0, headDim]);
}
