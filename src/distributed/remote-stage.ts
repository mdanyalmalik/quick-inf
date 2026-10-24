// The host's side of segments run on other devices (segmented-session.ts's Stage): each step's values go to the first
// device, which runs its segment and passes the values on to the next device itself, and so on; the last sends back
// what the host needs. Each device keeps its layers' KV cache; the adapter gets a stand-in for each present output
// (an empty tensor) that names the state the device holds, and passes it back as the past input of the next step, so
// the device continues that exact state. Stand-ins the adapter drops are freed on the device once they're garbage
// collected (it also keeps only a few states: see peer-stage.ts).

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import type { ModelSpec } from '../catalogue/models';
import { RuntimeError } from '../runtime/contract';
import type { GraphSplit, Segment, ValueType } from '../runtime/onnx/moe-split';
import { heldStates, type Stage, type StagePlan } from '../runtime/onnx/segmented-session';
import type { ByteSource } from '../runtime/onnx/external-weights';
import { emptyCacheDims, fromWire, isKvState, pastOf, toWire, type AssignFile, type AssignPart, type Channel, type Hop, type PipelineMessage, type WireInput, type WireTensor } from './protocol';

type TensorConstructor = new (type: Tensor.Type, data: Tensor.DataType, dims: readonly number[]) => Tensor;

/** A device the host was using went away (closed, lost its connection). */
export class DeviceGoneError extends RuntimeError {
  constructor(readonly peer: string) {
    super('device-disconnected', `A device running part of the model disconnected (${peer})`, true);
  }
}

/** How long a step may take on the devices before the one it's waiting on counts as gone (a phone gone to sleep). */
const RUN_TIMEOUT_MS = 120_000;

/**
 * The host's link to its devices: requests and their replies, by device. A device that says `bye` (its channel
 * closed) fails what is waiting on it, and anything sent its way after; so does a step that gets no answer in
 * RUN_TIMEOUT_MS.
 */
export class PipelineLink {
  private nextId = 1;
  private readonly runs = new Map<number, { peers: string[]; resolve: (m: Extract<PipelineMessage, { type: 'result' }>) => void; reject: (e: Error) => void }>();
  private readonly loads = new Map<string, { resolve: (m: Extract<PipelineMessage, { type: 'loaded' }>) => void; reject: (e: Error) => void }>();
  private readonly links = new Map<string, { resolve: (direct: boolean) => void; reject: (e: Error) => void }>();
  private readonly needs = new Map<string, (parts: string[]) => void>();
  private readonly acks = new Map<string, () => void>();
  private readonly gone = new Set<string>();

  constructor(readonly channel: Channel) {
    channel.onMessage((m) => {
      if (m.type === 'result' || m.type === 'run-error') {
        const pending = this.runs.get(m.id);
        if (!pending) return;
        this.runs.delete(m.id);
        if (m.type === 'result') pending.resolve(m);
        else pending.reject(new Error(`${m.peer}: ${m.error}`));
      } else if (m.type === 'linked') {
        this.links.get(`${m.peer}>${m.to}`)?.resolve(m.direct);
        this.links.delete(`${m.peer}>${m.to}`);
      } else if (m.type === 'loaded' || m.type === 'failed') {
        const pending = this.loads.get(m.peer);
        if (!pending) return;
        this.loads.delete(m.peer);
        if (m.type === 'loaded') pending.resolve(m);
        else pending.reject(new Error(`${m.peer} could not load its layers: ${m.error}`));
      } else if (m.type === 'need') this.needs.get(m.peer)?.(m.parts);
      else if (m.type === 'got') {
        this.acks.get(`${m.peer}:${m.sha256}:${m.end}`)?.();
        this.acks.delete(`${m.peer}:${m.sha256}:${m.end}`);
      } else if (m.type === 'bye') this.lost(m.peer);
    });
  }

  /** Assigns a device its segment; `onNeed` sends the made parts it asks for. Resolves once it has loaded. */
  load(message: Extract<PipelineMessage, { type: 'assign' }>, onNeed?: (parts: string[]) => Promise<void>): Promise<Extract<PipelineMessage, { type: 'loaded' }>> {
    if (this.gone.has(message.peer)) return Promise.reject(new DeviceGoneError(message.peer));
    return new Promise((resolve, reject) => {
      const done = (f: () => void) => {
        this.needs.delete(message.peer);
        f();
      };
      this.loads.set(message.peer, { resolve: (m) => done(() => resolve(m)), reject: (e) => done(() => reject(e)) });
      if (onNeed) this.needs.set(message.peer, (parts) => void onNeed(parts).catch(reject));
      this.channel.send(message);
    });
  }

  /** Sends made parts to a device in pieces of PIECE_BYTES, each after the last was acknowledged (memory stays flat). */
  async send(peer: string, hashes: string[], made: MadeParts): Promise<void> {
    for (const sha256 of hashes) {
      const part = made.get(sha256);
      if (!part) throw new Error(`${peer} asked for a part it wasn't given`);
      // Read whole, as it was hashed (made files give the right bytes for their tensors' ranges, not any slice).
      const bytes = part.source.read(part.offset, part.length);
      for (let at = 0; at < part.length; at += PIECE_BYTES) {
        if (this.gone.has(peer)) throw new DeviceGoneError(peer);
        const n = Math.min(PIECE_BYTES, part.length - at);
        const data = bytes.subarray(at, at + n);
        const acked = new Promise<void>((resolve) => this.acks.set(`${peer}:${sha256}:${at + n}`, resolve));
        this.channel.send({ type: 'part', peer, sha256, offset: at, data: data.slice().buffer });
        await acked;
      }
    }
  }

  /** Sends a run to its first hop's device; resolves with the last one's result. */
  run(hops: Hop[], feeds: Record<string, WireTensor>): Promise<Extract<PipelineMessage, { type: 'result' }>> {
    const peers = hops.map((h) => h.peer);
    const missing = peers.find((p) => this.gone.has(p));
    if (missing) return Promise.reject(new DeviceGoneError(missing));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.runs.has(id)) return;
        this.runs.delete(id);
        reject(new DeviceGoneError(peers.join(', ')));
      }, RUN_TIMEOUT_MS);
      const settle = <T,>(f: (v: T) => void) => (v: T) => {
        clearTimeout(timer);
        f(v);
      };
      this.runs.set(id, { peers, resolve: settle(resolve), reject: settle(reject) });
      this.channel.send({ type: 'run', peer: peers[0]!, id, hops, feeds });
    });
  }

  /** Asks device `from` to link to device `to` (which it passes runs on to); resolves once it has. */
  connect(from: string, to: string): Promise<boolean> {
    const missing = [from, to].find((p) => this.gone.has(p));
    if (missing) return Promise.reject(new DeviceGoneError(missing));
    return new Promise((resolve, reject) => {
      this.links.set(`${from}>${to}`, { resolve, reject });
      this.channel.send({ type: 'link', peer: from, to });
    });
  }

  private lost(peer: string): void {
    this.gone.add(peer);
    const error = new DeviceGoneError(peer);
    for (const [id, run] of this.runs) {
      if (!run.peers.includes(peer)) continue;
      this.runs.delete(id);
      run.reject(error);
    }
    this.loads.get(peer)?.reject(error);
    this.loads.delete(peer);
    // Pieces waiting for its acknowledgement: the sending loop sees it's gone.
    for (const [key, ack] of this.acks) {
      if (!key.startsWith(`${peer}:`)) continue;
      this.acks.delete(key);
      ack();
    }
    for (const [key, link] of this.links) {
      if (!key.split('>').includes(peer)) continue;
      this.links.delete(key);
      link.reject(error);
    }
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
  /** Layers [from, to) of each device, in order (shown on the device). */
  layers: [number, number][];
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
    const made: MadeParts = new Map();
    const files = await segmentFiles(segment, options.sources, made);
    const loaded = await link.load(
      { type: 'assign', peer, modelId: model.id, segment: options.index + i, layers: options.layers[i]!, graph: segment.build(types).buffer, files },
      (hashes) => link.send(peer, hashes, made),
    );
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
  // A cache output that later segments read too (Gemma 4 shares some layers' caches) travels as data, and comes back
  // in as data (not a state the device holds) on the next step.
  const returned = hops.flatMap((h) => h.segment.outputs.filter((n) => (n.startsWith('present') ? later.has(n) : later.has(n) || !inside.has(n))));
  const carried = new Set(hops.flatMap((h) => h.presentNames.filter((n) => later.has(n) || inside.has(n))));
  const feed = hops.map((h) => h.segment.inputs.filter((n) => !h.pastNames.includes(n)));
  const carry = hops.map((_, i) => [...new Set([...feed.slice(i + 1).flat(), ...returned])]);
  const holders = new WeakMap<Tensor, string>();
  // A stand-in the adapter never released (an error on the way) lets its state go once it's collected.
  const collected = new FinalizationRegistry<() => void>((release) => release());
  let runs = 0;
  const stage: Stage = {
    inputMetadata,
    outputTypes,
    async run(feeds) {
      const run = ++runs;
      const plan: Hop[] = hops.map((h, i) => {
        const past = h.pastNames.map((n) => feeds[n]).find((t) => t && holders.has(t));
        // KV caches that come in as data (carried ones, once they hold positions) go with the step; the adapter's own
        // fixed-size states (zeros, before the first step) don't: the device starts its own.
        const given = h.pastNames.filter((n) => isKvState(n) && feeds[n] && !holders.has(feeds[n]) && feeds[n].dims[2]! > 0);
        // The adapter's own empty cache starts a new state.
        return { peer: h.peer, state: past ? holders.get(past)! : null, next: `${options.index + i}:${run}`, feed: [...feed[i]!, ...given], carry: carry[i]! };
      });
      // Caches given to a later hop go on through the earlier ones.
      plan.forEach((hop, i) => (hop.carry = [...new Set([...hop.carry, ...plan.slice(i + 1).flatMap((h) => h.feed)])]));
      const sent: Record<string, WireTensor> = {};
      const given = new Set(plan.flatMap((h) => h.feed));
      const pasts = new Set(hops.flatMap((h) => h.pastNames));
      for (const [name, tensor] of Object.entries(feeds)) if (!pasts.has(name) || given.has(name)) sent[name] = await toWire(tensor);
      const result = await link.run(plan, sent);
      const outputs: Record<string, Tensor> = {};
      for (const [name, wire] of Object.entries(result.outputs)) if (!(name in feeds)) outputs[name] = fromWire(Tensor, wire);
      for (const [i, h] of hops.entries()) {
        const held = h.presentNames.filter((n) => !carried.has(n));
        if (!held.length) continue;
        // The device keeps this step's state until the last stand-in for it is released (copies of the adapter's
        // cache, as a checkpoint, are more stand-ins for the same state: the device never changes one in place).
        const { peer } = h;
        const state = plan[i]!.next;
        let refs = 0;
        const hold = (input: WireInput | undefined): Tensor => {
          const holder = emptyState(Tensor, input, model);
          holders.set(holder, state);
          refs++;
          let released = false;
          // (The callback mustn't reach the stand-in itself, or it could never be collected.)
          const token = {};
          const release = () => {
            if (released) return;
            released = true;
            collected.unregister(token);
            if (--refs === 0) link.channel.send({ type: 'free', peer, states: [state] });
          };
          heldStates.set(holder, { clone: () => hold(input), release });
          collected.register(holder, release, token);
          return holder;
        };
        for (const name of held) outputs[name] = hold(h.inputs.find((input) => input.name === pastOf(name)));
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

/** Made parts go to a device in pieces of this size. */
const PIECE_BYTES = 8 * 1024 * 1024;

/** Neighbouring weights closer than this are sent as one part; a part is at most PART_BYTES. */
const MERGE_GAP = 1024 * 1024;
const PART_BYTES = 64 * 1024 * 1024;

/**
 * The bytes a segment reads, by the files its graph names: ranges of the model's files, each with the SHA-256 of
 * this (verified) copy for the device to check what it fetches against, or (for files made here as the model loaded)
 * the hash of bytes the device asks for if it hasn't kept them (`made` says where to read them).
 */
/** Where the host reads a made part from, by its hash. */
type MadeParts = Map<string, { source: ByteSource; offset: number; length: number }>;

async function segmentFiles(segment: Segment, sources: Record<string, ByteSource>, made: MadeParts): Promise<AssignFile[]> {
  // Neighbouring ranges of a model file are merged; made files go tensor by tensor (they give the right bytes for a
  // tensor's own range, as ONNX Runtime reads them, not for a span across several).
  const byLocation = new Map<string, { offset: number; end: number }[]>();
  for (const e of [...segment.externals].sort((a, b) => a.offset - b.offset)) {
    const ranges = byLocation.get(e.location) ?? [];
    const last = ranges.at(-1);
    const merge = sources[e.location]?.origin && last && e.offset - last.end <= MERGE_GAP && e.offset + e.length - last.offset <= PART_BYTES;
    if (merge) last.end = Math.max(last.end, e.offset + e.length);
    else if (!(last && last.offset === e.offset && last.end === e.offset + e.length)) ranges.push({ offset: e.offset, end: e.offset + e.length });
    byLocation.set(e.location, ranges);
  }
  const files: AssignFile[] = [];
  for (const [location, ranges] of byLocation) {
    const source = sources[location];
    if (!source) throw new Error(`The weights file ${location} isn't open`);
    const parts: AssignPart[] = [];
    for (const { offset, end } of ranges) {
      const bytes = source.read(offset, end - offset);
      const hash = await sha256(bytes);
      if (source.origin) parts.push({ offset, length: bytes.length, from: { path: source.origin.path, offset: source.origin.offset + offset, sha256: hash } });
      else {
        parts.push({ offset, length: bytes.length, made: { sha256: hash } });
        made.set(hash, { source, offset, length: bytes.length });
      }
    }
    files.push({ location, size: source.size, parts });
  }
  return files;
}

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** An empty KV tensor (no positions), standing for a state held on the device. */
function emptyState(Tensor: TensorConstructor, input: WireInput | undefined, model: ModelSpec): Tensor {
  const dims = emptyCacheDims(input?.shape ?? [1, 'heads', 'past', 'size'], model.architecture.kvHeads, model.architecture.headDim);
  return input?.type === 'float32' ? new Tensor('float32', new Float32Array(0), dims) : new Tensor('float16', new Uint16Array(0), dims);
}
