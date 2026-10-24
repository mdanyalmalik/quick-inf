// A device's side of the pipeline (info/distributed-plan.md): it says hello to the host, loads the segment it's
// assigned (the host builds it; the device fetches just its weights' byte ranges, checking each against the host's
// hash), and runs its hop of each step, keeping its layers' KV cache here,
// then passes the values on to the next device, or back to the host if it's the last. Each hop continues the state
// the host names (none: a new one) and keeps the result under the next name; the most recent few states are kept, as
// the host holds at most a couple of caches at a time. Over a BroadcastChannel every window hears every message, so a
// `link` is answered at once; over WebRTC the page links devices itself (webrtc.ts) and never passes `link` on.

import type { Tensor } from 'onnxruntime-web';
import { findModel, type ModelSpec } from '../catalogue/models';
import { bytesSource, type ByteSource } from '../runtime/onnx/external-weights';
import { createSessionStage } from '../runtime/onnx/onnx-adapter';
import type { Stage } from '../runtime/onnx/segmented-session';
import { getOrt } from '../runtime/onnx/ort-env';
import { emptyCacheDims, fromWire, toWire, type AssignFile, type AssignPart, type Channel, type DeviceInfo, type PipelineMessage } from './protocol';

/** States kept at most; older ones are dropped (a run naming one fails). */
const KEPT_STATES = 4;
/** Parts fetched at once. */
const FETCHES = 4;

/** Where bytes the host sends are written, in order; `finish` checks their hash and opens them for reading. */
export interface MadeWriter {
  write(offset: number, bytes: Uint8Array): void;
  finish(): Promise<ByteSource & { close?(): void }>;
  /** Gives up an unfinished write (the load failed). */
  abort?(): void;
}

/** Kept in memory (no storage): the bytes, checked against their hash. */
function memoryWriter(sha256: string, length: number): MadeWriter {
  const bytes = new Uint8Array(length);
  return {
    write: (offset, data) => bytes.set(data, offset),
    async finish() {
      if ((await hashOf(bytes)) !== sha256) throw new Error('Bytes from the host arrived damaged');
      return bytesSource(bytes);
    },
  };
}

async function hashOf(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** What a device is doing, for its screen. */
export type PeerState =
  | { phase: 'waiting' }
  | { phase: 'loading'; modelId: string; layers: [number, number]; done: number; total: number }
  | { phase: 'ready'; modelId: string; layers: [number, number] }
  | { phase: 'failed'; error: string };

export interface PeerOptions {
  id: string;
  device: DeviceInfo;
  gpu: GPUDevice;
  channel: Channel;
  /**
   * A range of one of the model's files: from where this device keeps it, or fetched (and kept); its SHA-256 must
   * be `sha256`. `onBytes` reports bytes as they arrive.
   */
  fetchPart(model: ModelSpec, from: NonNullable<AssignPart['from']>, length: number, onBytes: (n: number) => void): Promise<ByteSource & { close?(): void }>;
  /** Bytes the host made, kept from an earlier session (by hash), or null; and somewhere to keep new ones (checked). */
  keptMade?(model: ModelSpec, sha256: string, length: number): Promise<(ByteSource & { close?(): void }) | null>;
  keepMade?(model: ModelSpec, sha256: string, length: number): Promise<MadeWriter>;
  log(text: string): void;
  onState?(state: PeerState): void;
}

/** A file of the segment's, answering reads from the parts it was given. */
function partsSource(file: AssignFile, sources: Map<AssignPart, ByteSource>): ByteSource {
  const find = (offset: number, length: number) => {
    const part = file.parts.find((p) => offset >= p.offset && offset + length <= p.offset + p.length);
    if (!part) throw new Error(`${file.location}: bytes ${offset}–${offset + length} weren't sent`);
    return { source: sources.get(part)!, at: offset - part.offset };
  };
  return {
    size: file.size,
    read(offset, length) {
      const { source, at } = find(offset, length);
      return source.read(at, length);
    },
    readInto(offset, target) {
      const { source, at } = find(offset, target.byteLength);
      if (source.readInto) source.readInto(at, target);
      else target.set(source.read(at, target.byteLength));
    },
  };
}

export function startPeer(options: PeerOptions): { stop(): void } {
  const { id, channel, log } = options;
  let stage: Stage | null = null;
  let pastNames: string[] = [];
  let emptyPast: (() => Record<string, Tensor>) | null = null;
  const states = new Map<string, Record<string, Tensor>>();
  let queue = Promise.resolve();

  const hello = () => channel.send({ type: 'hello', peer: id, device: options.device });
  /** Receivers of made bytes the host is sending, by hash (handled as they come, not in the message queue). */
  const incoming = new Map<string, (offset: number, data: Uint8Array) => void>();

  /** The segment's files, each answering reads from its parts: fetched (FETCHES at a time) or sent by the host. */
  const obtainFiles = async (model: ModelSpec, files: AssignFile[], onBytes: (n: number) => void) => {
    const parts = files.flatMap((f) => f.parts);
    // Each distinct content once (parts can repeat it: the same bytes at two places): by hash.
    const key = (p: AssignPart) => (p.from ? `from:${p.from.sha256}` : `made:${p.made!.sha256}`);
    const unique = [...new Map(parts.map((p) => [key(p), p])).values()];
    const obtained = new Map<string, ByteSource & { close?(): void }>();
    let next = 0;
    const fetched = unique.filter((p) => p.from);
    const worker = async () => {
      while (next < fetched.length) {
        const part = fetched[next++]!;
        obtained.set(key(part), await options.fetchPart(model, part.from!, part.length, onBytes));
      }
    };
    // Bytes the host made: those kept from before; the rest asked for once each has somewhere to go.
    const made = async () => {
      const missing: AssignPart[] = [];
      for (const part of unique.filter((p) => p.made)) {
        const kept = await options.keptMade?.(model, part.made!.sha256, part.length);
        if (kept) {
          obtained.set(key(part), kept);
          onBytes(part.length);
        } else missing.push(part);
      }
      if (!missing.length) return;
      const writers = await Promise.all(
        missing.map(async (part) => (await options.keepMade?.(model, part.made!.sha256, part.length)) ?? memoryWriter(part.made!.sha256, part.length)),
      );
      unfinished.push(...writers);
      const arrived = missing.map(
        (part, i) =>
          new Promise<void>((resolve, reject) => {
            let received = 0;
            incoming.set(part.made!.sha256, (offset, data) => {
              writers[i]!.write(offset, data);
              received += data.byteLength;
              onBytes(data.byteLength);
              channel.send({ type: 'got', peer: id, sha256: part.made!.sha256, end: offset + data.byteLength });
              if (received < part.length) return;
              incoming.delete(part.made!.sha256);
              unfinished.splice(unfinished.indexOf(writers[i]!), 1);
              writers[i]!.finish().then((source) => {
                obtained.set(key(part), source);
                resolve();
              }, reject);
            });
          }),
      );
      channel.send({ type: 'need', peer: id, parts: missing.map((p) => p.made!.sha256) });
      await Promise.all(arrived);
    };
    const unfinished: MadeWriter[] = [];
    try {
      await Promise.all([...Array.from({ length: FETCHES }, worker), made()]);
    } catch (err) {
      for (const s of obtained.values()) s.close?.();
      for (const w of unfinished) w.abort?.();
      for (const p of unique) if (p.made) incoming.delete(p.made.sha256);
      throw err;
    }
    // Repeated parts count as obtained too.
    onBytes(parts.length === unique.length ? 0 : parts.filter((p) => !unique.includes(p)).reduce((n, p) => n + p.length, 0));
    const sources = new Map(parts.map((p) => [p, obtained.get(key(p))!]));
    return {
      files: Object.fromEntries(files.map((f) => [f.location, partsSource(f, sources)])),
      close: () => obtained.forEach((s) => s.close?.()),
    };
  };
  const dropState = (name: string) => {
    for (const t of Object.values(states.get(name) ?? {})) t.dispose();
    states.delete(name);
  };

  const handle = async (m: PipelineMessage) => {
    if (m.type === 'host') hello();
    if (!('peer' in m) || m.peer !== id) return;
    if (m.type === 'assign') {
      const start = performance.now();
      try {
        // The old segment's cache states first: their tensors can't be disposed once its session is released.
        for (const s of [...states.keys()]) dropState(s);
        await stage?.release();
        stage = null;
        const model = findModel(m.modelId);
        if (!model) throw new Error(`Unknown model ${m.modelId}`);
        const total = m.files.reduce((n, f) => n + f.parts.reduce((k, p) => k + p.length, 0), 0);
        log(`loading segment ${m.segment} of ${model.name} (${(total / 1e9).toFixed(2)} GB of weights)`);
        let done = 0;
        const progress = () => options.onState?.({ phase: 'loading', modelId: model.id, layers: m.layers, done, total });
        progress();
        const obtained = await obtainFiles(model, m.files, (n) => {
          done += n;
          progress();
        });
        try {
          stage = await createSessionStage(new Uint8Array(m.graph), obtained.files, options.gpu);
        } finally {
          obtained.close();
        }
        const ort = getOrt();
        const inputs = stage.inputMetadata.flatMap((meta) => (meta.isTensor ? [{ name: meta.name, type: meta.type, shape: [...meta.shape] }] : []));
        pastNames = inputs.filter((i) => i.name.startsWith('past')).map((i) => i.name);
        const { kvHeads, headDim } = model.architecture;
        emptyPast = () =>
          Object.fromEntries(
            inputs
              .filter((i) => pastNames.includes(i.name))
              .map((i) => [
                i.name,
                i.type === 'float16'
                  ? new ort.Tensor('float16', new Uint16Array(0), emptyCacheDims(i.shape, kvHeads, headDim))
                  : new ort.Tensor('float32', new Float32Array(0), emptyCacheDims(i.shape, kvHeads, headDim)),
              ]),
          );
        const loadMs = performance.now() - start;
        log(`segment ${m.segment} loaded in ${(loadMs / 1000).toFixed(1)} s`);
        options.onState?.({ phase: 'ready', modelId: model.id, layers: m.layers });
        channel.send({ type: 'loaded', peer: id, inputs, outputTypes: [...stage.outputTypes], loadMs });
      } catch (err) {
        log(`load failed: ${String(err)}`);
        options.onState?.({ phase: 'failed', error: err instanceof Error ? err.message : String(err) });
        channel.send({ type: 'failed', peer: id, error: err instanceof Error ? err.message : String(err) });
      }
    } else if (m.type === 'link') {
      channel.send({ type: 'linked', peer: id, to: m.to, direct: true });
    } else if (m.type === 'run') {
      try {
        if (!stage || !emptyPast) throw new Error('No segment loaded');
        const start = performance.now();
        const ort = getOrt();
        const [hop, ...rest] = m.hops;
        const past = hop!.state ? states.get(hop!.state) : emptyPast();
        if (!past) throw new Error(`Cache state ${hop!.state} is no longer kept`);
        // Its own cache inputs only (a state can hold outputs no input takes: layers that share another's cache).
        const feeds: Record<string, Tensor> = Object.fromEntries(pastNames.flatMap((n) => (past[n] ? [[n, past[n]]] : [])));
        const missing = hop!.feed.filter((n) => !m.feeds[n]);
        if (missing.length) throw new Error(`the step didn't bring ${missing.join(', ')}`);
        for (const name of hop!.feed) feeds[name] = fromWire(ort.Tensor, m.feeds[name]!);
        const result = await stage.run(feeds);
        const present: Record<string, Tensor> = {};
        const bundle = { ...m.feeds };
        for (const [name, tensor] of Object.entries(result)) {
          if (name.startsWith('present')) {
            present[name.replace('present', 'past_key_values')] = tensor;
            // Also passed on when later layers read it (shared caches).
            if (hop!.carry.includes(name)) bundle[name] = await toWire(tensor);
          } else {
            if (hop!.carry.includes(name)) bundle[name] = await toWire(tensor);
            tensor.dispose();
          }
        }
        for (const name of Object.keys(bundle)) if (!hop!.carry.includes(name)) delete bundle[name];
        states.set(hop!.next, present);
        while (states.size > KEPT_STATES) dropState(states.keys().next().value!);
        if (rest.length) channel.send({ type: 'run', peer: rest[0]!.peer, id: m.id, hops: rest, feeds: bundle });
        else channel.send({ type: 'result', peer: id, id: m.id, outputs: bundle, ms: performance.now() - start });
      } catch (err) {
        channel.send({ type: 'run-error', peer: id, id: m.id, error: err instanceof Error ? err.message : String(err) });
      }
    } else if (m.type === 'free') {
      for (const s of m.states) dropState(s);
    } else if (m.type === 'release') {
      for (const s of [...states.keys()]) dropState(s);
      await stage?.release();
      stage = null;
      log('released');
      options.onState?.({ phase: 'waiting' });
    }
  };

  // One message at a time, in order; pieces of made bytes at once (the assignment waiting for them is in the queue).
  channel.onMessage((m) => {
    if (m.type === 'part') {
      if (m.peer === id) incoming.get(m.sha256)?.(m.offset, new Uint8Array(m.data));
      return;
    }
    queue = queue.then(() => handle(m)).catch((err) => log(`error: ${String(err)}`));
  });
  hello();
  return {
    stop() {
      channel.send({ type: 'bye', peer: id });
      channel.close();
    },
  };
}
