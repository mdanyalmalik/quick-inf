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
import { fromWire, toWire, type AssignFile, type AssignPart, type Channel, type DeviceInfo, type PipelineMessage } from './protocol';

/** States kept at most; older ones are dropped (a run naming one fails). */
const KEPT_STATES = 4;
/** Parts fetched at once. */
const FETCHES = 4;

export interface PeerOptions {
  id: string;
  device: DeviceInfo;
  gpu: GPUDevice;
  channel: Channel;
  /**
   * A range of one of the model's files: from where this device keeps it, or fetched (and kept); its SHA-256 must
   * be `sha256`. `onBytes` reports bytes as they arrive.
   */
  fetchPart(model: ModelSpec, from: NonNullable<AssignPart['from']>, length: number, onBytes: (n: number) => void): Promise<ByteSource>;
  log(text: string): void;
  /** Bytes of this device's weights obtained so far, of the total (fetching them). */
  onProgress?(done: number, total: number): void;
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

  /** The segment's files, each answering reads from its parts: fetched (FETCHES at a time) or sent by the host. */
  const obtainFiles = async (model: ModelSpec, files: AssignFile[], total: number): Promise<Record<string, ByteSource>> => {
    const parts = files.flatMap((f) => f.parts);
    const sources = new Map<AssignPart, ByteSource>();
    let done = 0;
    const tick = (n: number) => {
      done += n;
      options.onProgress?.(done, total);
    };
    let next = 0;
    const worker = async () => {
      while (next < parts.length) {
        const part = parts[next++]!;
        if (part.data) {
          sources.set(part, bytesSource(new Uint8Array(part.data)));
          tick(part.length);
        } else sources.set(part, await options.fetchPart(model, part.from!, part.length, tick));
      }
    };
    await Promise.all(Array.from({ length: FETCHES }, worker));
    return Object.fromEntries(files.map((f) => [f.location, partsSource(f, sources)]));
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
        await stage?.release();
        stage = null;
        const model = findModel(m.modelId);
        if (!model) throw new Error(`Unknown model ${m.modelId}`);
        const total = m.files.reduce((n, f) => n + f.parts.reduce((k, p) => k + p.length, 0), 0);
        log(`loading segment ${m.segment} of ${model.name} (${(total / 1e9).toFixed(2)} GB of weights)`);
        const files = await obtainFiles(model, m.files, total);
        stage = await createSessionStage(new Uint8Array(m.graph), files, options.gpu);
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
                  ? new ort.Tensor('float16', new Uint16Array(0), [1, kvHeads, 0, headDim])
                  : new ort.Tensor('float32', new Float32Array(0), [1, kvHeads, 0, headDim]),
              ]),
          );
        const loadMs = performance.now() - start;
        log(`segment ${m.segment} loaded in ${(loadMs / 1000).toFixed(1)} s`);
        channel.send({ type: 'loaded', peer: id, inputs, outputTypes: [...stage.outputTypes], loadMs });
      } catch (err) {
        log(`load failed: ${String(err)}`);
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
        const feeds: Record<string, Tensor> = { ...past };
        for (const name of hop!.feed) feeds[name] = fromWire(ort.Tensor, m.feeds[name]!);
        const result = await stage.run(feeds);
        const present: Record<string, Tensor> = {};
        const bundle = { ...m.feeds };
        for (const [name, tensor] of Object.entries(result)) {
          if (name.startsWith('present')) present[name.replace('present', 'past_key_values')] = tensor;
          else {
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
    }
  };

  // One message at a time, in order.
  channel.onMessage((m) => {
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
