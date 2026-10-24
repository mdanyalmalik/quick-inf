// A device's side of the pipeline (info/distributed-plan.md): it says hello to the host, loads the segment it's
// assigned (only that segment's weights are read), and runs its hop of each step, keeping its layers' KV cache here,
// then passes the values on to the next device, or back to the host if it's the last. Each hop continues the state
// the host names (none: a new one) and keeps the result under the next name; the most recent few states are kept, as
// the host holds at most a couple of caches at a time. Over a BroadcastChannel every window hears every message, so a
// `link` is answered at once; over WebRTC the page links devices itself (webrtc.ts) and never passes `link` on.

import type { Tensor } from 'onnxruntime-web';
import { findModel, type OnnxArtifact } from '../catalogue/models';
import type { LoadedArtifacts } from '../runtime/contract';
import { createSegmentStage } from '../runtime/onnx/onnx-adapter';
import type { Stage } from '../runtime/onnx/segmented-session';
import { getOrt } from '../runtime/onnx/ort-env';
import { fromWire, toWire, type Channel, type DeviceInfo, type PipelineMessage } from './protocol';

/** States kept at most; older ones are dropped (a run naming one fails). */
const KEPT_STATES = 4;

export interface PeerOptions {
  id: string;
  device: DeviceInfo;
  gpu: GPUDevice;
  channel: Channel;
  /** The model's graph and weight files (read by byte range: only this device's layers are read). */
  files(artifact: OnnxArtifact, modelId: string): Promise<Pick<LoadedArtifacts, 'graph' | 'externalData'>>;
  log(text: string): void;
  /** Extra detail for the loaded line (e.g. bytes fetched). */
  loadDetail?(): string;
}

export function startPeer(options: PeerOptions): { stop(): void } {
  const { id, channel, log } = options;
  let stage: Stage | null = null;
  let pastNames: string[] = [];
  let emptyPast: (() => Record<string, Tensor>) | null = null;
  const states = new Map<string, Record<string, Tensor>>();
  let queue = Promise.resolve();

  const hello = () => channel.send({ type: 'hello', peer: id, device: options.device });
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
        const artifact = model?.artifacts.find((a) => a.id === m.artifactId);
        if (!model || !artifact || artifact.engine !== 'onnx-webgpu') throw new Error(`Unknown model ${m.modelId}`);
        log(`loading segment ${m.segment} of ${model.name} (cuts ${m.cuts.join(', ')})`);
        const files = await options.files(artifact, model.id);
        const loaded = await createSegmentStage(artifact, files, m.cuts, m.segment, new Map(m.types), options.gpu);
        stage = loaded.stage;
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
        const detail = options.loadDetail?.();
        log(`segment ${m.segment} loaded in ${(loadMs / 1000).toFixed(1)} s (${(loaded.segment.weightBytes / 1e9).toFixed(2)} GB of weights${detail ? `; ${detail}` : ''})`);
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
