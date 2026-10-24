// The host's side of a segment run on another device (segmented-session.ts's Stage): each step's values go out and the
// segment's outputs come back. The device keeps its layers' KV cache; the adapter gets a stand-in for each of its
// present outputs (an empty tensor) that names the state the device holds, and passes it back as the past input of
// the next step, so the device continues that exact state. Stand-ins the adapter drops are freed on the device once
// they're garbage collected (it also keeps only a few states: see peer-stage.ts).

import type { InferenceSession, Tensor } from 'onnxruntime-web';
import type { ModelSpec } from '../catalogue/models';
import type { Segment, ValueType } from '../runtime/onnx/moe-split';
import type { Stage } from '../runtime/onnx/segmented-session';
import { fromWire, toWire, type Channel, type PipelineMessage, type WireTensor } from './protocol';

type TensorConstructor = new (type: Tensor.Type, data: Tensor.DataType, dims: readonly number[]) => Tensor;

/** The host's link to its devices: requests and their replies, by device. */
export class PipelineLink {
  private nextId = 1;
  private readonly runs = new Map<number, { resolve: (m: Extract<PipelineMessage, { type: 'result' }>) => void; reject: (e: Error) => void }>();
  private readonly loads = new Map<string, { resolve: (m: Extract<PipelineMessage, { type: 'loaded' }>) => void; reject: (e: Error) => void }>();

  constructor(readonly channel: Channel) {
    channel.onMessage((m) => {
      if (m.type === 'result' || m.type === 'run-error') {
        const pending = this.runs.get(m.id);
        if (!pending) return;
        this.runs.delete(m.id);
        if (m.type === 'result') pending.resolve(m);
        else pending.reject(new Error(`${m.peer}: ${m.error}`));
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

  run(peer: string, state: string | null, next: string, feeds: Record<string, WireTensor>): Promise<Extract<PipelineMessage, { type: 'result' }>> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.runs.set(id, { resolve, reject });
      this.channel.send({ type: 'run', peer, id, state, next, feeds });
    });
  }
}

export interface RemoteStageOptions {
  link: PipelineLink;
  peer: string;
  model: ModelSpec;
  artifactId: string;
  cuts: number[];
  index: number;
  segment: Segment;
  types: ReadonlyMap<string, ValueType>;
  Tensor: TensorConstructor;
}

/** Segment `index` on device `peer`: assigns it, waits for it to load, and returns its stage. */
export async function createRemoteStage(options: RemoteStageOptions): Promise<Stage> {
  const { link, peer, model, segment, Tensor } = options;
  const loaded = await link.load({
    type: 'assign',
    peer,
    modelId: model.id,
    artifactId: options.artifactId,
    cuts: options.cuts,
    segment: options.index,
    types: segment.passedIn.map((n) => [n, options.types.get(n)!]),
  });
  const inputMetadata = loaded.inputs.map((i) => ({ name: i.name, isTensor: true, type: i.type, shape: i.shape }) as InferenceSession.ValueMetadata);
  const outputTypes = new Map(loaded.outputTypes);
  const pastNames = loaded.inputs.filter((i) => isState(i.name)).map((i) => i.name);
  const presentNames = [...outputTypes.keys()].filter((n) => n.startsWith('present'));
  const holders = new WeakMap<Tensor, string>();
  const freed = new FinalizationRegistry<string>((state) => link.channel.send({ type: 'free', peer, states: [state] }));
  let states = 0;
  return {
    inputMetadata,
    outputTypes,
    async run(feeds) {
      const past = pastNames.map((n) => feeds[n]).find((t) => t && holders.has(t));
      // The adapter's own empty cache starts a new state.
      const state = past ? holders.get(past)! : null;
      const sent: Record<string, WireTensor> = {};
      for (const [name, tensor] of Object.entries(feeds)) if (!pastNames.includes(name)) sent[name] = await toWire(tensor);
      const next = `${options.index}:${++states}`;
      const result = await link.run(peer, state, next, sent);
      const outputs: Record<string, Tensor> = {};
      for (const [name, wire] of Object.entries(result.outputs)) outputs[name] = fromWire(Tensor, wire);
      for (const name of presentNames) {
        const meta = loaded.inputs.find((i) => i.name === name.replace('present', 'past_key_values'));
        const holder = emptyState(Tensor, meta?.type ?? 'float16', model);
        holders.set(holder, next);
        outputs[name] = holder;
      }
      freed.register(outputs[presentNames[0]!]!, next);
      return outputs;
    },
    async release() {
      link.channel.send({ type: 'release', peer });
    },
  };
}

const isState = (name: string) => name.startsWith('past');

/** An empty KV tensor, [1, KV heads, 0, head size], standing for a state held on the device. */
function emptyState(Tensor: TensorConstructor, type: string, model: ModelSpec): Tensor {
  const { kvHeads, headDim } = model.architecture;
  return type === 'float16'
    ? new Tensor('float16', new Uint16Array(0), [1, kvHeads, 0, headDim])
    : new Tensor('float32', new Float32Array(0), [1, kvHeads, 0, headDim]);
}
