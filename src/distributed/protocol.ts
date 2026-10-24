// Messages between the host and the devices running its model's layers (info/distributed-plan.md), and the channel
// they travel on: a BroadcastChannel between windows of this browser, or WebRTC data channels (webrtc.ts).
//
// Flow: the host announces its room, devices say hello with their memory and speed, the host plans the split and
// assigns each device a segment, each loads it and reports its types, and the host links each device to the next
// (over WebRTC: a direct connection, set up through the host; `signal` carries its offer and answer). Then each step
// goes out as a `run` through consecutive devices: each runs its hop and passes the values on to the next itself, and
// the last returns the `result` to the host. A device keeps its layers' KV cache: its hop names the cache state to
// continue (null: a new one) and the state to keep the result as.

import type { Tensor } from 'onnxruntime-web';
import type { ValueType } from '../runtime/onnx/moe-split';

/** What a device offers: memory for its share and speed, from its benchmark (or simulated). */
export interface DeviceInfo {
  label: string;
  memoryBytes: number;
  /** Effective memory bandwidth for running layers, GB/s. */
  bandwidthGBs: number;
}

/** A tensor as sent: CPU data. */
export interface WireTensor {
  type: Tensor.Type;
  dims: number[];
  data: ArrayBuffer;
}

/** A graph input as ONNX Runtime describes it. */
export interface WireInput {
  name: string;
  type: string;
  shape: (number | string)[];
}

/** One device's part of a run. */
export interface Hop {
  peer: string;
  state: string | null;
  next: string;
  /** Values from the run's bundle its segment takes (its own cache states aside). */
  feed: string[];
  /** Values the bundle keeps after it: what later hops take and what goes back to the host. */
  carry: string[];
}

export type PipelineMessage =
  | { type: 'host' }
  | { type: 'hello'; peer: string; device: DeviceInfo }
  | { type: 'assign'; peer: string; modelId: string; artifactId: string; cuts: number[]; segment: number; types: [string, ValueType][] }
  | { type: 'loaded'; peer: string; inputs: WireInput[]; outputTypes: [string, ValueType][]; loadMs: number }
  | { type: 'failed'; peer: string; error: string }
  | { type: 'run'; peer: string; id: number; hops: Hop[]; feeds: Record<string, WireTensor> }
  | { type: 'result'; peer: string; id: number; outputs: Record<string, WireTensor>; ms: number }
  | { type: 'run-error'; peer: string; id: number; error: string }
  | { type: 'free'; peer: string; states: string[] }
  | { type: 'link'; peer: string; to: string }
  | { type: 'linked'; peer: string; to: string; direct: boolean }
  | { type: 'signal'; peer: string; from: string; kind: 'offer' | 'answer'; sdp: string }
  | { type: 'release'; peer: string }
  | { type: 'bye'; peer: string };

export interface Channel {
  send(message: PipelineMessage): void;
  onMessage(handler: (message: PipelineMessage) => void): void;
  close(): void;
}

/**
 * Windows of this browser in the same room. `delayMs` holds each message back that long, as a network hop would
 * (messages keep their order).
 */
export function broadcastChannel(room: string, delayMs = 0): Channel {
  const channel = new BroadcastChannel(`quick-inf-pipeline:${room}`);
  return {
    send(message) {
      if (delayMs > 0) setTimeout(() => channel.postMessage(message), delayMs);
      else channel.postMessage(message);
    },
    onMessage(handler) {
      channel.addEventListener('message', (event: MessageEvent<PipelineMessage>) => handler(event.data));
    },
    close: () => channel.close(),
  };
}

/** A tensor's data on the CPU (read back from the GPU if it's there), as sent. */
export async function toWire(tensor: Tensor): Promise<WireTensor> {
  const data = tensor.location === 'gpu-buffer' ? await tensor.getData() : tensor.data;
  const view = data as ArrayBufferView;
  return { type: tensor.type, dims: [...tensor.dims], data: view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer };
}

type TensorConstructor = new (type: Tensor.Type, data: Tensor.DataType, dims: readonly number[]) => Tensor;

const ARRAYS: Partial<Record<Tensor.Type, (buffer: ArrayBuffer) => Tensor.DataType>> = {
  float32: (b) => new Float32Array(b),
  float16: (b) => new Uint16Array(b),
  int64: (b) => new BigInt64Array(b),
  int32: (b) => new Int32Array(b),
  uint8: (b) => new Uint8Array(b),
  bool: (b) => new Uint8Array(b),
};

export function fromWire(TensorClass: TensorConstructor, wire: WireTensor): Tensor {
  const array = ARRAYS[wire.type];
  if (!array) throw new Error(`Unsupported tensor type ${wire.type}`);
  return new TensorClass(wire.type, array(wire.data), wire.dims);
}
