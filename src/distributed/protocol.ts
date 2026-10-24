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

/**
 * Bytes of a file a device's segment reads (as the segment's graph names it), at `offset`: a range of one of the
 * model's files, which the device fetches itself and checks against `sha256` (the host's hash of its verified copy),
 * or bytes the host made as it loaded (weights repacked for the GPU), which a device that hasn't kept them asks for
 * (`need`) and the host sends in pieces (`part`, each acknowledged with `got`).
 */
export interface AssignPart {
  offset: number;
  length: number;
  from?: { path: string; offset: number; sha256: string };
  made?: { sha256: string };
}

export interface AssignFile {
  location: string;
  size: number;
  parts: AssignPart[];
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
  /** The device's segment as the host built it (its graph and where each of its weights is). */
  | { type: 'assign'; peer: string; modelId: string; segment: number; layers: [number, number]; graph: ArrayBuffer; files: AssignFile[] }
  | { type: 'loaded'; peer: string; inputs: WireInput[]; outputTypes: [string, ValueType][]; loadMs: number }
  | { type: 'failed'; peer: string; error: string }
  | { type: 'run'; peer: string; id: number; hops: Hop[]; feeds: Record<string, WireTensor> }
  | { type: 'result'; peer: string; id: number; outputs: Record<string, WireTensor>; ms: number }
  | { type: 'run-error'; peer: string; id: number; error: string }
  | { type: 'free'; peer: string; states: string[] }
  | { type: 'need'; peer: string; parts: string[] }
  | { type: 'part'; peer: string; sha256: string; offset: number; data: ArrayBuffer }
  | { type: 'got'; peer: string; sha256: string; end: number }
  | { type: 'link'; peer: string; to: string }
  | { type: 'linked'; peer: string; to: string; direct: boolean }
  | { type: 'signal'; peer: string; from: string; kind: 'offer' | 'answer'; sdp: string }
  | { type: 'release'; peer: string }
  | { type: 'bye'; peer: string };

export interface Channel {
  send(message: PipelineMessage): void;
  onMessage(handler: (message: PipelineMessage) => void): void;
  /** Called once if the other end goes away (WebRTC channels). */
  onClose?(handler: () => void): void;
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

/** A KV cache input (it grows by the new positions); other states (convolution, recurrent) keep a fixed size. */
export function isKvState(name: string): boolean {
  return /^past_key_values\.\d+\.(key|value)$/.test(name);
}

/** The cache input a cache output feeds next step: `present.N.key` → `past_key_values.N.key`, `present_conv.N` → `past_conv.N`. */
export function pastOf(present: string): string {
  return present.startsWith('present.') ? `past_key_values.${present.slice('present.'.length)}` : present.replace(/^present_/, 'past_');
}

/**
 * An empty KV cache input's dimensions: [batch 1, KV heads, 0 positions, head size], from the graph's own shape where
 * it gives sizes (layers can differ: Gemma 4's full-attention heads are twice as wide), else the model's.
 */
export function emptyCacheDims(shape: readonly (number | string)[], kvHeads: number, headDim: number): number[] {
  return shape.map((d, i) => (i === 2 ? 0 : typeof d === 'number' ? d : i === 1 ? kvHeads : i === 3 ? headDim : 1));
}

/** A tensor's data on the CPU (read back from the GPU if it's there), as sent: a read-back's own buffer, else a copy. */
export async function toWire(tensor: Tensor): Promise<WireTensor> {
  const view = (tensor.location === 'gpu-buffer' ? await tensor.getData() : tensor.data) as ArrayBufferView;
  const whole = tensor.location === 'gpu-buffer' && view.byteOffset === 0 && view.byteLength === view.buffer.byteLength;
  return { type: tensor.type, dims: [...tensor.dims], data: whole ? (view.buffer as ArrayBuffer) : (view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer) };
}

/** A message's tensor buffers, to transfer instead of copy when it's posted between a page and its worker. */
export function buffersOf(message: PipelineMessage): ArrayBuffer[] {
  if (message.type === 'assign') return [message.graph];
  if (message.type === 'part') return [message.data];
  const tensors = message.type === 'run' ? message.feeds : message.type === 'result' ? message.outputs : {};
  return [...new Set(Object.values(tensors).map((t) => t.data))];
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
