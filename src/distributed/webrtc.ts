// Devices on the local network connected directly over WebRTC data channels, introduced through the relay
// (relay-client.ts): a device says hello, the host sends it an offer, the device answers, and the data channel opens;
// then the relay isn't used again. Each side gathers all its addresses before sending its description (no trickle),
// so one offer and one answer suffice. No STUN or TURN servers: local networks only for now.
//
// A data channel carries text or binary messages of limited size (Chrome: 256 KB), so each PipelineMessage is encoded
// as one binary blob (a JSON header with its ArrayBuffers after it) and sent in frames of FRAME_BYTES, reassembled on
// arrival. The channel is ordered and reliable; sending waits while too much is queued.
//
// Both sides show a match code from the two connection certificates' fingerprints: if they agree, nothing (the relay
// included) sits between the devices.

import type { Channel, PipelineMessage } from './protocol';
import { openRoom, type RelayMessage } from './relay-client';

const FRAME_BYTES = 64 * 1024;
const FRAME_HEADER = 12;
const QUEUE_HIGH = 8 * 1024 * 1024;
const QUEUE_LOW = 1024 * 1024;
const GATHER_TIMEOUT_MS = 3000;
const CONFIG: RTCConfiguration = { iceServers: [] };

export interface Connected {
  peer: string;
  label: string;
  channel: Channel;
  /** The same on both devices when they're connected to each other directly. */
  matchCode: string;
}

/** The host's side: accepts each device that says hello in the room, calling `onDevice` once its channel is open. */
export async function hostRoom(secret: string, onDevice: (device: Connected) => void, log: (line: string) => void): Promise<{ close(): void }> {
  const me = 'host';
  const room = await openRoom(secret, me);
  const pending = new Map<string, { pc: RTCPeerConnection; label: string }>();
  const stop = room.listen((m) => {
    void (async () => {
      if (m.kind === 'hello' && !pending.has(m.from)) {
        const pc = new RTCPeerConnection(CONFIG);
        pending.set(m.from, { pc, label: m.label ?? m.from });
        const dc = pc.createDataChannel('pipeline', { ordered: true });
        dc.binaryType = 'arraybuffer';
        dc.onopen = async () => {
          const label = pending.get(m.from)?.label ?? m.from;
          onDevice({ peer: m.from, label, channel: dataChannel(dc), matchCode: await matchCode(pc) });
        };
        await pc.setLocalDescription(await pc.createOffer());
        await gathered(pc);
        await room.post({ from: me, to: m.from, kind: 'offer', sdp: pc.localDescription!.sdp });
        log(`offered a connection to ${m.label ?? m.from}`);
      } else if (m.kind === 'answer') {
        const entry = pending.get(m.from);
        if (entry && !entry.pc.remoteDescription) await entry.pc.setRemoteDescription({ type: 'answer', sdp: m.sdp! });
      }
    })().catch((err: unknown) => log(`could not connect ${m.from}: ${String(err)}`));
  });
  return {
    close() {
      stop();
      for (const { pc } of pending.values()) pc.close();
    },
  };
}

/** A device's side: says hello, answers the host's offer, and resolves once the channel is open. */
export async function joinRoom(secret: string, label: string, log: (line: string) => void): Promise<Connected> {
  const me = `${label}-${Math.random().toString(36).slice(2, 7)}`;
  const room = await openRoom(secret, me);
  const pc = new RTCPeerConnection(CONFIG);
  const open = new Promise<RTCDataChannel>((resolve) => {
    pc.ondatachannel = (event) => {
      event.channel.binaryType = 'arraybuffer';
      event.channel.onopen = () => resolve(event.channel);
    };
  });
  let answered = false;
  const stop = room.listen((m: RelayMessage) => {
    if (m.kind !== 'offer' || answered) return;
    answered = true;
    void (async () => {
      await pc.setRemoteDescription({ type: 'offer', sdp: m.sdp! });
      await pc.setLocalDescription(await pc.createAnswer());
      await gathered(pc);
      await room.post({ from: me, to: 'host', kind: 'answer', sdp: pc.localDescription!.sdp });
      log('answered the host');
    })().catch((err: unknown) => log(`could not answer: ${String(err)}`));
  });
  await room.post({ from: me, kind: 'hello', label });
  log('asked the host to connect');
  const dc = await open;
  stop();
  return { peer: me, label, channel: dataChannel(dc), matchCode: await matchCode(pc) };
}

/** Waits until the connection has found all its addresses (or a few seconds, then uses what it has). */
function gathered(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      if (pc.iceGatheringState !== 'complete') return;
      pc.removeEventListener('icegatheringstatechange', done);
      resolve();
    };
    pc.addEventListener('icegatheringstatechange', done);
    setTimeout(resolve, GATHER_TIMEOUT_MS);
  });
}

/** Four digits from both sides' certificate fingerprints (each SDP's a=fingerprint), the same on both devices. */
async function matchCode(pc: RTCPeerConnection): Promise<string> {
  const fingerprint = (sdp = '') => /a=fingerprint:\S+ (\S+)/.exec(sdp)?.[1] ?? '';
  const both = [fingerprint(pc.localDescription?.sdp), fingerprint(pc.remoteDescription?.sdp)].sort().join('|');
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(both)));
  return String(((digest[0]! << 8) | digest[1]!) % 10000).padStart(4, '0');
}

/** A PipelineMessage channel over a data channel. */
export function dataChannel(dc: RTCDataChannel): Channel {
  const handlers: ((m: PipelineMessage) => void)[] = [];
  const partial = new Map<number, { frames: Uint8Array[]; got: number }>();
  let nextId = 0;
  let sending = Promise.resolve();
  dc.bufferedAmountLowThreshold = QUEUE_LOW;
  dc.addEventListener('message', (event: MessageEvent<ArrayBuffer>) => {
    const view = new DataView(event.data);
    const [id, index, count] = [view.getUint32(0), view.getUint32(4), view.getUint32(8)];
    const entry = partial.get(id) ?? { frames: new Array<Uint8Array>(count), got: 0 };
    entry.frames[index] = new Uint8Array(event.data, FRAME_HEADER);
    entry.got++;
    partial.set(id, entry);
    if (entry.got < count) return;
    partial.delete(id);
    const message = decode(join(entry.frames));
    for (const h of handlers) h(message);
  });
  const roomFor = async () => {
    while (dc.bufferedAmount > QUEUE_HIGH) await new Promise((r) => dc.addEventListener('bufferedamountlow', r, { once: true }));
  };
  return {
    send(message) {
      const id = nextId++;
      const blob = encode(message);
      sending = sending.then(async () => {
        const count = Math.max(1, Math.ceil(blob.length / FRAME_BYTES));
        for (let i = 0; i < count; i++) {
          await roomFor();
          const piece = blob.subarray(i * FRAME_BYTES, (i + 1) * FRAME_BYTES);
          const frame = new Uint8Array(FRAME_HEADER + piece.length);
          const view = new DataView(frame.buffer);
          view.setUint32(0, id);
          view.setUint32(4, i);
          view.setUint32(8, count);
          frame.set(piece, FRAME_HEADER);
          dc.send(frame);
        }
      });
    },
    onMessage(handler) {
      handlers.push(handler);
    },
    close: () => dc.close(),
  };
}

/** A message as bytes: its JSON (each ArrayBuffer replaced by its number), then the buffers, each length-prefixed. */
export function encode(message: PipelineMessage): Uint8Array<ArrayBuffer> {
  const buffers: ArrayBuffer[] = [];
  const json = new TextEncoder().encode(
    JSON.stringify(message, (_key, value: unknown) => (value instanceof ArrayBuffer ? { $buffer: buffers.push(value) - 1 } : value)),
  );
  const out = new Uint8Array(4 + json.length + buffers.reduce((n, b) => n + 4 + b.byteLength, 0));
  const view = new DataView(out.buffer);
  view.setUint32(0, json.length);
  out.set(json, 4);
  let at = 4 + json.length;
  for (const b of buffers) {
    view.setUint32(at, b.byteLength);
    out.set(new Uint8Array(b), at + 4);
    at += 4 + b.byteLength;
  }
  return out;
}

export function decode(bytes: Uint8Array): PipelineMessage {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const jsonLength = view.getUint32(0);
  const buffers: ArrayBuffer[] = [];
  for (let at = 4 + jsonLength; at < bytes.length; ) {
    const length = view.getUint32(at);
    buffers.push(bytes.slice(at + 4, at + 4 + length).buffer);
    at += 4 + length;
  }
  return JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + jsonLength)), (_key, value: unknown) =>
    value && typeof value === 'object' && '$buffer' in value ? buffers[(value as { $buffer: number }).$buffer] : value,
  ) as PipelineMessage;
}

function join(frames: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(frames.reduce((n, f) => n + f.length, 0));
  let at = 0;
  for (const f of frames) {
    out.set(f, at);
    at += f.length;
  }
  return out;
}
