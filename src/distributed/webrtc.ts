// Devices on the local network connected directly over WebRTC data channels, introduced through the relay
// (relay-client.ts): a device says hello, the host sends it an offer, the device answers, and the data channel opens;
// then the relay isn't used again. Devices that pass runs to each other link up the same way, their offer and answer
// carried by the host (offerLink, answerLink). Each side gathers all its addresses before sending its description (no trickle),
// so one offer and one answer suffice. Local networks only (no TURN server to carry traffic between networks).
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
/** How long a joining device waits for the host's offer, then for the direct connection to open, before giving up. */
const OFFER_WAIT_MS = 20_000;
const CONNECT_WAIT_MS = 20_000;
export const NO_OFFER = 'the host didn’t answer: check its share panel is still open, then open a new join link from it';
export const NO_LINK =
  'this device couldn’t reach the host directly: check both are on the same Wi-Fi network (not a guest network, and no VPN). ' +
  'On a Mac, also let this browser find devices on the local network (System Settings → Privacy & Security → Local Network).';
/** How long a connection may stay disconnected (it can recover) before the device counts as gone. */
const DISCONNECTED_MS = 3000;
// Browsers hide a device's local address behind a random .local name, which the other device looks up by multicast
// DNS on the network; where that's blocked (security software on a Mac did), the names never resolve and the link
// fails. A STUN server gives each side a second address to try, the one its router shows outside (the server sees
// only that address).
const CONFIG: RTCConfiguration = { iceServers: [{ urls: ['stun:stun.l.google.com:19302', 'stun:stun.cloudflare.com:3478'] }] };

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
        pc.onconnectionstatechange = () => {
          if (pc.connectionState === 'failed') void linkDetails(pc).then((details) => log(`could not connect ${m.label ?? m.from}: ${details}`));
        };
        const dc = pc.createDataChannel('pipeline', { ordered: true });
        dc.binaryType = 'arraybuffer';
        dc.onopen = async () => {
          const label = pending.get(m.from)?.label ?? m.from;
          onDevice({ peer: m.from, label, channel: dataChannel(dc, pc), matchCode: await matchCode(pc) });
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
  // Without these it would wait forever: for an offer from a host that's gone, or for a connection the network (or, on
  // a Mac, the browser's Local Network permission) never lets open.
  let fail!: (reason: Error) => void;
  const failed = new Promise<never>((_, reject) => (fail = reject));
  failed.catch(() => {}); // a failure after the channel opened is the channel's to report
  let timer = setTimeout(() => fail(new Error(NO_OFFER)), OFFER_WAIT_MS);
  const noLink = () => void linkDetails(pc).then((details) => fail(new Error(`${NO_LINK} (${details})`)));
  pc.onconnectionstatechange = () => {
    if (pc.connectionState === 'failed') noLink();
  };
  const stop = room.listen((m: RelayMessage) => {
    if (m.kind !== 'offer' || answered) return;
    answered = true;
    clearTimeout(timer);
    timer = setTimeout(noLink, CONNECT_WAIT_MS);
    void (async () => {
      await pc.setRemoteDescription({ type: 'offer', sdp: m.sdp! });
      await pc.setLocalDescription(await pc.createAnswer());
      await gathered(pc);
      await room.post({ from: me, to: 'host', kind: 'answer', sdp: pc.localDescription!.sdp });
      log('answered the host');
    })().catch((err: unknown) => log(`could not answer: ${String(err)}`));
  });
  let dc: RTCDataChannel;
  try {
    await room.post({ from: me, kind: 'hello', label });
    log('asked the host to connect');
    dc = await Promise.race([open, failed]);
  } catch (err) {
    pc.close();
    throw err;
  } finally {
    clearTimeout(timer);
    stop();
  }
  return { peer: me, label, channel: dataChannel(dc, pc), matchCode: await matchCode(pc) };
}

/**
 * A direct link from this device to another, set up through the host's channels instead of the relay: `send` carries
 * the offer, `accept` takes the answer, `open` resolves with the channel.
 */
export async function offerLink(send: (sdp: string) => void): Promise<{ accept(sdp: string): Promise<void>; open: Promise<Channel> }> {
  const pc = new RTCPeerConnection(CONFIG);
  const dc = pc.createDataChannel('pipeline', { ordered: true });
  dc.binaryType = 'arraybuffer';
  const open = new Promise<Channel>((resolve) => (dc.onopen = () => resolve(dataChannel(dc, pc))));
  await pc.setLocalDescription(await pc.createOffer());
  await gathered(pc);
  send(pc.localDescription!.sdp);
  return { accept: (sdp) => pc.setRemoteDescription({ type: 'answer', sdp }), open };
}

/** The other end of offerLink: answers `offer` (through `send`) and resolves with the channel once open. */
export async function answerLink(offer: string, send: (sdp: string) => void): Promise<Channel> {
  const pc = new RTCPeerConnection(CONFIG);
  const open = new Promise<Channel>((resolve) => {
    pc.ondatachannel = (event) => {
      event.channel.binaryType = 'arraybuffer';
      event.channel.onopen = () => resolve(dataChannel(event.channel, pc));
    };
  });
  await pc.setRemoteDescription({ type: 'offer', sdp: offer });
  await pc.setLocalDescription(await pc.createAnswer());
  await gathered(pc);
  send(pc.localDescription!.sdp);
  return open;
}

/**
 * What a connection that didn't open had to work with, for its error: the kinds of address each side offered (a
 * .local name, a plain address, or one from STUN) and how its checks between them ended.
 */
export async function linkDetails(pc: RTCPeerConnection): Promise<string> {
  const kinds = (sdp = '') => {
    const found = [...sdp.matchAll(/^a=candidate:\S+ \d+ (\S+) \d+ (\S+) \d+ typ (\w+)/gm)]
      .filter(([, protocol]) => protocol?.toLowerCase() === 'udp')
      .map(([, , address, type]) => (type === 'host' ? (address!.endsWith('.local') ? 'name' : 'address') : type === 'srflx' ? 'STUN' : type!));
    const counts = new Map<string, number>();
    for (const k of found) counts.set(k, (counts.get(k) ?? 0) + 1);
    return [...counts].map(([k, n]) => `${n} ${k}`).join(', ') || 'none';
  };
  const pairs = new Map<string, number>();
  try {
    for (const stat of (await pc.getStats()).values()) {
      if (stat.type === 'candidate-pair') pairs.set(stat.state, (pairs.get(stat.state) ?? 0) + 1);
    }
  } catch {
    // no stats: the rest still says something
  }
  const checks = [...pairs].map(([state, n]) => `${n} ${state}`).join(', ') || 'none tried';
  return `this device offered ${kinds(pc.localDescription?.sdp)}; the other offered ${kinds(pc.remoteDescription?.sdp)}; checks: ${checks}`;
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

/**
 * A PipelineMessage channel over a data channel. It counts as closed when the data channel closes or the connection
 * fails, or stays disconnected for DISCONNECTED_MS (a page that just vanishes often closes nothing: only the
 * connection notices, after a few seconds). Closing it closes the connection.
 */
export function dataChannel(dc: RTCDataChannel, pc: RTCPeerConnection): Channel {
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
    onClose(handler) {
      let fired = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const closed = () => {
        if (fired) return;
        fired = true;
        clearTimeout(timer);
        handler();
      };
      dc.addEventListener('close', closed, { once: true });
      pc.addEventListener('connectionstatechange', () => {
        clearTimeout(timer);
        if (pc.connectionState === 'failed' || pc.connectionState === 'closed') closed();
        else if (pc.connectionState === 'disconnected') timer = setTimeout(closed, DISCONNECTED_MS);
      });
    },
    close: () => pc.close(),
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
