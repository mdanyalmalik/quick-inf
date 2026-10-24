// Carrying pipeline messages between a page's worker (where the engine runs; workers have no WebRTC) and the other
// devices' WebRTC channels, which the page holds (info/distributed-plan.md). Used by the app (share-session.ts) and
// the development page (pipeline.ts).
//
// The host page has a channel to each device and learns which device each is from the messages that come from it.
// Signals and runs a device sends on are for another device, so they go straight to that device. A device's page has
// a channel to the host and links directly to the device it passes runs on to (offerLink / answerLink, their offer and
// answer carried by the host); a run goes there if linked, else through the host.

import type { Channel, PipelineMessage } from './protocol';
import { answerLink, offerLink, type Connected } from './webrtc';

/** How long two devices get to link directly before their runs go through the host instead. */
const LINK_TIMEOUT_MS = 10_000;

export interface HostBridge {
  /** A device's channel, once connected: its messages go to the worker, except those for other devices. */
  add(device: Connected): void;
  /** A message from the worker, to the device it names (or every device). */
  fromWorker(message: PipelineMessage): void;
}

export function hostBridge(toWorker: (message: PipelineMessage) => void, onClosed?: (device: Connected) => void): HostBridge {
  const devices = new Map<string, Channel>();
  return {
    add(device) {
      const { channel } = device;
      let id: string | null = null;
      channel.onMessage((m) => {
        if (m.type === 'signal' || m.type === 'run') {
          if (m.type === 'signal') devices.set(m.from, channel);
          devices.get(m.peer)?.send(m);
          return;
        }
        if ('peer' in m) {
          id = m.peer;
          devices.set(m.peer, channel);
        }
        toWorker(m);
      });
      channel.onClose?.(() => {
        if (id) {
          devices.delete(id);
          toWorker({ type: 'bye', peer: id });
        }
        onClosed?.(device);
      });
    },
    fromWorker(m) {
      if ('peer' in m && devices.has(m.peer)) devices.get(m.peer)!.send(m);
      else for (const channel of new Set(devices.values())) channel.send(m);
    },
  };
}

export interface DeviceBridge {
  /** A message from the worker: on to the next device if it's linked, else to the host. */
  fromWorker(message: PipelineMessage): void;
}

export function deviceBridge(host: Channel, toWorker: (message: PipelineMessage) => void, log: (line: string) => void): DeviceBridge {
  const neighbours = new Map<string, Channel>();
  const links = new Map<string, { accept(sdp: string): Promise<void> }>();
  let me = '';
  const linked = (to: string, channel: Channel | null) => {
    if (channel) {
      neighbours.set(to, channel);
      channel.onMessage(toWorker);
    }
    log(channel ? `linked directly to ${to}` : `no direct link to ${to}: runs go through the host`);
    host.send({ type: 'linked', peer: me, to, direct: channel !== null });
  };
  host.onMessage((m) => {
    if (m.type === 'link') {
      void offerLink((sdp) => host.send({ type: 'signal', peer: m.to, from: me, kind: 'offer', sdp })).then(({ accept, open }) => {
        links.set(m.to, { accept });
        void Promise.race([open, new Promise<null>((r) => setTimeout(() => r(null), LINK_TIMEOUT_MS))]).then((channel) => linked(m.to, channel));
      });
    } else if (m.type === 'signal' && m.kind === 'offer') {
      void answerLink(m.sdp, (sdp) => host.send({ type: 'signal', peer: m.from, from: me, kind: 'answer', sdp })).then((channel) => {
        neighbours.set(m.from, channel);
        channel.onMessage(toWorker);
      });
    } else if (m.type === 'signal') void links.get(m.from)?.accept(m.sdp);
    else toWorker(m);
  });
  return {
    fromWorker(m) {
      if (m.type === 'hello') me = m.peer;
      if (m.type === 'run' && neighbours.has(m.peer)) neighbours.get(m.peer)!.send(m);
      else host.send(m);
    },
  };
}
