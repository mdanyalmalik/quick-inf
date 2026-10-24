// The page's side of sharing a model across devices (info/distributed-plan.md): the host's room (its join link and
// the devices connected to it) and a device's connection to a host. Both carry pipeline messages between the WebRTC
// channels, which only a page can hold, and the runtime worker (bridge.ts). Framework-free (subscribe/getSnapshot).

import { deviceBudget } from '../recommend/budget';
import { memoryEvidence } from '../recommend/memory-evidence';
import { DEFAULT_CALIBRATION } from '../recommend/estimate';
import { recommendationStore } from '../recommend/recommendation-store';
import type { RuntimeClient } from '../runtime/client';
import type { ShareRequest } from '../runtime/worker/protocol';
import { deviceBridge, hostBridge } from './bridge';
import type { PeerState } from './peer-stage';
import type { DeviceInfo } from './protocol';
import { newSecret } from './relay-client';
import { hostRoom, joinRoom } from './webrtc';

const GB = 1e9;
/** What a device without a benchmark offers: a cautious share, at a modest speed. */
const UNMEASURED_MEMORY_BYTES = 2 * GB;
const UNMEASURED_BANDWIDTH_GBS = 30;

/**
 * This device, as it offers itself to a host: memory and speed from its saved benchmark (benchmarks are kept per site,
 * so one run on another address doesn't count here), or cautious guesses.
 */
export async function thisDevice(): Promise<DeviceInfo & { measured: boolean }> {
  await recommendationStore.init();
  const budget = deviceBudget(memoryEvidence.get());
  const test = recommendationStore.getSnapshot().test;
  return {
    label: deviceLabel(),
    memoryBytes: budget?.bytes ?? UNMEASURED_MEMORY_BYTES,
    bandwidthGBs: test ? test.bandwidthGBps * DEFAULT_CALIBRATION.bandwidthEfficiency : UNMEASURED_BANDWIDTH_GBS,
    measured: budget !== null,
  };
}

/** "iPhone", "Android phone", "Mac"…: how a device is named to the others. */
export function deviceLabel(ua = navigator.userAgent): string {
  if (/iPhone/.test(ua)) return 'iPhone';
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) return 'iPad';
  if (/Android/.test(ua)) return /Mobile/.test(ua) ? 'Android phone' : 'Android tablet';
  if (/Macintosh/.test(ua)) return 'Mac';
  if (/Windows/.test(ua)) return 'Windows PC';
  if (/CrOS/.test(ua)) return 'Chromebook';
  return 'Linux PC';
}

export interface SharedDevice {
  /** Its pipeline id, once it has said hello. */
  id: string | null;
  label: string;
  matchCode: string;
  device: DeviceInfo | null;
  connected: boolean;
}

export interface ShareHostSnapshot {
  joinUrl: string;
  devices: SharedDevice[];
  error: string | null;
}

/** The host's room: a join link and QR code for other devices, and those connected. */
export class ShareHost {
  private snapshot: ShareHostSnapshot;
  private readonly listeners = new Set<() => void>();
  private room: { close(): void } | null = null;

  constructor(private readonly client: RuntimeClient) {
    const secret = newSecret();
    this.snapshot = { joinUrl: `${location.origin}/join#${secret}`, devices: [], error: null };
    const bridge = hostBridge(
      (m) => client.sendPipe(m),
      (connected) => this.updateDevice(connected.matchCode, { connected: false }),
    );
    client.onPipe = (m) => bridge.fromWorker(m);
    hostRoom(
      secret,
      (connected) => {
        this.update({ devices: [...this.snapshot.devices, { id: null, label: connected.label, matchCode: connected.matchCode, device: null, connected: true }] });
        connected.channel.onMessage((m) => {
          if (m.type === 'hello') this.updateDevice(connected.matchCode, { id: m.peer, device: m.device, label: m.device.label });
        });
        bridge.add(connected);
      },
      (line) => console.info(`[shared model] ${line}`),
    )
      .then((room) => (this.room = room))
      .catch((err: unknown) => this.update({ error: `The room couldn’t be opened: ${err instanceof Error ? err.message : String(err)}` }));
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): ShareHostSnapshot => this.snapshot;

  /** What to load with: this device and those connected that have said hello. */
  async request(): Promise<ShareRequest> {
    const { label, memoryBytes, bandwidthGBs } = await thisDevice();
    return {
      host: { label, memoryBytes, bandwidthGBs },
      devices: this.snapshot.devices.flatMap((d) => (d.connected && d.id && d.device ? [{ id: d.id, device: d.device }] : [])),
    };
  }

  stop(): void {
    this.room?.close();
    this.client.onPipe = null;
  }

  private updateDevice(matchCode: string, change: Partial<SharedDevice>): void {
    this.update({ devices: this.snapshot.devices.map((d) => (d.matchCode === matchCode ? { ...d, ...change } : d)) });
  }

  private update(change: Partial<ShareHostSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...change };
    for (const l of this.listeners) l();
  }
}

export interface JoinSnapshot {
  phase: 'connecting' | 'connected' | 'disconnected' | 'failed';
  matchCode: string | null;
  /** What this device is doing for the host's model. */
  peer: PeerState;
  /** Whether this device's memory and speed come from its benchmark (else cautious guesses). */
  measured: boolean;
  error: string | null;
}

/** A device's connection to a host whose join link it opened: it runs the layers the host gives it. */
export class JoinSession {
  private snapshot: JoinSnapshot;
  private readonly listeners = new Set<() => void>();

  constructor(client: RuntimeClient, secret: string) {
    this.snapshot = { phase: 'connecting', matchCode: null, peer: { phase: 'waiting' }, measured: true, error: null };
    client.onPeerState((peer) => this.update({ peer }));
    const log = (line: string) => console.info(`[shared model] ${line}`);
    thisDevice()
      .then(async (device) => {
        this.update({ measured: device.measured });
        return { device, connected: await joinRoom(secret, device.label, log) };
      })
      .then(({ device, connected }) => {
        this.update({ phase: 'connected', matchCode: connected.matchCode });
        const bridge = deviceBridge(connected.channel, (m) => client.sendPipe(m), log);
        client.onPipe = (m) => bridge.fromWorker(m);
        connected.channel.onClose?.(() => this.update({ phase: 'disconnected' }));
        // Closing this page closes the connection, so the host knows at once instead of after a timeout.
        addEventListener('pagehide', () => connected.channel.close(), { once: true });
        const { label, memoryBytes, bandwidthGBs } = device;
        return client.join(connected.peer, { label, memoryBytes, bandwidthGBs });
      })
      .catch((err: unknown) => this.update({ phase: 'failed', error: err instanceof Error ? err.message : String(err) }));
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): JoinSnapshot => this.snapshot;

  private update(change: Partial<JoinSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...change };
    for (const l of this.listeners) l();
  }
}
