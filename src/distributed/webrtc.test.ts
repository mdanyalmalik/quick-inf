import { afterEach, describe, expect, it, vi } from 'vitest';
import type { RelayMessage } from './relay-client';

// A relay room that delivers what the test pushes to the listener (the host's offer, or nothing).
const relay = vi.hoisted(() => ({ deliver: null as ((m: RelayMessage) => void) | null }));
vi.mock('./relay-client', () => ({
  openRoom: async () => ({
    post: async () => {},
    listen: (handler: (m: RelayMessage) => void) => {
      relay.deliver = handler;
      return () => {};
    },
  }),
}));

const { joinRoom, linkDetails, NO_LINK, NO_OFFER } = await import('./webrtc');

/** A connection that answers offers but never opens a channel; its state is the test's to set. */
class StuckConnection {
  static last: StuckConnection;
  connectionState = 'new';
  onconnectionstatechange: (() => void) | null = null;
  ondatachannel = null;
  iceGatheringState = 'complete';
  localDescription = { sdp: 'v=0' };
  closed = false;
  constructor() {
    StuckConnection.last = this;
  }
  setRemoteDescription = async () => {};
  setLocalDescription = async () => {};
  createAnswer = async () => ({ type: 'answer', sdp: 'v=0' });
  close() {
    this.closed = true;
  }
}

describe('joinRoom', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('gives up with a reason when the host never answers', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('RTCPeerConnection', StuckConnection);
    const joining = joinRoom('secret', 'Mac', () => {});
    const result = expect(joining).rejects.toThrow(NO_OFFER);
    await vi.advanceTimersByTimeAsync(20_000);
    await result;
    expect(StuckConnection.last.closed).toBe(true);
  });

  it('gives up with a reason when the direct connection fails', async () => {
    vi.stubGlobal('RTCPeerConnection', StuckConnection);
    const joining = joinRoom('secret', 'Mac', () => {});
    const result = expect(joining).rejects.toThrow(NO_LINK);
    await vi.waitFor(() => expect(relay.deliver).not.toBeNull());
    relay.deliver!({ from: 'host', to: 'me', kind: 'offer', sdp: 'v=0' });
    const pc = StuckConnection.last;
    pc.connectionState = 'failed';
    pc.onconnectionstatechange!();
    await result;
    expect(pc.closed).toBe(true);
  });

  it('says what a failed link had to work with', async () => {
    const candidate = (address: string, type: string, protocol = 'udp') => `a=candidate:1 1 ${protocol} 2122 ${address} 5000 typ ${type}`;
    const pc = {
      localDescription: { sdp: ['v=0', candidate('ab12.local', 'host'), candidate('ab12.local', 'host', 'tcp'), candidate('203.0.113.5', 'srflx')].join('\r\n') },
      remoteDescription: { sdp: ['v=0', candidate('cd34.local', 'host'), candidate('192.168.0.9', 'host')].join('\r\n') },
      getStats: async () => new Map([['p1', { type: 'candidate-pair', state: 'failed' }], ['p2', { type: 'candidate-pair', state: 'failed' }]]),
    } as unknown as RTCPeerConnection;
    expect(await linkDetails(pc)).toBe('this device offered 1 name, 1 STUN; the other offered 1 name, 1 address; checks: 2 failed');
  });
});
