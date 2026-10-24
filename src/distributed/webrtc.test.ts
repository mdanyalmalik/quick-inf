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

const { joinRoom, NO_LINK, NO_OFFER } = await import('./webrtc');

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
});
