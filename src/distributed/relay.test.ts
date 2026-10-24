import { afterEach, describe, expect, it, vi } from 'vitest';
import { handleRelay, MAX_MESSAGE_BYTES, MAX_MESSAGES, memoryStore } from '../../api/_relay';
import { newSecret, openRoom, type RelayMessage } from './relay-client';
import type { PipelineMessage } from './protocol';
import { decode, encode } from './webrtc';

const ROOM = '0123456789abcdef0123456789abcdef';
const request = (method: string, body: string | null = null, after = 0, room = ROOM) => ({ method, room, after, body, client: 'test' });

describe('relay', () => {
  it('keeps a room’s messages in order and pages through them', async () => {
    const store = memoryStore();
    expect((await handleRelay(request('POST', 'a'), store)).body).toEqual({ index: 0 });
    await handleRelay(request('POST', 'b'), store);
    expect((await handleRelay(request('GET'), store)).body).toEqual({ messages: ['a', 'b'], next: 2 });
    expect((await handleRelay(request('GET', null, 1), store)).body).toEqual({ messages: ['b'], next: 2 });
  });

  it('refuses bad rooms, large messages and full rooms', async () => {
    const store = memoryStore();
    expect((await handleRelay(request('GET', null, 0, 'not-a-room'), store)).status).toBe(404);
    expect((await handleRelay(request('POST', 'x'.repeat(MAX_MESSAGE_BYTES + 1)), store)).status).toBe(413);
    for (let i = 0; i < MAX_MESSAGES; i++) await handleRelay({ ...request('POST', 'm'), client: `c${i}` }, store);
    expect((await handleRelay(request('POST', 'one more'), store)).status).toBe(409);
  });
});

describe('relay room', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('carries encrypted messages between the host and a device, which the relay can’t read', async () => {
    const store = memoryStore();
    const stored: string[] = [];
    vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
      const url = new URL(input, 'https://localhost');
      const body = typeof init?.body === 'string' ? init.body : null;
      if (body) stored.push(body);
      const result = await handleRelay({ method: init?.method ?? 'GET', room: url.pathname.split('/').pop()!, after: Number(url.searchParams.get('after') ?? 0), body, client: 'x' }, store);
      return new Response(JSON.stringify(result.body), { status: result.status });
    });
    const secret = newSecret();
    const host = await openRoom(secret, 'host');
    const device = await openRoom(secret, 'phone');
    const stranger = await openRoom(newSecret(), 'stranger');
    const got: RelayMessage[] = [];
    const stop = host.listen((m) => got.push(m), 5);
    await device.post({ from: 'phone', kind: 'hello', label: 'Phone' });
    await stranger.post({ from: 'stranger', kind: 'hello' });
    await vi.waitFor(() => expect(got).toHaveLength(1));
    stop();
    expect(got[0]).toEqual({ from: 'phone', kind: 'hello', label: 'Phone' });
    expect(stored.join()).not.toContain('Phone');
  });
});

describe('data channel messages', () => {
  it('round-trip with their tensors’ bytes', () => {
    const data = Float32Array.of(1.5, -2, 3).buffer;
    const message: Extract<PipelineMessage, { type: 'run' }> = { type: 'run', peer: 'p', id: 7, hops: [{ peer: 'p', state: null, next: '1:1', feed: ['x'], carry: [] }], feeds: { x: { type: 'float32', dims: [1, 3], data } } };
    const back = decode(encode(message));
    expect(back).toMatchObject({ type: 'run', peer: 'p', id: 7, hops: [{ peer: 'p', state: null, next: '1:1', feed: ['x'], carry: [] }] });
    const x = (back as typeof message).feeds.x!;
    expect(Array.from(new Float32Array(x.data))).toEqual([1.5, -2, 3]);
    expect(x.dims).toEqual([1, 3]);
  });
});
