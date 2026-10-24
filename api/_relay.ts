// The pipeline's signalling relay (info/distributed-plan.md): a short-lived mailbox per room through which a host and
// the devices joining it swap their WebRTC offers and answers. After that they talk directly; nothing about the
// model or the chat comes here. Messages are encrypted on the devices with a key from the join link's fragment
// (never sent to a server), so the relay only ever stores opaque text under a random room id.
//
//   POST /api/rooms/<room>            body: one message (text)        → { index }
//   GET  /api/rooms/<room>?after=<n>  messages from position n on     → { messages, next }
//
// Rooms expire ROOM_SECONDS after their last message; each holds at most MAX_MESSAGES of at most MAX_MESSAGE_BYTES,
// and each client may make RATE_PER_MINUTE requests a minute. Used by api/rooms/[room].ts (Vercel, Upstash Redis
// through its REST API) and by the dev server (vite.config.ts, in memory). Files starting with _ aren't functions.

export const ROOM_SECONDS = 600;
export const MAX_MESSAGES = 64;
export const MAX_MESSAGE_BYTES = 16 * 1024;
export const RATE_PER_MINUTE = 300;
const ROOM = /^[0-9a-f]{32}$/;

export interface RelayStore {
  /** Appends to the room's list, renewing its expiry; returns the new length. */
  push(key: string, value: string, seconds: number): Promise<number>;
  length(key: string): Promise<number>;
  /** The list from position `from` on. */
  range(key: string, from: number): Promise<string[]>;
  /** Counts a request in this window and returns the count. */
  count(key: string, seconds: number): Promise<number>;
}

export interface RelayResponse {
  status: number;
  body: unknown;
}

export async function handleRelay(
  request: { method: string; room: string; after: number; body: string | null; client: string },
  store: RelayStore,
): Promise<RelayResponse> {
  if (!ROOM.test(request.room)) return { status: 404, body: { error: 'no such room' } };
  const minute = Math.floor(Date.now() / 60_000);
  if ((await store.count(`relay:rate:${request.client}:${minute}`, 60)) > RATE_PER_MINUTE) return { status: 429, body: { error: 'too many requests' } };
  const key = `relay:room:${request.room}`;
  if (request.method === 'GET') {
    const after = Number.isInteger(request.after) && request.after >= 0 ? request.after : 0;
    const messages = await store.range(key, after);
    return { status: 200, body: { messages, next: after + messages.length } };
  }
  if (request.method === 'POST') {
    const body = request.body ?? '';
    if (!body || new TextEncoder().encode(body).length > MAX_MESSAGE_BYTES) return { status: 413, body: { error: 'message too large' } };
    if ((await store.length(key)) >= MAX_MESSAGES) return { status: 409, body: { error: 'room full' } };
    const length = await store.push(key, body, ROOM_SECONDS);
    return { status: 200, body: { index: length - 1 } };
  }
  return { status: 405, body: { error: 'method not allowed' } };
}

/** In memory, for the dev server and tests. */
export function memoryStore(): RelayStore {
  const lists = new Map<string, { values: string[]; expires: number }>();
  const counts = new Map<string, { n: number; expires: number }>();
  const live = <T extends { expires: number }>(map: Map<string, T>, key: string) => {
    const entry = map.get(key);
    if (entry && entry.expires < Date.now()) map.delete(key);
    return map.get(key);
  };
  return {
    async push(key, value, seconds) {
      const list = live(lists, key) ?? { values: [], expires: 0 };
      list.values.push(value);
      list.expires = Date.now() + seconds * 1000;
      lists.set(key, list);
      return list.values.length;
    },
    async length(key) {
      return live(lists, key)?.values.length ?? 0;
    },
    async range(key, from) {
      return live(lists, key)?.values.slice(from) ?? [];
    },
    async count(key, seconds) {
      const entry = live(counts, key) ?? { n: 0, expires: Date.now() + seconds * 1000 };
      entry.n++;
      counts.set(key, entry);
      return entry.n;
    },
  };
}

/** Upstash Redis through its REST API (the database connected in Vercel: KV_REST_API_URL, KV_REST_API_TOKEN). */
export function restStore(url: string, token: string): RelayStore {
  const pipeline = async (commands: (string | number)[][]): Promise<unknown[]> => {
    const response = await fetch(`${url}/pipeline`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify(commands),
    });
    if (!response.ok) throw new Error(`Redis: HTTP ${response.status}`);
    const results = (await response.json()) as { result?: unknown; error?: string }[];
    for (const r of results) if (r.error) throw new Error(`Redis: ${r.error}`);
    return results.map((r) => r.result);
  };
  return {
    async push(key, value, seconds) {
      const [length] = await pipeline([
        ['RPUSH', key, value],
        ['EXPIRE', key, seconds],
      ]);
      return Number(length);
    },
    async length(key) {
      return Number((await pipeline([['LLEN', key]]))[0]);
    },
    async range(key, from) {
      return (await pipeline([['LRANGE', key, from, -1]]))[0] as string[];
    },
    async count(key, seconds) {
      const [n] = await pipeline([
        ['INCR', key],
        ['EXPIRE', key, seconds],
      ]);
      return Number(n);
    },
  };
}
