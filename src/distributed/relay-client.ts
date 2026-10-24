// A room on the signalling relay (api/_relay.ts), as the host and the devices joining it use it to swap WebRTC offers
// and answers. The room comes from a secret the host makes and puts in the join link's fragment (after #, never sent
// to a server): the room id is a hash of it, and every message is encrypted with a key derived from it (AES-GCM), so
// the relay sees only a random id and opaque text.

/** What goes through the relay. */
export interface RelayMessage {
  from: string;
  /** For one device; absent: for everyone in the room. */
  to?: string;
  kind: 'hello' | 'offer' | 'answer';
  label?: string;
  sdp?: string;
}

export interface RelayRoom {
  post(message: RelayMessage): Promise<void>;
  /** Calls `handler` with each new message from others, checking every `intervalMs`, until stopped. */
  listen(handler: (message: RelayMessage) => void, intervalMs?: number): () => void;
}

const SECRET_BYTES = 16;

/** A new room secret, for the join link's fragment (base64url). */
export function newSecret(): string {
  return base64url(crypto.getRandomValues(new Uint8Array(SECRET_BYTES)));
}

export async function openRoom(secret: string, me: string, base = '/api/rooms'): Promise<RelayRoom> {
  const raw = new TextEncoder().encode(secret);
  const digest = async (label: string) => new Uint8Array(await crypto.subtle.digest('SHA-256', concat(new TextEncoder().encode(label), raw)));
  const room = [...(await digest('quick-inf room:')).subarray(0, 16)].map((b) => b.toString(16).padStart(2, '0')).join('');
  const key = await crypto.subtle.importKey('raw', await digest('quick-inf key:'), 'AES-GCM', false, ['encrypt', 'decrypt']);
  const url = `${base}/${room}`;

  const seal = async (message: RelayMessage) => {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const sealed = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, new TextEncoder().encode(JSON.stringify(message))));
    return base64url(concat(iv, sealed));
  };
  const open = async (text: string): Promise<RelayMessage | null> => {
    try {
      const bytes = fromBase64url(text);
      const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: bytes.subarray(0, 12) }, key, bytes.subarray(12));
      return JSON.parse(new TextDecoder().decode(plain)) as RelayMessage;
    } catch {
      return null; // not ours (or tampered with)
    }
  };

  return {
    async post(message) {
      const response = await fetch(url, { method: 'POST', body: await seal(message) });
      if (!response.ok) throw new Error(`The relay refused a message (HTTP ${response.status})`);
    },
    listen(handler, intervalMs = 1000) {
      let next = 0;
      let stopped = false;
      const poll = async () => {
        while (!stopped) {
          try {
            const response = await fetch(`${url}?after=${next}`, { cache: 'no-store' });
            if (response.ok) {
              const page = (await response.json()) as { messages: string[]; next: number };
              next = page.next;
              for (const text of page.messages) {
                const message = await open(text);
                if (message && message.from !== me && (!message.to || message.to === me)) handler(message);
              }
            }
          } catch {
            // The relay or the network is briefly unavailable: try again.
          }
          await new Promise((r) => setTimeout(r, intervalMs));
        }
      };
      void poll();
      return () => {
        stopped = true;
      };
    },
  };
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

function base64url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64url(text: string): Uint8Array<ArrayBuffer> {
  const binary = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}
