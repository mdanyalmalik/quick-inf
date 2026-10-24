import { describe, expect, it, vi } from 'vitest';
import type { RuntimeClient } from '../runtime/client';
import type { BuildSignals } from '../runtime/onnx/ort-env';

const signals = vi.hoisted(() => ({ current: { jspi: false, webkit: true, webgpu: true } as BuildSignals }));
vi.mock('../runtime/onnx/ort-env', async (original) => ({
  ...(await original<typeof import('../runtime/onnx/ort-env')>()),
  browserSignals: () => signals.current,
}));

const { JoinSession } = await import('./share-session');

/** A client that fails the test if the session uses it: an unsupported device mustn't join. */
const client = new Proxy({} as RuntimeClient, {
  get: (_, name) => {
    throw new Error(`used the runtime (${String(name)})`);
  },
});

describe('JoinSession', () => {
  it('declines to join where models run on the CPU only (Safari’s engine without JSPI)', () => {
    signals.current = { jspi: false, webkit: true, webgpu: true };
    const snapshot = new JoinSession(client, 'secret').getSnapshot();
    expect(snapshot.phase).toBe('unsupported');
    expect(snapshot.error).toMatch(/only run on the CPU/);
  });

  it('declines to join without WebGPU', () => {
    signals.current = { jspi: false, webkit: false, webgpu: false };
    const snapshot = new JoinSession(client, 'secret').getSnapshot();
    expect(snapshot.phase).toBe('unsupported');
    expect(snapshot.error).toMatch(/no WebGPU/);
  });
});
