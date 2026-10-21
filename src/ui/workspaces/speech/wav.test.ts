import { describe, expect, it } from 'vitest';
import { encodeWav } from './wav';

describe('encodeWav', () => {
  it('writes a mono 16-bit header and the pieces in order, clipped to [-1, 1]', async () => {
    const blob = encodeWav([new Float32Array([0, 0.5]), new Float32Array([-1, 2])], 24000);
    const view = new DataView(await blob.arrayBuffer());
    expect(blob.size).toBe(44 + 8);
    expect(view.getUint32(24, true)).toBe(24000);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(40, true)).toBe(8);
    // 0.5 × 32767 = 16383.5, which DataView truncates.
    expect([0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true))).toEqual([0, 16383, -32768, 32767]);
  });
});
