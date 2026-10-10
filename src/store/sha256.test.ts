import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { Sha256 } from './sha256';

const node = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');

describe('Sha256', () => {
  it('matches the FIPS 180-4 test vectors', () => {
    const text = (s: string) => new Sha256().update(new TextEncoder().encode(s)).hex();
    expect(text('')).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(text('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    expect(text('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq')).toBe(
      '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
    );
  });

  it('gives the same hash however the data is split', () => {
    const data = Uint8Array.from({ length: 100_003 }, (_, i) => (i * 7919) & 0xff);
    const whole = new Sha256().update(data).hex();
    expect(whole).toBe(node(data));
    for (const size of [1, 55, 56, 63, 64, 65, 4096, 77_777]) {
      const sha = new Sha256();
      for (let i = 0; i < data.length; i += size) sha.update(data.subarray(i, i + size));
      expect(sha.hex()).toBe(whole);
    }
  });

  it('handles every length around the padding boundary', () => {
    for (let n = 0; n < 200; n++) {
      const data = Uint8Array.from({ length: n }, (_, i) => (i * 31 + n) & 0xff);
      expect(new Sha256().update(data).hex()).toBe(node(data));
    }
  });

  it('hashes 64 MB at a usable speed', () => {
    const data = new Uint8Array(64 * 1024 * 1024).map((_, i) => i & 0xff);
    const start = performance.now();
    expect(new Sha256().update(data).hex()).toBe(node(data));
    const mbPerSecond = 64 / ((performance.now() - start) / 1000);
    console.log(`sha256: ${Math.round(mbPerSecond)} MB/s`);
    expect(mbPerSecond).toBeGreaterThan(50);
  });
});
