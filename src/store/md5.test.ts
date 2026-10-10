import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { Md5 } from './md5';

const node = (data: Uint8Array) => createHash('md5').update(data).digest('hex');

describe('Md5', () => {
  it('matches the RFC 1321 test vectors', () => {
    const text = (s: string) => new Md5().update(new TextEncoder().encode(s)).hex();
    expect(text('')).toBe('d41d8cd98f00b204e9800998ecf8427e');
    expect(text('abc')).toBe('900150983cd24fb0d6963f7d28e17f72');
    expect(text('message digest')).toBe('f96b697d7cb7938d525a2f31aaf161d0');
    expect(text('1234567890'.repeat(8))).toBe('57edf4a22be3c955ac49da2e2107b67a');
  });

  it('gives the same hash however the data is split', () => {
    const data = Uint8Array.from({ length: 100_003 }, (_, i) => (i * 7919) & 0xff);
    const whole = new Md5().update(data).hex();
    expect(whole).toBe(node(data));
    for (const size of [1, 55, 56, 63, 64, 65, 4096, 77_777]) {
      const md5 = new Md5();
      for (let i = 0; i < data.length; i += size) md5.update(data.subarray(i, i + size));
      expect(md5.hex()).toBe(whole);
    }
  });

  it('handles every length around the padding boundary', () => {
    for (let n = 0; n < 200; n++) {
      const data = Uint8Array.from({ length: n }, (_, i) => (i * 31 + n) & 0xff);
      expect(new Md5().update(data).hex()).toBe(node(data));
    }
  });
});
