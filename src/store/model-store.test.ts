import { describe, expect, it } from 'vitest';
import { reportedFreeBytes } from './model-store';

const GiB = 1024 ** 3;

describe('reportedFreeBytes', () => {
  it('ignores Chrome’s fixed report of usage + 10 GiB', () => {
    expect(reportedFreeBytes(10 * GiB, 0)).toBeNull();
    expect(reportedFreeBytes(14 * GiB + 12345, 4 * GiB + 12345)).toBeNull();
  });

  it('uses real figures (other browsers, older Chrome)', () => {
    expect(reportedFreeBytes(300 * GiB, 20 * GiB)).toBe(280 * GiB);
    expect(reportedFreeBytes(2 * GiB, GiB)).toBe(GiB);
  });

  it('knows nothing without both figures', () => {
    expect(reportedFreeBytes(undefined, 0)).toBeNull();
    expect(reportedFreeBytes(GiB, undefined)).toBeNull();
  });
});
