import { afterEach, describe, expect, it, vi } from 'vitest';
import { isMobileDevice } from './mobile';

const MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit (KHTML, like Gecko) Safari';

function browser(userAgent: string, maxTouchPoints: number) {
  vi.stubGlobal('navigator', { userAgent, maxTouchPoints });
}

describe('isMobileDevice', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('treats iPad Safari asking for desktop sites (a Mac user agent with a touch screen) as mobile', () => {
    browser(MAC, 5);
    expect(isMobileDevice()).toBe(true);
  });

  it('keeps a Mac a desktop', () => {
    browser(MAC, 0);
    expect(isMobileDevice()).toBe(false);
  });

  it('knows phones and tablets by their user agent', () => {
    browser('Mozilla/5.0 (iPad; CPU OS like Mac OS X) AppleWebKit (KHTML, like Gecko) CriOS Mobile Safari', 5);
    expect(isMobileDevice()).toBe(true);
    browser('Mozilla/5.0 (Linux; Android) AppleWebKit (KHTML, like Gecko) Chrome Mobile Safari', 5);
    expect(isMobileDevice()).toBe(true);
  });
});
