import { simulation } from './simulation';

/** Whether this looks like a phone or tablet browser (works in workers too). */
export function isMobileDevice(): boolean {
  if (simulation()?.mobile) return true;
  const nav = navigator as Navigator & { userAgentData?: { mobile?: boolean } };
  if (nav.userAgentData?.mobile !== undefined) return nav.userAgentData.mobile;
  if (/Android|iPhone|iPad/i.test(nav.userAgent)) return true;
  // iPad Safari asks for desktop sites by default, with a Mac user agent; only the touch screen gives it away.
  return /Macintosh/.test(nav.userAgent) && (nav.maxTouchPoints ?? 0) > 1;
}
