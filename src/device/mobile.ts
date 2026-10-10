/** Whether this looks like a phone or tablet browser (works in workers too). */
export function isMobileDevice(): boolean {
  const data = (navigator as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData;
  return data?.mobile ?? /Android|iPhone|iPad/i.test(navigator.userAgent);
}
