import type { Session } from '../types';

export interface SessionGroup {
  label: string;
  items: Session[];
}

const DAY = 24 * 60 * 60 * 1000;

/** Buckets sessions into Today / Yesterday / Previous 7 days / Older, newest first. */
export function groupByRecency(sessions: Session[], now = new Date()): SessionGroup[] {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const buckets: SessionGroup[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 days', items: [] },
    { label: 'Older', items: [] },
  ];

  const sorted = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const s of sorted) {
    const index =
      s.updatedAt >= startOfToday ? 0
      : s.updatedAt >= startOfToday - DAY ? 1
      : s.updatedAt >= startOfToday - 7 * DAY ? 2
      : 3;
    buckets[index]!.items.push(s);
  }
  return buckets.filter((b) => b.items.length > 0);
}
