import type { LoadResult } from '../../runtime/contract';
import { sharedSummary } from '../../distributed/share-session';
import { useShareSnapshot, useSharing } from './SharePanel';

/** Beside "Loaded" in a chat: the model is shared, with which devices, and whether they're all still connected. */
export function SharedBadge({ shared }: { shared: NonNullable<LoadResult['shared']> }) {
  const { devices } = useShareSnapshot(useSharing());
  const lost = shared.devices.filter((d) => devices.some((c) => c.id === d.id && !c.connected));
  const names = shared.devices.map((d) => d.label).join(', ');
  return (
    <span className={`shared-badge${lost.length ? ' shared-badge--lost' : ''}`} title={`Shared: ${sharedSummary(shared)}`}>
      {lost.length
        ? `${lost.map((d) => d.label).join(', ')} disconnected`
        : `Shared with ${shared.devices.length === 1 ? names : `${shared.devices.length} devices`}`}
    </span>
  );
}
