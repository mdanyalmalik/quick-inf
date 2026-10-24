import qrcode from 'qrcode-generator';
import { useState, useSyncExternalStore } from 'react';
import { ShareHost, type ShareHostSnapshot } from '../../distributed/share-session';
import { getRuntimeClient } from '../../runtime/client';
import { formatBytes } from '../common/format';

// Sharing a model across devices (src/distributed): one room per page, opened while the person has sharing on.

let host: ShareHost | null = null;
const listeners = new Set<() => void>();
const changed = () => listeners.forEach((l) => l());

export const sharing = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  get: (): ShareHost | null => host,
  start(): void {
    host ??= new ShareHost(getRuntimeClient());
    changed();
  },
  stop(): void {
    host?.stop();
    host = null;
    changed();
  },
};

export function useSharing(): ShareHost | null {
  return useSyncExternalStore(sharing.subscribe, sharing.get);
}

const EMPTY: ShareHostSnapshot = { joinUrl: '', devices: [], error: null };

export function useShareSnapshot(share: ShareHost | null): ShareHostSnapshot {
  return useSyncExternalStore(share?.subscribe ?? sharing.subscribe, share?.getSnapshot ?? (() => EMPTY));
}

/** The room: a QR code and link for the other devices, and those connected. */
export function SharePanel({ share }: { share: ShareHost }) {
  const { joinUrl, devices, error } = useShareSnapshot(share);
  const [copied, setCopied] = useState(false);
  const [svg] = useState(() => {
    const qr = qrcode(0, 'M');
    qr.addData(joinUrl);
    qr.make();
    return qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
  });
  const connected = devices.filter((d) => d.connected);
  const copy = () => {
    void navigator.clipboard?.writeText(joinUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="share-panel">
      <div className="share-panel__qr" aria-label="QR code with the join link" dangerouslySetInnerHTML={{ __html: svg }} />
      <div className="share-panel__body">
        <span>
          Scan this, or open the link, on other devices on the same Wi-Fi. Each runs part of the model, so together they can
          run a model too big for one.
        </span>
        <span className="share-panel__link">
          <code>{joinUrl}</code>
          <button className="button button--ghost button--small" type="button" onClick={copy}>
            {copied ? 'Copied' : 'Copy link'}
          </button>
        </span>
        {error && <span className="load-panel__warning">{error}</span>}
        {connected.length === 0 ? (
          <span className="muted">No devices connected yet.</span>
        ) : (
          <ul className="share-panel__devices">
            {connected.map((d) => (
              <li key={d.matchCode}>
                <strong>{d.label}</strong>{' '}
                <span className="muted">
                  · match code {d.matchCode}
                  {d.device ? ` · offers ${formatBytes(d.device.memoryBytes)}` : ' · getting ready'}
                </span>
              </li>
            ))}
          </ul>
        )}
        {connected.length > 0 && <span className="muted load-panel__note">Each device shows a match code: it should be the same as here.</span>}
      </div>
    </div>
  );
}
