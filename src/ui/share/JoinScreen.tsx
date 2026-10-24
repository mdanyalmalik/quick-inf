import { useEffect, useSyncExternalStore } from 'react';
import { findModel } from '../../catalogue/models';
import { JoinSession, layerRange } from '../../distributed/share-session';
import { getRuntimeClient } from '../../runtime/client';
import { formatBytes } from '../common/format';
import { DeleteButton } from '../models/ModelLoadPanel';
import { useStorage } from '../runtime/use-runtime';

// The page another device opens from a host's join link (/join#<secret>): it connects to the host and runs the layers
// the host gives it. One session per page (React may render twice in development).

let session: JoinSession | null = null;

export function JoinScreen() {
  session ??= new JoinSession(getRuntimeClient(), location.hash.slice(1));
  // Development: the session, for scripted tests (tools/bench/share-app.mjs).
  if (import.meta.env.DEV) Object.assign(window, { __quickInfJoin: session });
  const snapshot = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const storage = useStorage();
  const { phase, matchCode, peer, measured, error } = snapshot;
  const active = phase === 'connected';
  useWakeLock(active);

  const model = 'modelId' in peer ? findModel(peer.modelId) : undefined;
  const layers = 'layers' in peer ? `${layerRange(peer.layers)} of ${model?.architecture.layers ?? '?'}` : '';

  return (
    <main className="join-screen">
      <div className="join-screen__card">
        <h1>Sharing a model</h1>
        {phase === 'connecting' && <p>Connecting to the host…</p>}
        {phase === 'failed' && <p className="load-panel__warning">Couldn’t connect to the host: {error}</p>}
        {phase === 'disconnected' && <p>The host disconnected.</p>}
        {matchCode && phase !== 'disconnected' && (
          <p>
            Connected. Match code <strong className="join-screen__code">{matchCode}</strong>: check the host shows the same.
          </p>
        )}
        {active && peer.phase === 'waiting' && <p className="muted">Waiting for the host to load a model.</p>}
        {active && peer.phase === 'loading' && (
          <>
            <p>
              Getting its share of {model?.name ?? 'the model'} ({layers}): {formatBytes(peer.done)} of {formatBytes(peer.total)}
            </p>
            <progress max={peer.total} value={peer.done} />
            <p className="muted load-panel__note">Kept on this device, so next time it starts at once.</p>
          </>
        )}
        {active && peer.phase === 'ready' && (
          <p>
            Running {layers} of <strong>{model?.name ?? 'the model'}</strong>. Keep this page open and the screen on while the host chats.
          </p>
        )}
        {peer.phase === 'failed' && <p className="load-panel__warning">Couldn’t load its share: {peer.error}</p>}
        {Object.entries(storage.models)
          .filter(([, m]) => m.sharedPartBytes > 0)
          .map(([id, m]) => (
            <p key={id} className="join-screen__kept">
              <span>
                {formatBytes(m.sharedPartBytes)} of {findModel(id)?.name ?? id} is kept on this device, so it starts at once next time.
              </span>
              {/* Not while it runs: its layers are in use. */}
              <DeleteButton modelId={id} disabled={active && peer.phase !== 'waiting'} />
            </p>
          ))}
        {(phase === 'disconnected' || phase === 'failed') && (
          <p>
            <a className="button button--primary" href="/">
              Back to home
            </a>
          </p>
        )}
        {!measured && phase !== 'disconnected' && (
          <p className="muted load-panel__note">
            This device hasn’t run the benchmark, so the host gives it a cautious share. Running it once (on the Quick
            Inference home page) lets the host give it as much as it can take.
          </p>
        )}
      </div>
    </main>
  );
}

/** Keeps the screen on while `on` (a phone that sleeps stops running its layers). */
function useWakeLock(on: boolean): void {
  useEffect(() => {
    if (!on || !('wakeLock' in navigator)) return;
    let lock: { release(): Promise<void> } | null = null;
    const request = () => {
      if (document.visibilityState !== 'visible') return;
      (navigator as Navigator & { wakeLock: { request(type: 'screen'): Promise<{ release(): Promise<void> }> } }).wakeLock
        .request('screen')
        .then((l) => (lock = l))
        .catch(() => {});
    };
    request();
    document.addEventListener('visibilitychange', request);
    return () => {
      document.removeEventListener('visibilitychange', request);
      void lock?.release().catch(() => {});
    };
  }, [on]);
}
