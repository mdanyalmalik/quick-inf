import { useEffect, useState } from 'react';
import { artifactBytes, engineName, findModel } from '../../catalogue/models';
import { findImageModel, imageModelBytes, runsInApp } from '../../catalogue/image-models';
import { getRuntimeClient } from '../../runtime/client';
import type { LoadProgress } from '../../runtime/contract';
import type { CheckResult } from '../../runtime/worker/protocol';
import { formatBytes, formatMs, formatRemaining } from '../common/format';
import { DownloadIcon } from '../common/icons';
import { errorText } from '../runtime/error-text';
import { modelRuntimeState, useRuntimeStatus, useStorage } from '../runtime/use-runtime';

const STEP_COUNT = 5;

/** Step number and label for each load phase. Saved models skip step 3 (they were verified when downloaded). */
function stepFor(progress: LoadProgress): { step: number; label: string } {
  switch (progress.phase) {
    case 'checking':
      return { step: 1, label: 'Checking this device' };
    case 'waiting':
      return { step: 1, label: 'Waiting for another tab to finish downloading' };
    case 'downloading':
      return { step: 2, label: progress.resumed ? 'Resuming download' : 'Downloading' };
    case 'reading':
      return { step: 2, label: 'Reading saved files' };
    case 'verifying':
      return { step: 3, label: 'Verifying files' };
    case 'loading':
      return { step: 4, label: 'Loading onto the GPU' };
    case 'validating':
      return { step: 5, label: 'Running a test pass' };
  }
}

interface ModelLoadPanelProps {
  modelId: string;
  /** Show a compact confirmation (with Unload) once the model is ready. */
  showReady?: boolean;
}

/** Load / progress / error card for one model. Shows the download size before anything starts. */
export function ModelLoadPanel({ modelId, showReady = false }: ModelLoadPanelProps) {
  const status = useRuntimeStatus();
  const state = modelRuntimeState(status, modelId);
  const storage = useStorage();
  const saved = storage.models[modelId];
  const [check, setCheck] = useState<CheckResult | null>(null);
  const model = findModel(modelId);
  const imageModel = findImageModel(modelId);

  useEffect(() => {
    let alive = true;
    getRuntimeClient()
      .check(modelId)
      .then((result) => alive && setCheck(result))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [modelId]);

  if (!model && !(imageModel && runsInApp(imageModel))) return null;
  const name = model?.name ?? imageModel!.name;

  const client = getRuntimeClient();
  const load = () => {
    client.load(modelId).catch(() => {}); // failures arrive through the runtime status
  };

  if (state === 'ready' || state === 'busy') {
    if (!showReady || !status.load) return null;
    const t = status.load.timings;
    return (
      <div className="load-panel load-panel--ready">
        <div className="load-panel__row">
          <span className="load-panel__ready-text">
            <span className="status-dot" aria-hidden="true" />
            Loaded ({engineName(status.load.engine)}, {status.load.precision}) ·{' '}
            {status.load.downloadedBytes > 0
              ? `downloaded in ${formatMs(t.downloadMs)}`
              : t.readMs >= 1
                ? `read from this device in ${formatMs(t.readMs)}`
                : 'from this device'}{' '}
            · GPU load {formatMs(t.sessionMs + t.validateMs)}
          </span>
          <UnloadButton disabled={state === 'busy'} />
        </div>
      </div>
    );
  }

  if (state === 'loading') {
    const progress = status.progress ?? { phase: 'checking' as const };
    const { step, label } = stepFor(progress);
    const downloading = progress.phase === 'downloading' ? progress : null;
    const fraction = downloading && downloading.totalBytes > 0 ? downloading.loadedBytes / downloading.totalBytes : null;
    const remaining =
      downloading?.bytesPerSecond && downloading.bytesPerSecond > 0
        ? (downloading.totalBytes - downloading.loadedBytes) / downloading.bytesPerSecond
        : null;
    return (
      <div className="load-panel" role="status">
        <div className="load-panel__row">
          <span>
            <span className="muted">
              Step {step} of {STEP_COUNT} ·{' '}
            </span>
            {label}
            {progress.phase !== 'downloading' && '…'}
          </span>
          <button className="button button--ghost button--small" type="button" onClick={() => client.cancelLoad()}>
            Cancel
          </button>
        </div>
        <div className={`progress${fraction === null ? ' progress--indeterminate' : ''}`}>
          <div className="progress__bar" style={fraction === null ? undefined : { width: `${fraction * 100}%` }} />
        </div>
        {downloading && (
          <span className="muted load-panel__detail">
            {formatBytes(downloading.loadedBytes)} of {formatBytes(downloading.totalBytes)}
            {downloading.bytesPerSecond ? ` · ${formatBytes(downloading.bytesPerSecond)}/s` : ''}
            {remaining !== null ? ` · ${formatRemaining(remaining)}` : ''}
          </span>
        )}
        {(progress.phase === 'loading' || progress.phase === 'validating') && (
          <span className="muted load-panel__detail">Cancelling takes effect when this step finishes.</span>
        )}
        {progress.phase === 'downloading' && (
          <span className="muted load-panel__detail">Saved as it downloads; if interrupted, the next load continues where it stopped.</span>
        )}
      </div>
    );
  }

  const compat = check?.compatibility;
  // What a load would fetch: the chat model's best file for this device, or the image model's files.
  const chatArtifact = model ? (compat?.artifact ?? model.artifacts[0]!) : null;
  const artifact = chatArtifact
    ? { id: chatArtifact.id, precision: chatArtifact.precision, bytes: artifactBytes(model!, chatArtifact) }
    : { id: 'onnx', precision: imageModel!.precision ?? 'fp16', bytes: imageModelBytes(imageModel!) };
  const size = formatBytes(artifact.bytes);

  if (compat && !compat.compatible) {
    const text = errorText(check.device.webgpu === 'available' ? 'incompatible-device' : 'webgpu-unavailable');
    return (
      <div className="load-panel load-panel--error" role="alert">
        <strong>{text.title}</strong>
        <span>{text.hint}</span>
        <ErrorDetails message={compat.reasons.join(' ')} />
      </div>
    );
  }

  if (state === 'failed' && status.error) {
    const text = errorText(status.error.code);
    return (
      <div className="load-panel load-panel--error" role="alert">
        <strong>{text.title}</strong>
        <span>{text.hint}</span>
        <ErrorDetails message={status.error.message} />
        {status.error.retryable && (
          <div>
            <button className="button button--primary" type="button" onClick={load}>
              {status.needsRestart ? 'Restart and load again' : 'Try again'}
            </button>
          </div>
        )}
      </div>
    );
  }

  const savedHere = saved?.completeArtifacts.includes(artifact.id) ?? false;
  const remaining = artifact.bytes - (saved?.partial ? saved.resumableBytes : 0);
  const evictable = storage.persisted === false ? ' The browser may clear saved models if storage runs low.' : '';

  let action: string;
  let note: string;
  if (savedHere) {
    action = 'Load';
    note = `Saved on this device (${formatBytes(saved!.savedBytes)}); loads without downloading.${evictable}`;
  } else if (saved?.partial) {
    action = `Resume download · ${formatBytes(remaining)} left`;
    note = `The last download stopped part-way; ${formatBytes(saved.resumableBytes)} is already saved.`;
  } else if (!storage.available) {
    action = `Download · ${size}`;
    note = `Downloads ${size} (${artifact.precision}) from Hugging Face. This browser window can’t save models, so each load downloads it again.`;
  } else {
    action = `Download · ${size}`;
    note = `Downloads ${size} (${artifact.precision}) from Hugging Face once, then keeps it on this device.`;
  }

  return (
    <div className="load-panel">
      <div className="load-panel__row">
        <span>
          <strong>{name}</strong> isn’t loaded.
        </span>
        <span className="load-panel__actions">
          {saved && <DeleteButton modelId={modelId} />}
          <button className="button button--primary" type="button" onClick={load}>
            {!savedHere && <DownloadIcon />}
            {action}
          </button>
        </span>
      </div>
      <span className="muted">{note}</span>
      {model?.experimental && <span className="load-panel__note load-panel__warning">{model.experimental}</span>}
      {imageModel?.license && (
        <span className="load-panel__note muted">
          By downloading you accept the{' '}
          <a href={imageModel.license.url} target="_blank" rel="noreferrer">
            {imageModel.license.name}
          </a>
          {imageModel.license.usePolicyUrl && (
            <>
              {' '}
              and its{' '}
              <a href={imageModel.license.usePolicyUrl} target="_blank" rel="noreferrer">
                acceptable use policy
              </a>
            </>
          )}
          .
        </span>
      )}
    </div>
  );
}

function ErrorDetails({ message }: { message: string }) {
  if (!message) return null;
  return (
    <details className="error-details">
      <summary>Technical details</summary>
      <code>{message}</code>
    </details>
  );
}

const CONFIRM_MS = 4000;

/** Deletes the model's saved files, after a second click to confirm. */
function DeleteButton({ modelId }: { modelId: string }) {
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!confirming) return;
    const timer = setTimeout(() => setConfirming(false), CONFIRM_MS);
    return () => clearTimeout(timer);
  }, [confirming]);

  const onClick = () => {
    if (!confirming) {
      setError(null);
      setConfirming(true);
      return;
    }
    setConfirming(false);
    getRuntimeClient()
      .deleteModel(modelId)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  };

  return (
    <button
      className={`button button--ghost button--small${confirming ? ' button--danger' : ''}`}
      type="button"
      onClick={onClick}
      title={error ?? 'Delete the saved copy from this device'}
    >
      {error ? 'Couldn’t delete' : confirming ? 'Confirm delete' : 'Delete'}
    </button>
  );
}

/** Frees the model's GPU memory. */
export function UnloadButton({ disabled = false }: { disabled?: boolean }) {
  return (
    <button
      className="button button--ghost button--small"
      type="button"
      disabled={disabled}
      title={disabled ? 'Stop the model first' : 'Free the model’s GPU memory'}
      onClick={() => {
        getRuntimeClient()
          .unload()
          .catch(() => {});
      }}
    >
      Unload
    </button>
  );
}
