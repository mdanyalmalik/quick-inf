import { useEffect, useState } from 'react';
import { findModel } from '../../catalogue/models';
import { getRuntimeClient } from '../../runtime/client';
import type { CheckResult } from '../../runtime/worker/protocol';
import { formatBytes, formatMs } from '../common/format';
import { DownloadIcon } from '../common/icons';
import { modelRuntimeState, useRuntimeStatus } from '../runtime/use-runtime';

const PHASE_LABEL = {
  checking: 'Checking this device…',
  downloading: 'Downloading',
  verifying: 'Verifying files…',
  loading: 'Loading onto the GPU…',
  validating: 'Running a test pass…',
} as const;

interface ModelLoadPanelProps {
  modelId: string;
  /** Show a compact confirmation once the model is ready. */
  showReady?: boolean;
}

/** Load / progress / error card for one model. Shows the download size before anything starts. */
export function ModelLoadPanel({ modelId, showReady = false }: ModelLoadPanelProps) {
  const status = useRuntimeStatus();
  const state = modelRuntimeState(status, modelId);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const model = findModel(modelId);

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

  if (!model) return null;

  const load = () => {
    getRuntimeClient()
      .load(modelId)
      .catch(() => {}); // failures arrive through the runtime status
  };

  if (state === 'ready' || state === 'busy') {
    if (!showReady || !status.load) return null;
    const t = status.load.timings;
    return (
      <div className="load-panel load-panel--ready">
        <span className="status-dot" aria-hidden="true" />
        Loaded ({status.load.precision}) · download {formatMs(t.downloadMs)} · GPU load {formatMs(t.sessionMs + t.validateMs)}
      </div>
    );
  }

  if (state === 'loading') {
    const progress = status.progress;
    const fraction =
      progress?.phase === 'downloading' && progress.totalBytes > 0 ? progress.loadedBytes / progress.totalBytes : null;
    return (
      <div className="load-panel" role="status">
        <div className="load-panel__row">
          <span>{PHASE_LABEL[progress?.phase ?? 'checking']}</span>
          {progress?.phase === 'downloading' && (
            <span className="muted">
              {formatBytes(progress.loadedBytes)} / {formatBytes(progress.totalBytes)}
            </span>
          )}
        </div>
        <div className={`progress${fraction === null ? ' progress--indeterminate' : ''}`}>
          <div className="progress__bar" style={fraction === null ? undefined : { width: `${fraction * 100}%` }} />
        </div>
      </div>
    );
  }

  const compat = check?.compatibility;
  const artifact = compat?.artifact ?? model.artifacts[0]!;
  const size = formatBytes(model.tokenizer.sizeBytes + artifact.graph.sizeBytes);

  if (compat && !compat.compatible) {
    return (
      <div className="load-panel load-panel--error" role="alert">
        <strong>This device can’t run {model.name}.</strong>
        <span>{compat.reasons.join(' ')}</span>
        <span className="muted">A recent Chrome or Edge on a desktop or laptop is the best bet for WebGPU support.</span>
      </div>
    );
  }

  if (state === 'failed' && status.error) {
    return (
      <div className="load-panel load-panel--error" role="alert">
        <strong>Loading failed.</strong>
        <span>{status.error.message}</span>
        <div>
          <button className="button button--primary" type="button" onClick={load}>
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="load-panel">
      <div className="load-panel__row">
        <span>
          <strong>{model.name}</strong> isn’t loaded.
        </span>
        <button className="button button--primary" type="button" onClick={load}>
          <DownloadIcon />
          Load · {size}
        </button>
      </div>
      <span className="muted">
        Downloads {size} ({artifact.precision}) from Hugging Face. It isn’t saved yet, so reloading the page downloads it
        again.
      </span>
    </div>
  );
}
