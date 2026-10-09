import { useEffect, useRef, useState } from 'react';
import { recommendationStore } from '../../recommend/recommendation-store';
import { CloseIcon, GaugeIcon } from '../common/icons';
import { useRecommendations } from '../recommend/use-recommendations';

interface BenchmarkPromptProps {
  onClose: () => void;
  /** The test finished: show the results. */
  onComplete: () => void;
}

/** First-open welcome: offers the quick device test, which then opens the recommendations page. */
export function BenchmarkPrompt({ onClose, onComplete }: BenchmarkPromptProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const snap = useRecommendations();
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  useEffect(() => {
    if (started && !snap.running && !snap.error && snap.test) onComplete();
  }, [started, snap.running, snap.error, snap.test, onComplete]);

  const run = () => {
    setStarted(true);
    void recommendationStore.runDeviceTest();
  };

  return (
    <dialog
      ref={dialogRef}
      className="dialog"
      aria-labelledby="benchmark-title"
      onCancel={(e) => {
        e.preventDefault();
        if (!snap.running) onClose();
      }}
    >
      <button className="icon-button dialog__close" type="button" onClick={onClose} aria-label="Close" disabled={!!snap.running}>
        <CloseIcon />
      </button>

      <div className="dialog__icon">
        <GaugeIcon width={26} height={26} />
      </div>

      <h2 id="benchmark-title" className="dialog__title">
        Welcome to Quick Inference
      </h2>
      <p className="dialog__text">
        Run a quick device test so Quick Inference can suggest models that suit this device. It takes about five seconds,
        downloads nothing, and runs entirely in your browser.
      </p>

      {snap.running && (
        <div className="dialog__notice" role="status">
          Testing your GPU…
        </div>
      )}
      {started && snap.error && (
        <div className="dialog__notice" role="alert">
          {snap.error}
        </div>
      )}
      {snap.status === 'unavailable' && (
        <div className="dialog__notice" role="alert">
          WebGPU isn’t available in this browser, so models can’t run here.
        </div>
      )}

      <div className="dialog__actions">
        <button className="button button--ghost" type="button" onClick={onClose} disabled={!!snap.running}>
          {started && snap.error ? 'Continue' : 'Skip for now'}
        </button>
        <button
          className="button button--primary"
          type="button"
          onClick={run}
          disabled={!!snap.running || snap.status !== 'ready'}
        >
          {snap.running ? 'Testing…' : started && snap.error ? 'Try again' : 'Run device test'}
        </button>
      </div>
    </dialog>
  );
}
