import { useEffect, useRef } from 'react';
import { CloseIcon, GaugeIcon } from '../common/icons';
import { useRecommendations } from '../recommend/use-recommendations';

interface BenchmarkPromptProps {
  onClose: () => void;
  /** Run the benchmark: the benchmark page opens and runs it there. */
  onRun: () => void;
}

/** First-open welcome: offers the device benchmark, which runs on the benchmark page. */
export function BenchmarkPrompt({ onClose, onRun }: BenchmarkPromptProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const snap = useRecommendations();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className="dialog"
      aria-labelledby="benchmark-title"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <button className="icon-button dialog__close" type="button" onClick={onClose} aria-label="Close">
        <CloseIcon />
      </button>

      <div className="dialog__icon">
        <GaugeIcon width={26} height={26} />
      </div>

      <h2 id="benchmark-title" className="dialog__title">
        Welcome to Quick Inference
      </h2>
      <p className="dialog__text">
        Run a quick benchmark so Quick Inference can suggest models that suit this device. It takes a few seconds, downloads nothing,
        and runs entirely in your browser.
      </p>
      <p className="dialog__warning">
        It ends by filling GPU memory until it runs out, so the tab may crash. If it does, reload the tab or reopen the site: the results
        are kept.
      </p>

      {snap.status === 'unavailable' && (
        <div className="dialog__notice" role="alert">
          This browser has no GPU the app can use, so the benchmark can’t run. Smaller models still run on the CPU, more slowly.
        </div>
      )}

      <div className="dialog__actions">
        <button className="button button--ghost" type="button" onClick={onClose}>
          Skip for now
        </button>
        <button className="button button--primary" type="button" onClick={onRun} disabled={snap.status === 'unavailable'}>
          Run benchmark
        </button>
      </div>
    </dialog>
  );
}
