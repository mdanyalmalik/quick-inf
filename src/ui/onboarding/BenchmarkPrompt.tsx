import { useEffect, useRef, useState } from 'react';
import { CloseIcon, GaugeIcon } from '../common/icons';

interface BenchmarkPromptProps {
  /** Whether this is the first-open welcome or a later re-run from the sidebar. */
  variant: 'welcome' | 'rerun';
  onClose: () => void;
}

// The benchmark itself is not implemented; "Run" only shows a placeholder state.
export function BenchmarkPrompt({ variant, onClose }: BenchmarkPromptProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [requested, setRequested] = useState(false);

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
        {variant === 'welcome' ? 'Welcome to Quick Inference' : 'Hardware benchmark'}
      </h2>
      <p className="dialog__text">
        Run a quick hardware check so Quick Inference can suggest models that suit this device. Everything runs locally in
        your browser.
      </p>

      {requested ? (
        <div className="dialog__notice" role="status">
          Benchmarking isn't available yet. You can run it later from the sidebar.
        </div>
      ) : null}

      <div className="dialog__actions">
        {requested ? (
          <button className="button button--primary" type="button" onClick={onClose}>
            Continue
          </button>
        ) : (
          <>
            <button className="button button--ghost" type="button" onClick={onClose}>
              {variant === 'welcome' ? 'Skip for now' : 'Cancel'}
            </button>
            <button className="button button--primary" type="button" onClick={() => setRequested(true)}>
              Run benchmark
            </button>
          </>
        )}
      </div>
    </dialog>
  );
}
