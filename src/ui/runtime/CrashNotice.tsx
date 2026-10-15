import { useSyncExternalStore } from 'react';
import { findImageModel } from '../../catalogue/image-models';
import { findModel } from '../../catalogue/models';
import { getRuntimeClient } from '../../runtime/client';
import type { Breadcrumb } from '../../runtime/breadcrumbs';

const STEP_TEXT: Record<Breadcrumb['step'], string> = {
  load: 'loading',
  reply: 'writing a reply with',
  image: 'making an image with',
  benchmark: 'running the benchmark',
};

/**
 * Shown after the browser closed the page during a risky step on the last visit (see runtime/breadcrumbs.ts),
 * which is almost always running out of memory. The app has already recorded it, so it won't suggest the same.
 */
export function CrashNotice() {
  const client = getRuntimeClient();
  const crashes = useSyncExternalStore(client.subscribe, client.getCrashes);
  if (crashes.length === 0) return null;
  const crash = crashes[crashes.length - 1]!;
  const name = crash.modelId ? (findModel(crash.modelId)?.name ?? findImageModel(crash.modelId)?.name ?? crash.modelId) : null;
  const what = crash.step === 'benchmark' ? STEP_TEXT.benchmark : `${STEP_TEXT[crash.step]} ${name ?? 'a model'}`;
  return (
    <div className="crash-notice" role="alert">
      <span>
        <strong>The page closed while {what}.</strong> The browser most likely ran out of memory.{' '}
        {crash.step === 'benchmark'
          ? 'The next benchmark checks less memory.'
          : `${name ?? 'That model'} is now marked as probably too big for this device; smaller models and settings are suggested instead.`}
      </span>
      <button className="button button--ghost button--small" type="button" onClick={() => client.dismissCrashes()}>
        Dismiss
      </button>
    </div>
  );
}
