// The device test page's image tab: each image model with how long an image takes here (measured once it has
// run, else estimated from the device test), its download, and whether this GPU can run it.

import { IMAGE_ARENA_SOURCE, IMAGE_MODELS, imageModelBytes, runsInApp, type ImageModelSpec } from '../../catalogue/image-models';
import type { DeviceTestResult } from '../../device/device-test';
import { imageVerdict, recommendedImage, type ImageVerdict } from '../../recommend/image-recommend';
import { defaultSteps, imageCorrection, imageSeconds, useImageSpeed } from '../../recommend/image-speed';
import { formatBytes } from '../common/format';
import { useStorage } from '../runtime/use-runtime';

const LABEL: Record<ImageVerdict, string> = {
  good: 'Likely runs well',
  slow: 'May be slow',
  unsupported: 'Not supported on this GPU',
  untested: 'Not tested yet',
  preview: 'Preview',
};

export function ImageRecommendations({ test, onOpenModel }: { test: DeviceTestResult | null; onOpenModel: (id: string) => void }) {
  const pick = recommendedImage(IMAGE_MODELS, test);
  // The recommended model first, then models that run (smallest first), previews last.
  const models = [...IMAGE_MODELS].sort(
    (a, b) =>
      Number(b.id === pick?.model.id) - Number(a.id === pick?.model.id) ||
      Number(!runsInApp(a)) - Number(!runsInApp(b)) ||
      a.parameterCount - b.parameterCount,
  );
  return (
    <div className="recs__list">
      {models.map((m) => (
        <ImageRecCard key={m.id} model={m} test={test} recommended={m.id === pick?.model.id} onOpen={() => onOpenModel(m.id)} />
      ))}
    </div>
  );
}

/** What to expect from one image model here. Without `onOpen` (the model's own page), no name or Open button. */
export function ImageRecCard({
  model,
  test,
  recommended = false,
  onOpen,
}: {
  model: ImageModelSpec;
  test: DeviceTestResult | null;
  recommended?: boolean;
  onOpen?: () => void;
}) {
  const speed = useImageSpeed(model.id);
  const storage = useStorage();
  const time = imageSeconds(model, speed, test, imageCorrection(IMAGE_MODELS, test));
  const verdict = imageVerdict(model, test, time?.seconds);
  const steps = defaultSteps(model);
  const saved = (storage.models[model.id]?.completeArtifacts.length ?? 0) > 0;
  return (
    <article className={`rec-card${recommended ? ' rec-card--recommended' : ''}`}>
      <header className="rec-card__header">
        {onOpen && <h2 className="rec-card__name">{model.name}</h2>}
        {recommended && <span className="badge">Recommended</span>}
        <span className={`verdict verdict--${verdict === 'preview' ? 'untested' : verdict}`}>{LABEL[verdict]}</span>
        {onOpen && (
          <button className="button button--ghost button--small rec-card__open" type="button" onClick={onOpen}>
            Open
          </button>
        )}
      </header>
      {onOpen && (
        <p className="rec-card__meta muted">
          {model.publisher} · {model.parameters} parameters · {model.sizeLabel}
        </p>
      )}
      {verdict === 'unsupported' && (
        <ul className="rec-card__reasons">
          <li>Needs GPU support for 16-bit maths ({(model.requiredGpuFeatures ?? []).join(', ')}), which this GPU lacks.</li>
        </ul>
      )}
      {runsInApp(model) && (
        <dl className="rec-card__stats">
          <div>
            <dt>Time per image</dt>
            <dd>
              {time ? `${time.seconds < 10 ? time.seconds.toFixed(1) : Math.round(time.seconds)} s` : '—'}
              {time && <span className="rec-card__source"> {time.source}</span>}
              {model.inputs.some((i) => i.id === 'steps') && (
                <span className="rec-card__source">
                  {' '}
                  · {steps} {steps === 1 ? 'step' : 'steps'}
                </span>
              )}
            </dd>
          </div>
          <div>
            <dt>{saved ? 'On this device' : 'Download'}</dt>
            <dd>{formatBytes(imageModelBytes(model))}</dd>
          </div>
        </dl>
      )}
      {onOpen && (
        <div className="rec-card__quality">
          <p>{model.summary}</p>
          {model.arena !== undefined && (
            <p className="rec-card__scores muted" title={`Source: ${IMAGE_ARENA_SOURCE}`}>
              Independent rating: people’s votes (Artificial Analysis image arena) {model.arena}
            </p>
          )}
          {!runsInApp(model) && <p className="muted">Can’t run in the app yet: its tab makes placeholder images.</p>}
        </div>
      )}
    </article>
  );
}
