import type { DeviceTestResult } from '../../device/device-test';
import { speed, type Verdict } from '../../recommend/recommend';
import { recommendationStore, type RecommendationSnapshot } from '../../recommend/recommendation-store';
import { formatBytes } from '../common/format';
import { BackIcon, GaugeIcon } from '../common/icons';
import { RecommendationCard } from './RecommendationCard';
import { useRecommendations } from './use-recommendations';

const STEP_LABEL: Record<NonNullable<RecommendationSnapshot['running']>, string> = {
  starting: 'Starting…',
  bandwidth: 'Measuring memory speed…',
  compute: 'Measuring compute speed…',
  dispatch: 'Measuring per-step overhead…',
  memory: 'Checking GPU memory…',
};

/** The list in sections, best first; models that probably can't run here start folded away. */
const SECTIONS: { title: string; verdicts: Verdict[]; folded?: boolean }[] = [
  { title: 'Runs well here', verdicts: ['good'] },
  { title: 'Works, but slowly', verdicts: ['slow'] },
  { title: 'Not judged yet', verdicts: ['untested'] },
  { title: 'Probably won’t run well here', verdicts: ['may-not-fit', 'failed', 'unsupported'], folded: true },
];

interface RecommendationsPageProps {
  onOpenModel: (modelId: string) => void;
  /** Phone layout: back to the model list. */
  onBack?: () => void;
}

/** Device test results and every catalogue model's verdict, best first. */
export function RecommendationsPage({ onOpenModel, onBack }: RecommendationsPageProps) {
  const snap = useRecommendations();
  const { test, running } = snap;

  return (
    <div className="recs">
      <header className="recs__header">
        {onBack && (
          <button className="icon-button" type="button" onClick={onBack} aria-label="Back">
            <BackIcon />
          </button>
        )}
        <div className="recs__heading">
          <h1 className="recs__title">Device test and recommendations</h1>
          <p className="recs__sub muted">
            {test ? `${gpuName(test)} · tested ${new Date(test.ranAt).toLocaleString()}` : 'Not tested yet'}
          </p>
        </div>
        <button
          className="button button--primary"
          type="button"
          disabled={!!running || snap.status !== 'ready'}
          onClick={() => void recommendationStore.runDeviceTest()}
        >
          <GaugeIcon />
          {running ? 'Testing…' : test ? 'Run again' : 'Run device test'}
        </button>
      </header>

      <div className="recs__body">
        {running && (
          <p className="recs__progress" role="status">
            {STEP_LABEL[running]} <span className="muted">This takes a few seconds.</span>
          </p>
        )}
        {snap.error && (
          <p className="recs__error" role="alert">
            {snap.error}
          </p>
        )}
        {snap.status === 'unavailable' && (
          <p className="recs__error" role="alert">
            WebGPU isn’t available in this browser, so models can’t run here. Use a recent Chrome or Edge on a desktop or
            laptop.
          </p>
        )}

        {test ? (
          <>
            <DeviceSummary test={test} />
            {snap.memoryCheckOutdated && !running && (
              <p className="recs__note muted">
                Larger models were added after this test, so its memory check didn’t cover them. Run it again to check
                them too.
              </p>
            )}
          </>
        ) : (
          snap.status === 'ready' &&
          !running && (
            <p className="recs__intro muted">
              The device test runs a few small programs on your GPU for about five seconds to measure memory speed,
              compute speed and per-step overhead, and checks how much GPU memory it can hold. Nothing is downloaded. The
              results estimate how fast each model would run here.
            </p>
          )
        )}

        {SECTIONS.map(({ title, verdicts, folded }) => {
          // The recommended model first, then fastest first (measured where it has run here, else estimated).
          const recs = snap.recommendations
            .filter((r) => verdicts.includes(r.verdict))
            .sort((a, b) => Number(b.recommended) - Number(a.recommended) || speed(b) - speed(a));
          if (recs.length === 0) return null;
          const cards = (
            <div className="recs__list">
              {recs.map((rec) => (
                <RecommendationCard key={rec.model.id} rec={rec} onOpen={() => onOpenModel(rec.model.id)} />
              ))}
            </div>
          );
          return folded ? (
            <details key={title} className="recs__section">
              <summary className="recs__section-title">
                {title} ({recs.length})
              </summary>
              {cards}
            </details>
          ) : (
            <section key={title} className="recs__section">
              {test && <h2 className="recs__section-title">{title}</h2>}
              {cards}
            </section>
          );
        })}

        <p className="recs__footnote muted">
          Estimates come from the device test and each model’s size; once you use a model, its measured numbers replace
          them and also refine the estimates for the others. Speeds drop on battery saver, when the computer is hot, or
          while other apps use the GPU.
        </p>
      </div>
    </div>
  );
}

function DeviceSummary({ test }: { test: DeviceTestResult }) {
  const memory =
    test.memory.status === 'ok'
      ? `Held ${formatBytes(test.memory.testedBytes)}`
      : test.memory.status === 'failed'
        ? `Only ${formatBytes(test.memory.failedAtBytes)} of ${formatBytes(test.memory.testedBytes)}`
        : `Skipped (${test.memory.reason})`;
  return (
    <dl className="recs__device">
      <div title="How fast the GPU reads memory. Mostly decides generation speed.">
        <dt>Memory speed</dt>
        <dd>{Math.round(test.bandwidthGBps)} GB/s</dd>
      </div>
      <div title="Matrix maths throughput. Mostly decides how fast messages are read in.">
        <dt>Compute</dt>
        <dd>
          {Math.round(test.computeGflops)} GFLOP/s <span className="muted">{test.computePrecision}</span>
        </dd>
      </div>
      <div title="Fixed cost of each small GPU step. Dominates small models.">
        <dt>Per GPU step</dt>
        <dd>{test.dispatchUs.toFixed(1)} µs</dd>
      </div>
      <div title="Sending work to the GPU and reading a result back, as happens for every token.">
        <dt>Round trip</dt>
        <dd>{test.roundTripMs.toFixed(2)} ms</dd>
      </div>
      <div title="Whether the GPU could set aside as much memory as the largest model needs.">
        <dt>GPU memory check</dt>
        <dd>{memory}</dd>
      </div>
    </dl>
  );
}

function gpuName(test: DeviceTestResult): string {
  const { description, vendor, architecture } = test.gpu;
  return description || [vendor, architecture].filter(Boolean).join(' ') || 'This GPU';
}
