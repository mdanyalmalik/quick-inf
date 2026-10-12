import { useState, type ReactNode } from 'react';
import type { DeviceTestResult } from '../../device/device-test';
import { speed, type Recommendation, type Verdict } from '../../recommend/recommend';
import { recommendationStore, type RecommendationSnapshot } from '../../recommend/recommendation-store';
import { formatBytes } from '../common/format';
import { BackIcon, ChatIcon, GaugeIcon, ImageIcon } from '../common/icons';
import { IMAGE_MODELS } from '../../catalogue/image-models';
import { bestImagePick, ImageRecommendations } from './ImageRecommendations';
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

/** Kinds of model, one tab each. */
export type RecommendationTab = 'chat' | 'image';
const TAB_KEY = 'quick-inf:recs-tab';

function readTab(fallback: RecommendationTab | undefined): RecommendationTab {
  if (fallback) return fallback;
  try {
    return localStorage.getItem(TAB_KEY) === 'image' ? 'image' : 'chat';
  } catch {
    return 'chat';
  }
}

interface RecommendationsPageProps {
  onOpenModel: (modelId: string) => void;
  /** Phone layout: back to the model list. */
  onBack?: () => void;
  /** The tab to open on (the kind of model the user came from); otherwise the last one viewed. */
  initialTab?: RecommendationTab;
}

/**
 * The device test (shared by every kind of model), the best pick per kind, then a tab per kind with each
 * catalogue model's verdict, best first.
 */
export function RecommendationsPage({ onOpenModel, onBack, initialTab }: RecommendationsPageProps) {
  const snap = useRecommendations();
  const { test, running } = snap;
  const [tab, setTabState] = useState<RecommendationTab>(() => readTab(initialTab));
  const setTab = (next: RecommendationTab) => {
    setTabState(next);
    try {
      localStorage.setItem(TAB_KEY, next);
    } catch {
      // Remembering the tab is a convenience.
    }
  };
  const bestChat = snap.recommendations.find((r) => r.recommended) ?? null;
  const bestImage = bestImagePick(test);

  return (
    <div className="recs">
      <header className="recs__header">
        {onBack && (
          <button className="icon-button" type="button" onClick={onBack} aria-label="Back">
            <BackIcon />
          </button>
        )}
        <div className="recs__heading">
          <h1 className="recs__title">Benchmark and recommendations</h1>
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
          {running ? 'Testing…' : test ? 'Run again' : 'Run benchmark'}
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
            WebGPU isn’t available in this browser, so models can’t run here. Use a recent Chrome or Edge on a desktop or laptop.
          </p>
        )}

        <h2 className="recs__part-title">This device</h2>
        {test ? (
          <>
            <DeviceSummary test={test} />
            {snap.memoryCheckOutdated && !running && (
              <p className="recs__note muted">
                Larger models were added after this test, so its memory check didn’t cover them. Run it again to check them too.
              </p>
            )}
          </>
        ) : (
          snap.status === 'ready' &&
          !running && (
            <p className="recs__intro muted">
              The benchmark runs a few small programs on your GPU for about five seconds to measure memory speed, compute speed and
              per-step overhead, and checks how much GPU memory it can hold. Nothing is downloaded. The results estimate how fast each model
              would run here.
            </p>
          )
        )}

        <h2 className="recs__part-title">Best picks for this device</h2>
        <ul className="recs__picks">
          <BestPick icon={<ChatIcon />} kind="Chat">
            {bestChat ? (
              <button type="button" className="recs__pick-model" onClick={() => onOpenModel(bestChat.model.id)}>
                {bestChat.model.name}
                <span className="muted"> · {pickSpeed(bestChat)}</span>
              </button>
            ) : (
              <span className="muted">{test ? 'Nothing runs well here yet.' : 'Run the benchmark to get a pick.'}</span>
            )}
          </BestPick>
          <BestPick icon={<ImageIcon />} kind="Image generation">
            {bestImage ? (
              <button type="button" className="recs__pick-model" onClick={() => onOpenModel(bestImage.model.id)}>
                {bestImage.model.name}
                <span className="muted">
                  {' '}
                  · {bestImage.source === 'measured' ? '' : 'about '}
                  {bestImage.seconds < 10 ? bestImage.seconds.toFixed(1) : Math.round(bestImage.seconds)} s per image
                </span>
              </button>
            ) : (
              <span className="muted">{test ? 'Nothing runs well here yet.' : 'Run the benchmark to get a pick.'}</span>
            )}
          </BestPick>
        </ul>

        <div
          className="recs__tabs"
          role="tablist"
          aria-label="Kinds of model"
          onKeyDown={(e) => {
            if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
            const next = tab === 'chat' ? 'image' : 'chat';
            setTab(next);
            document.getElementById(`recs-tab-${next}`)?.focus();
          }}
        >
          <TabButton
            id="chat"
            current={tab}
            onSelect={setTab}
            icon={<ChatIcon />}
            label="Chat"
            detail={String(snap.recommendations.length)}
          />
          <TabButton
            id="image"
            current={tab}
            onSelect={setTab}
            icon={<ImageIcon />}
            label="Image generation"
            detail={String(IMAGE_MODELS.length)}
          />
        </div>

        <div role="tabpanel" id={`recs-panel-${tab}`} aria-labelledby={`recs-tab-${tab}`} className="recs__panel">
          {tab === 'image' ? (
            <ImageRecommendations test={test} onOpenModel={onOpenModel} />
          ) : (
            SECTIONS.map(({ title, verdicts, folded }) => {
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
                  {test && <h3 className="recs__section-title">{title}</h3>}
                  {cards}
                </section>
              );
            })
          )}
        </div>

        <p className="recs__footnote muted">
          Estimates come from the benchmark and each model’s size; once you use a model, its measured numbers replace them and also refine
          the estimates for the others. Speeds drop on battery saver, when the computer is hot, or while other apps use the GPU.
        </p>
      </div>
    </div>
  );
}

function BestPick({ icon, kind, children }: { icon: ReactNode; kind: string; children: ReactNode }) {
  return (
    <li className="recs__pick">
      <span className="recs__pick-icon">{icon}</span>
      <span className="recs__pick-kind">{kind}</span>
      <span className="recs__pick-value">{children}</span>
    </li>
  );
}

function TabButton({
  id,
  current,
  onSelect,
  icon,
  label,
  detail,
}: {
  id: RecommendationTab;
  current: RecommendationTab;
  onSelect: (tab: RecommendationTab) => void;
  icon: ReactNode;
  label: string;
  detail: string;
}) {
  const selected = id === current;
  return (
    <button
      type="button"
      role="tab"
      id={`recs-tab-${id}`}
      aria-selected={selected}
      aria-controls={`recs-panel-${id}`}
      tabIndex={selected ? 0 : -1}
      className="recs__tab"
      onClick={() => onSelect(id)}
    >
      {icon}
      {label}
      <span className="recs__tab-detail">{detail}</span>
    </button>
  );
}

/** The best pick's headline speed: measured here if it has run, else estimated. */
function pickSpeed(rec: Recommendation): string {
  const measured = rec.measured.tokensPerSecond;
  const tps = measured ?? rec.estimate?.tokensPerSecond;
  if (!tps) return 'recommended';
  return measured ? `${Math.round(tps)} tok/s measured` : `about ${Math.round(tps)} tok/s`;
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
      <div title="Matrix maths throughput. Mostly decides how fast input (a prompt, an image) is processed.">
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
