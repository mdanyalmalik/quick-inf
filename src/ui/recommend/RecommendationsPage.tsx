import { useBudget } from '../runtime/use-budget';
import { useState, type ReactNode } from 'react';
import type { DeviceTestResult, DeviceTestSoFar } from '../../device/device-test';
import { speed, type Recommendation, type Verdict } from '../../recommend/recommend';
import { recommendationStore, type RecommendationSnapshot } from '../../recommend/recommendation-store';
import { formatBytes } from '../common/format';
import { BackIcon, ChatIcon, GaugeIcon, ImageIcon } from '../common/icons';
import { IMAGE_MODELS } from '../../catalogue/image-models';
import { recommendedImage, type ImagePick } from '../../recommend/image-recommend';
import { ImageRecommendations } from './ImageRecommendations';
import { RecommendationCard } from './RecommendationCard';
import { useRecommendations } from './use-recommendations';

const STEP_LABEL: Record<NonNullable<RecommendationSnapshot['running']>, string> = {
  starting: 'Starting…',
  bandwidth: 'Measuring memory speed…',
  compute: 'Measuring compute speed…',
  dispatch: 'Measuring per-step overhead…',
  memory: 'Checking how much GPU memory this device can hold…',
};

/** Shown before and during the benchmark: the memory check holds memory until it runs out. */
const CRASH_WARNING =
  'The memory check fills GPU memory until it runs out, so this tab may crash. If it does, reload the tab or reopen the site: the results are kept.';

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
  const budget = useBudget();
  const bestImage = recommendedImage(IMAGE_MODELS, test, undefined, budget);

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
            {STEP_LABEL[running]} {running !== 'memory' && <span className="muted">This takes a few seconds.</span>}
          </p>
        )}
        {snap.status === 'ready' && (
          <p className="recs__warning" role={running === 'memory' ? 'alert' : undefined}>
            {CRASH_WARNING}
          </p>
        )}
        {snap.error && (
          <p className="recs__error" role="alert">
            {snap.error}
          </p>
        )}
        {snap.status === 'unavailable' && (
          <p className="recs__error" role="alert">
            This browser has no GPU the app can use, so the benchmark can’t run. Smaller models still run on the CPU, more slowly.
          </p>
        )}

        <h2 className="recs__part-title">This device</h2>
        {running ? (
          // Each result fills in as it's measured.
          <DeviceSummary values={snap.sofar ?? {}} memory={running === 'memory' ? liveMemory(snap.memoryHeldBytes ?? 0) : undefined} />
        ) : test ? (
          <>
            <DeviceSummary values={test} memory={memoryText(test.memory)} />
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
              per-step overhead, then checks how much GPU memory it can hold. Nothing is downloaded. The results estimate which models fit
              and how fast each would run here.
            </p>
          )
        )}

        {/* Nothing is judged or recommended before a benchmark. */}
        {test && (
          <>
            <h2 className="recs__part-title">Best picks for this device</h2>
            <ul className="recs__picks">
              <BestPick icon={<ChatIcon />} kind="Chat">
                {bestChat ? (
                  <button type="button" className="recs__pick-model" onClick={() => onOpenModel(bestChat.model.id)}>
                    {bestChat.model.name}
                    <span className="muted"> · {pickSpeed(bestChat)}</span>
                    {bestChat.pick && bestChat.pick !== 'runs-well' && <span className="recs__pick-note muted">{CHAT_PICK_NOTE[bestChat.pick]}</span>}
                  </button>
                ) : (
                  <span className="muted">{test ? 'No chat model can run on this device.' : 'Run the benchmark to get a pick.'}</span>
                )}
              </BestPick>
              <BestPick icon={<ImageIcon />} kind="Image generation">
                {bestImage ? (
                  <button type="button" className="recs__pick-model" onClick={() => onOpenModel(bestImage.model.id)}>
                    {bestImage.model.name}
                    {bestImage.time && <span className="muted"> · {imagePickTime(bestImage.time)}</span>}
                    {bestImage.why !== 'runs-well' && <span className="recs__pick-note muted">{IMAGE_PICK_NOTE[bestImage.why]}</span>}
                  </button>
                ) : (
                  <span className="muted">{test ? 'No image model can run on this device.' : 'Run the benchmark to get a pick.'}</span>
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
          </>
        )}

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
/** What a fallback pick means (a pick that runs well needs no note). */
const CHAT_PICK_NOTE: Record<Exclude<NonNullable<Recommendation['pick']>, 'runs-well'>, string> = {
  'least-slow': 'Nothing runs comfortably fast here; this is the best of the slower ones.',
  smallest: 'The smallest model; it may not fit in this device’s memory.',
};

const IMAGE_PICK_NOTE: Record<Exclude<ImagePick['why'], 'runs-well'>, string> = {
  quickest: 'Nothing makes an image in under 30 seconds here; this is the quickest.',
};

function imagePickTime(time: NonNullable<ImagePick['time']>): string {
  const seconds = time.seconds < 10 ? time.seconds.toFixed(1) : String(Math.round(time.seconds));
  return `${time.source === 'measured' ? '' : 'about '}${seconds} s per image`;
}

function pickSpeed(rec: Recommendation): string {
  const measured = rec.measured.tokensPerSecond;
  const tps = measured ?? rec.estimate?.tokensPerSecond;
  if (!tps) return 'recommended';
  return measured ? `${Math.round(tps)} tok/s measured` : `about ${Math.round(tps)} tok/s`;
}

function memoryText(memory: DeviceTestResult['memory']): string {
  if (memory.status === 'ok') return `Held ${formatBytes(memory.testedBytes)}`;
  if (memory.status === 'failed') return `${formatBytes(memory.failedAtBytes)}${MEMORY_STOP[memory.stoppedBy ?? 'out-of-memory']}`;
  return memory.reason === 'not checked' ? 'Not finished' : `Skipped (${memory.reason})`;
}

const liveMemory = (heldBytes: number) => `${formatBytes(heldBytes)} so far…`;

/** The device's results; one not measured yet (the benchmark is running) shows as pending. */
function DeviceSummary({ values, memory }: { values: DeviceTestSoFar; memory?: string }) {
  const show = (value: number | undefined, text: (v: number) => ReactNode) =>
    value === undefined ? <span className="muted">Measuring…</span> : text(value);
  return (
    <dl className="recs__device">
      <div title="How fast the GPU reads memory. Mostly decides generation speed.">
        <dt>Memory speed</dt>
        <dd>{show(values.bandwidthGBps, (v) => `${Math.round(v)} GB/s`)}</dd>
      </div>
      <div title="Matrix maths throughput. Mostly decides how fast input (a prompt, an image) is processed.">
        <dt>Compute</dt>
        <dd>
          {show(values.computeGflops, (v) => (
            <>
              {Math.round(v)} GFLOP/s <span className="muted">{values.computePrecision}</span>
            </>
          ))}
        </dd>
      </div>
      <div title="Fixed cost of each small GPU step. Dominates small models.">
        <dt>Per GPU step</dt>
        <dd>{show(values.dispatchUs, (v) => `${v.toFixed(1)} µs`)}</dd>
      </div>
      <div title="Sending work to the GPU and reading a result back, as happens for every token.">
        <dt>Round trip</dt>
        <dd>{show(values.roundTripMs, (v) => `${v.toFixed(2)} ms`)}</dd>
      </div>
      <div title="How much GPU memory the benchmark could hold before running out (or as much as the largest model needs).">
        <dt>GPU memory</dt>
        <dd>{memory ?? <span className="muted">Measured last</span>}</dd>
      </div>
    </dl>
  );
}

/** After the amount held: why the memory check stopped there. */
const MEMORY_STOP: Record<'out-of-memory' | 'slowdown' | 'page-closed', string> = {
  'out-of-memory': ', then ran out',
  slowdown: ', then the computer started running short of memory',
  'page-closed': ', then the browser closed the page',
};

function gpuName(test: DeviceTestResult): string {
  const { description, vendor, architecture } = test.gpu;
  return description || [vendor, architecture].filter(Boolean).join(' ') || 'This GPU';
}
