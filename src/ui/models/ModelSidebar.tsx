import type { ModelEntry, ModelSize } from '../types';
import { ChipIcon, DownloadIcon, GaugeIcon, MoonIcon, SaveIcon, SunIcon } from '../common/icons';
import { useChatBackup } from '../sessions/use-chat-backup';
import type { Theme } from '../common/use-theme';
import { ResizeHandle } from '../common/ResizeHandle';
import { useResizableWidth, type WidthBounds } from '../common/use-resizable-width';
import { modelRuntimeState, useRuntimeStatus } from '../runtime/use-runtime';
import { VerdictBadge } from '../recommend/RecommendationCard';
import { useRecommendations } from '../recommend/use-recommendations';
import type { Verdict } from '../../recommend/recommend';

const WIDTH: WidthBounds = { initial: 264, min: 200, max: 440 };

interface ModelSidebarProps {
  models: ModelEntry[];
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
  onOpenBenchmark: () => void;
  theme: Theme;
  onToggleTheme: () => void;
  /** Back to the start page (no model selected). */
  onHome: () => void;
}

export function ModelSidebar({
  models,
  selectedModelId,
  onSelectModel,
  onOpenBenchmark,
  theme,
  onToggleTheme,
  onHome,
}: ModelSidebarProps) {
  const onDevice = models.filter((m) => m.availability === 'on-device');
  const available = models.filter((m) => m.availability === 'available');
  const [width, setWidth] = useResizableWidth('quick-inf:models-width', WIDTH);
  const backup = useChatBackup();
  const saveLabel = backup.needsBackup ? 'Save chats (changed since the last backup)' : 'Save chats';

  return (
    <nav className="sidebar sidebar--models" aria-label="Models" style={{ width }}>
      <div className="sidebar__brand">
        <button type="button" className="brand-link" onClick={onHome} title="Home">
          <span className="brand-mark">
            <ChipIcon />
          </span>
          <span className="brand-text">
            <span className="brand-name">Quick Inference</span>
            <span className="brand-byline">by Danyal Malik</span>
          </span>
        </button>
      </div>

      <div className="sidebar__scroll">
        <ModelGroup title="On this device" models={onDevice} selectedModelId={selectedModelId} onSelect={onSelectModel} />
        {SIZE_GROUPS.map(({ size, title, hint }) => (
          <ModelGroup
            key={size}
            title={title}
            hint={hint}
            models={available.filter((m) => m.size === size)}
            selectedModelId={selectedModelId}
            onSelect={onSelectModel}
          />
        ))}
      </div>

      <div className="sidebar__footer">
        <button className="footer-button" type="button" onClick={onOpenBenchmark} title="Hardware benchmark">
          <GaugeIcon />
          <span className="footer-button__label">Benchmark</span>
        </button>
        <button
          className="icon-button backup-button"
          type="button"
          onClick={() => void backup.save()}
          disabled={backup.busy || backup.chatCount === 0}
          aria-label={saveLabel}
          title={backup.message?.kind === 'error' ? backup.message.text : saveLabel}
        >
          <SaveIcon />
          {backup.needsBackup && <span className="backup-button__dot" aria-hidden="true" />}
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      <ResizeHandle label="Resize models panel" width={width} bounds={WIDTH} onResize={setWidth} />
    </nav>
  );
}

const SIZE_GROUPS: { size: ModelSize; title: string; hint: string }[] = [
  { size: 'small', title: 'Small', hint: 'phones and any laptop' },
  { size: 'medium', title: 'Medium', hint: 'most laptops' },
  { size: 'large', title: 'Large', hint: '16 GB of memory or more' },
  { size: 'experimental', title: 'Experimental', hint: 'untested' },
];

interface ModelGroupProps {
  title: string;
  /** Who it suits, after the title. */
  hint?: string;
  models: ModelEntry[];
  selectedModelId: string | null;
  onSelect: (id: string) => void;
}

const PROBLEMS = new Set<Verdict>(['may-not-fit', 'failed', 'unsupported']);

const RUNTIME_LABEL = { loading: 'Loading…', ready: 'Loaded', busy: 'Running', failed: 'Failed' } as const;

function ModelGroup({ title, hint, models, selectedModelId, onSelect }: ModelGroupProps) {
  const status = useRuntimeStatus();
  const { recommendations } = useRecommendations();
  if (models.length === 0) return null;
  return (
    <section className="list-group">
      <h2 className="list-group__title">
        {title}
        {hint && <span className="list-group__hint"> · {hint}</span>}
      </h2>
      <ul className="list">
        {models.map((m) => {
          const runtime = modelRuntimeState(status, m.id);
          const rec = recommendations.find((r) => r.model.id === m.id);
          return (
            <li key={m.id}>
              <button
                type="button"
                className="model-item"
                title={m.name}
                aria-current={m.id === selectedModelId ? 'true' : undefined}
                onClick={() => onSelect(m.id)}
              >
                <span className="model-item__avatar" aria-hidden="true">
                  {m.publisher.charAt(0)}
                </span>
                <span className="model-item__text">
                  <span className="model-item__name">{m.name}</span>
                  <span className="model-item__meta">
                    {m.parameters} · {m.downloadSize}
                  </span>
                  {m.experimental && (
                    <span className="model-item__verdict">
                      <span className="badge badge--warning">Experimental</span>
                    </span>
                  )}
                  {/* Only the recommendation and real problems; every verdict is on the benchmark page. */}
                  {rec && !m.experimental && (rec.recommended || PROBLEMS.has(rec.verdict)) && (
                    <span className="model-item__verdict">
                      {rec.recommended && <span className="badge">Recommended</span>}
                      {PROBLEMS.has(rec.verdict) && <VerdictBadge verdict={rec.verdict} short />}
                    </span>
                  )}
                </span>
                {runtime === 'not-loaded' ? (
                  m.availability === 'available' && (
                    <span className="model-item__status" title="Not downloaded">
                      <DownloadIcon width={15} height={15} />
                    </span>
                  )
                ) : (
                  <span className={`runtime-pill runtime-pill--${runtime}`}>{RUNTIME_LABEL[runtime]}</span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
