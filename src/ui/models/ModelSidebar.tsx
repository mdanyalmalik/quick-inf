import type { ModelEntry } from '../types';
import { ChipIcon, DownloadIcon, GaugeIcon, MoonIcon, SunIcon } from '../common/icons';
import type { Theme } from '../common/use-theme';
import { ResizeHandle } from '../common/ResizeHandle';
import { useResizableWidth, type WidthBounds } from '../common/use-resizable-width';
import { modelRuntimeState, useRuntimeStatus } from '../runtime/use-runtime';

const WIDTH: WidthBounds = { initial: 264, min: 200, max: 440 };

interface ModelSidebarProps {
  models: ModelEntry[];
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
  onOpenBenchmark: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export function ModelSidebar({
  models,
  selectedModelId,
  onSelectModel,
  onOpenBenchmark,
  theme,
  onToggleTheme,
}: ModelSidebarProps) {
  const onDevice = models.filter((m) => m.availability === 'on-device');
  const available = models.filter((m) => m.availability === 'available');
  const [width, setWidth] = useResizableWidth('quick-inf:models-width', WIDTH);

  return (
    <nav className="sidebar sidebar--models" aria-label="Models" style={{ width }}>
      <div className="sidebar__brand">
        <span className="brand-mark">
          <ChipIcon />
        </span>
        <span className="brand-name">Quick Inference</span>
      </div>

      <div className="sidebar__scroll">
        <ModelGroup title="On this device" models={onDevice} selectedModelId={selectedModelId} onSelect={onSelectModel} />
        <ModelGroup title="Available" models={available} selectedModelId={selectedModelId} onSelect={onSelectModel} />
      </div>

      <div className="sidebar__footer">
        <button className="footer-button" type="button" onClick={onOpenBenchmark}>
          <GaugeIcon />
          <span className="footer-button__label">Hardware benchmark</span>
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

interface ModelGroupProps {
  title: string;
  models: ModelEntry[];
  selectedModelId: string | null;
  onSelect: (id: string) => void;
}

const RUNTIME_LABEL = { loading: 'Loading…', ready: 'Loaded', busy: 'Running', failed: 'Failed' } as const;

function ModelGroup({ title, models, selectedModelId, onSelect }: ModelGroupProps) {
  const status = useRuntimeStatus();
  if (models.length === 0) return null;
  return (
    <section className="list-group">
      <h2 className="list-group__title">{title}</h2>
      <ul className="list">
        {models.map((m) => {
          const runtime = modelRuntimeState(status, m.id);
          return (
            <li key={m.id}>
              <button
                type="button"
                className="model-item"
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
                    {m.recommended && <span className="badge">Recommended</span>}
                  </span>
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
