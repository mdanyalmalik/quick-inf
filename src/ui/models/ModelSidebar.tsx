import { useBudget } from '../runtime/use-budget';
import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import type { ModelEntry, ModelSize } from '../types';
import {
  ChatIcon,
  ChevronIcon,
  ChipIcon,
  DeviceIcon,
  DownloadIcon,
  GaugeIcon,
  ImageIcon,
  MoonIcon,
  SaveIcon,
  SearchIcon,
  SunIcon,
} from '../common/icons';
import { matchesSearch } from './model-search';
import { useChatBackup } from '../sessions/use-chat-backup';
import type { Theme } from '../common/use-theme';
import { ResizeHandle } from '../common/ResizeHandle';
import { useResizableWidth, type WidthBounds } from '../common/use-resizable-width';
import { modelRuntimeState, useRuntimeStatus, useStorageKnown } from '../runtime/use-runtime';
import { VerdictBadge } from '../recommend/RecommendationCard';
import { useRecommendations } from '../recommend/use-recommendations';
import type { Verdict } from '../../recommend/recommend';
import { imageVerdict, recommendedImage } from '../../recommend/image-recommend';
import { findImageModel, IMAGE_MODELS } from '../../catalogue/image-models';

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
  /** A model opened from a link: open the sections it's listed in and scroll to it, then call `onRevealed`. */
  revealModelId?: string | null;
  onRevealed?: () => void;
}

export function ModelSidebar({
  models,
  selectedModelId,
  onSelectModel,
  onOpenBenchmark,
  theme,
  onToggleTheme,
  onHome,
  revealModelId = null,
  onRevealed,
}: ModelSidebarProps) {
  const [query, setQuery] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);
  const searching = query.trim() !== '';
  const matching = models.filter((m) => matchesSearch(m, query));
  useRevealFirstSaved(models);
  useRevealLinked(models, revealModelId, onRevealed);
  const chat = matching.filter((m) => m.task === 'chat');
  const image = matching.filter((m) => m.task === 'image');

  // "/" focuses the search, unless typing somewhere else.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey || target?.closest('input, textarea, [contenteditable="true"]')) return;
      e.preventDefault();
      searchRef.current?.focus();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const [width, setWidth] = useResizableWidth('quick-inf:models-width', WIDTH);
  const backup = useChatBackup();
  const saveLabel = backup.needsBackup ? 'Back up your work (changed since the last backup)' : 'Back up your work';

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

      <div className="model-search">
        <SearchIcon width={15} height={15} />
        <input
          ref={searchRef}
          className="model-search__input"
          type="search"
          placeholder="Search models"
          aria-label="Search models"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              setQuery('');
              e.currentTarget.blur();
            }
          }}
        />
        {!searching && (
          <kbd className="model-search__key" aria-hidden="true">
            /
          </kbd>
        )}
      </div>

      <div className="sidebar__scroll">
        {searching && chat.length === 0 && image.length === 0 && <p className="sidebar__empty muted">No models match “{query.trim()}”.</p>}
        {(!searching || chat.length > 0) && (
          <TaskSection id="chat" title="Chat" icon={<ChatIcon />} count={chat.length} forceOpen={searching}>
            <TaskModels id="chat" models={chat} bySize searching={searching} selectedModelId={selectedModelId} onSelect={onSelectModel} />
          </TaskSection>
        )}
        {(!searching || image.length > 0) && (
          <TaskSection id="image" title="Image generation" icon={<ImageIcon />} count={image.length} forceOpen={searching}>
            <TaskModels id="image" models={image} bySize searching={searching} selectedModelId={selectedModelId} onSelect={onSelectModel} />
          </TaskSection>
        )}
      </div>

      <div className="sidebar__footer">
        <button className="footer-button" type="button" onClick={onOpenBenchmark} title="Benchmark and recommendations">
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

/** A collapsible group of models of one kind (chat, image generation). */
function TaskSection({
  id,
  title,
  icon,
  count,
  forceOpen,
  children,
}: {
  id: string;
  title: string;
  icon: ReactNode;
  count: number;
  forceOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <Disclosure
      forceOpen={forceOpen}
      id={id}
      level="task"
      className="task-section"
      header={
        <>
          <span className="task-section__icon">{icon}</span>
          <span className="disclosure__label task-section__title">{title}</span>
          <span className="disclosure__count">{count}</span>
        </>
      }
    >
      {children}
    </Disclosure>
  );
}

/**
 * One kind's models: those saved on this device, then those available to download (by size when `bySize`,
 * with the same tiers for every kind; a tier with no models isn't shown).
 */
function TaskModels({
  id,
  models,
  bySize = false,
  searching,
  selectedModelId,
  onSelect,
}: {
  id: string;
  models: ModelEntry[];
  bySize?: boolean;
  searching: boolean;
  selectedModelId: string | null;
  onSelect: (id: string) => void;
}) {
  const onDevice = models.filter((m) => m.availability === 'on-device');
  const available = models.filter((m) => m.availability === 'available');
  return (
    <>
      {(!searching || onDevice.length > 0) && (
        <Disclosure
          forceOpen={searching}
          id={`${id}:device`}
          level="availability"
          className="availability availability--device"
          header={
            <>
              <DeviceIcon width={14} height={14} />
              <span className="disclosure__label">On this device</span>
              <span className="disclosure__count">{onDevice.length}</span>
            </>
          }
        >
          {onDevice.length === 0 ? (
            <p className="availability__empty">Nothing downloaded yet. Pick a model below; it stays here for offline use.</p>
          ) : (
            <ModelGroup models={onDevice} selectedModelId={selectedModelId} onSelect={onSelect} />
          )}
        </Disclosure>
      )}
      {(!searching || available.length > 0) && (
        <Disclosure
          forceOpen={searching}
          id={`${id}:available`}
          level="availability"
          className="availability"
          header={
            <>
              <DownloadIcon width={14} height={14} />
              <span className="disclosure__label">Available to download</span>
              <span className="disclosure__count">{available.length}</span>
            </>
          }
        >
          {!bySize ? (
            <ModelGroup models={available} selectedModelId={selectedModelId} onSelect={onSelect} />
          ) : (
            SIZE_GROUPS.map(({ size, title }) => {
              const group = available.filter((m) => m.size === size);
              if (group.length === 0) return null;
              return (
                <Disclosure
                  forceOpen={searching}
                  key={size}
                  id={`${id}:available:${size}`}
                  level="size"
                  className="size-group"
                  header={
                    <>
                      <span className="disclosure__label">
                        <span className="size-group__title">{title}</span>
                      </span>
                      <span className="disclosure__count">{group.length}</span>
                    </>
                  }
                >
                  <ModelGroup models={group} selectedModelId={selectedModelId} onSelect={onSelect} />
                </Disclosure>
              );
            })
          )}
        </Disclosure>
      )}
    </>
  );
}

/**
 * Which sections are open, remembered per browser. All start closed, so a first visit shows every kind of model
 * (chat, image generation) at a glance.
 */
const sections = (() => {
  const key = (id: string) => `quick-inf:models-section:${id}`;
  const listeners = new Set<() => void>();
  return {
    isOpen(id: string): boolean {
      try {
        return localStorage.getItem(key(id)) === 'open';
      } catch {
        return false;
      }
    },
    set(id: string, open: boolean): void {
      try {
        localStorage.setItem(key(id), open ? 'open' : 'closed');
      } catch {
        // Remembering is a convenience.
      }
      for (const l of listeners) l();
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
})();

/**
 * The first time a kind of model has one on this device (its first download, once loaded), opens that kind and
 * its "On this device" list so it's easy to find again. Once only, so closing them later sticks.
 */
function useRevealFirstSaved(models: ModelEntry[]): void {
  const status = useRuntimeStatus();
  useEffect(() => {
    if (status.state !== 'ready' || !status.modelId) return;
    const model = models.find((m) => m.id === status.modelId);
    if (!model || model.availability !== 'on-device') return;
    const flag = `quick-inf:models-revealed:${model.task}`;
    try {
      if (localStorage.getItem(flag)) return;
      localStorage.setItem(flag, '1');
    } catch {
      return;
    }
    sections.set(model.task, true);
    sections.set(`${model.task}:device`, true);
  }, [status.state, status.modelId, models]);
}

/** How long to wait for the saved-models report before opening a linked model's sections anyway. */
const STORAGE_WAIT_MS = 2000;

/**
 * Opens the sections a linked model is listed in (its kind, "On this device" or "Available to download", and its
 * size group), as if they'd been opened by hand, and scrolls it into view. Waits for the saved-models report (until
 * then every model looks not downloaded), or a moment if it doesn't come.
 */
function useRevealLinked(models: ModelEntry[], modelId: string | null, onRevealed?: () => void): void {
  const storageKnown = useStorageKnown();
  const [waited, setWaited] = useState(false);
  useEffect(() => {
    if (!modelId || storageKnown) return;
    const timer = setTimeout(() => setWaited(true), STORAGE_WAIT_MS);
    return () => clearTimeout(timer);
  }, [modelId, storageKnown]);
  useEffect(() => {
    if (!modelId || !(storageKnown || waited)) return;
    const model = models.find((m) => m.id === modelId);
    if (model) {
      sections.set(model.task, true);
      if (model.availability === 'on-device') {
        sections.set(`${model.task}:device`, true);
      } else {
        sections.set(`${model.task}:available`, true);
        sections.set(`${model.task}:available:${model.size}`, true);
      }
      // After the opened sections render.
      requestAnimationFrame(() =>
        document.querySelector(`.sidebar--models .model-item[aria-current="true"]`)?.scrollIntoView({ block: 'center' }),
      );
    }
    onRevealed?.();
  }, [models, modelId, storageKnown, waited, onRevealed]);
}

/**
 * A heading that opens and closes the content under it (nested: model type › on this device / available ›
 * size). All start closed; open or closed is remembered per browser.
 */
function Disclosure({
  id,
  level,
  className,
  header,
  forceOpen = false,
  children,
}: {
  id: string;
  level: 'task' | 'availability' | 'size';
  className: string;
  header: ReactNode;
  /** Shown open regardless (while searching), without changing what's remembered. */
  forceOpen?: boolean;
  children: ReactNode;
}) {
  const remembered = useSyncExternalStore(sections.subscribe, () => sections.isOpen(id));
  const open = forceOpen || remembered;
  const toggle = () => {
    if (!forceOpen) sections.set(id, !remembered);
  };
  return (
    <section className={`disclosure disclosure--${level} ${className}${open ? ' disclosure--open' : ''}`}>
      <button type="button" className="disclosure__header" aria-expanded={open} onClick={toggle}>
        <ChevronIcon className="disclosure__chevron" width={level === 'task' ? 14 : 12} height={level === 'task' ? 14 : 12} />
        {header}
      </button>
      {open && <div className="disclosure__body">{children}</div>}
    </section>
  );
}

const SIZE_GROUPS: { size: ModelSize; title: string }[] = [
  // By download size (see model-entries.ts).
  { size: 'small', title: 'Small' },
  { size: 'medium', title: 'Medium' },
  { size: 'large', title: 'Large' },
  { size: 'xlarge', title: 'Very large' },
  { size: 'experimental', title: 'Experimental' },
];

interface ModelGroupProps {
  models: ModelEntry[];
  selectedModelId: string | null;
  onSelect: (id: string) => void;
}

const PROBLEMS = new Set<Verdict>(['may-not-fit', 'failed', 'unsupported']);

const RUNTIME_LABEL = {
  loading: 'Loading…',
  ready: 'Loaded',
  busy: 'Running',
  failed: 'Failed',
} as const;

function ModelGroup({ models, selectedModelId, onSelect }: ModelGroupProps) {
  const status = useRuntimeStatus();
  const { recommendations, test } = useRecommendations();
  // One recommended model per kind: chat's comes with the recommendations, image generation's from its own pick.
  const budget = useBudget();
  const imagePick = recommendedImage(IMAGE_MODELS, test, undefined, budget?.bytes)?.model.id;
  if (models.length === 0) return null;
  return (
    <section className="list-group">
      <ul className="list">
        {models.map((m) => {
          const runtime = modelRuntimeState(status, m.id);
          const rec = recommendations.find((r) => r.model.id === m.id);
          const recommended = rec?.recommended || (m.task === 'image' && m.id === imagePick);
          // Image models' problems (chat models' come with their recommendation); nothing before a benchmark.
          const imageSpec = m.task === 'image' ? findImageModel(m.id) : undefined;
          const imageProblem = imageSpec && test ? imageVerdict(imageSpec, test, undefined) : undefined;
          const problem: Verdict | undefined =
            rec && PROBLEMS.has(rec.verdict) ? rec.verdict : imageProblem === 'may-not-fit' || imageProblem === 'unsupported' ? imageProblem : undefined;
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
                    {m.parameters} · {m.preview ? 'preview' : m.availability === 'on-device' ? `saved, ${m.downloadSize}` : m.downloadSize}
                    {m.reasoning && ' · thinks first'}
                  </span>
                  {m.experimental && (
                    <span className="model-item__verdict">
                      <span className="badge badge--warning">Experimental</span>
                    </span>
                  )}
                  {/* Only the recommendation and real problems; every verdict is on the benchmark page. */}
                  {!m.experimental && (recommended || problem) && (
                    <span className="model-item__verdict">
                      {recommended && <span className="badge">Recommended</span>}
                      {problem && <VerdictBadge verdict={problem} short />}
                    </span>
                  )}
                </span>
                {runtime === 'not-loaded' ? (
                  m.availability === 'available' &&
                  !m.preview && (
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
