// Detects that the browser killed the page while it was doing something risky. Mobile browsers end a tab that
// uses too much memory with no error the page can catch (iOS Safari at about 1.5 GB), and a desktop browser can
// lose the whole renderer the same way. So before each risky step (loading a model, a reply, an image, the
// benchmark) the page notes it in localStorage, keeps a heartbeat on it, and clears it when the step ends either
// way. A note still there when the app next starts — in this tab, or from a tab whose heartbeat stopped — means the
// page died during that step. The app then records it as a failure for that model on this device and says so,
// instead of trying the same thing again (as WebLLM journals its runs to resume them after a crash).

export type RiskyStep = 'load' | 'reply' | 'image' | 'benchmark';

export interface Breadcrumb {
  step: RiskyStep;
  modelId?: string;
  /** For a load: how far it got (a load phase, e.g. 'downloading' or 'loading'). */
  phase?: string;
  /** The page was in the background at the last note (phones close background pages to reclaim memory). */
  hidden?: boolean;
  /** Left by this tab (before a reload), not by another. Set when read back. */
  ownTab?: boolean;
  /** Memory the step was about to use, when known (the benchmark's memory check). */
  bytes?: number;
  /** When the step started. */
  at: number;
  /** Last heartbeat. */
  beat: number;
}

const PREFIX = 'quick-inf:in-progress:';
const TAB_KEY = 'quick-inf:tab';
const BEAT_MS = 5_000;
/** A note whose heartbeat is older than this belongs to a tab that is gone. */
const STALE_MS = 30_000;

type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem' | 'removeItem' | 'key' | 'length'>;

export class Breadcrumbs {
  private readonly tab: string;
  /** Distinguishes this page load's notes from those left by an earlier load of the same tab. */
  private readonly page = Math.random().toString(36).slice(2, 8);
  private readonly active = new Map<string, { crumb: Breadcrumb; timer: ReturnType<typeof setInterval>; write: () => void }>();
  private next = 1;

  constructor(
    private readonly store: Storage | null = safeStorage(() => localStorage),
    tabStore: Storage | null = safeStorage(() => sessionStorage),
    private readonly now: () => number = Date.now,
  ) {
    // The tab keeps its id across reloads (sessionStorage), so a reload after a kill finds its own note.
    let tab = tabStore?.getItem(TAB_KEY) ?? null;
    if (!tab) {
      tab = Math.random().toString(36).slice(2);
      try {
        tabStore?.setItem(TAB_KEY, tab);
      } catch {
        // Without a tab id, notes from this tab are only found once their heartbeat is stale.
      }
    }
    this.tab = tab;
  }

  /** Notes that a risky step is starting; call the returned function when it ends, however it ends. */
  begin(step: RiskyStep, modelId?: string, bytes?: number): () => void {
    if (!this.store) return () => {};
    const key = `${PREFIX}${this.tab}:${this.page}:${this.next++}`;
    const at = this.now();
    const crumb: Breadcrumb = { step, modelId, bytes, at, beat: at };
    const write = () => {
      try {
        this.store!.setItem(key, JSON.stringify({ ...crumb, hidden: pageHidden() || undefined, beat: this.now() }));
      } catch {
        // Storage full or blocked: the step still runs, it just can't be detected if it kills the page.
      }
    };
    write();
    const timer = setInterval(write, BEAT_MS);
    this.active.set(key, { crumb, timer, write });
    return () => {
      clearInterval(timer);
      this.active.delete(key);
      try {
        this.store!.removeItem(key);
      } catch {
        // Nothing more to do.
      }
    };
  }

  /** Adds to the notes of steps in progress (e.g. a load's phase), and records whether the page is in the background. */
  note(patch: Pick<Breadcrumb, 'phase'> = {}): void {
    for (const { crumb, write } of this.active.values()) {
      Object.assign(crumb, patch);
      write();
    }
  }

  /**
   * Notes left by steps that never ended: this tab's (from before a reload) and other tabs' that stopped beating.
   * Removes them, so each is reported once. Call once at start-up.
   */
  takeLeftovers(): Breadcrumb[] {
    if (!this.store) return [];
    const found: Breadcrumb[] = [];
    const keys: string[] = [];
    for (let i = 0; i < this.store.length; i++) {
      const key = this.store.key(i);
      if (key?.startsWith(PREFIX)) keys.push(key);
    }
    for (const key of keys) {
      if (this.active.has(key)) continue;
      const mine = key.startsWith(`${PREFIX}${this.tab}:`);
      let crumb: Breadcrumb | null = null;
      try {
        crumb = JSON.parse(this.store.getItem(key) ?? 'null') as Breadcrumb | null;
      } catch {
        crumb = null;
      }
      if (crumb && !mine && this.now() - crumb.beat < STALE_MS) continue; // another tab, still working
      try {
        this.store.removeItem(key);
      } catch {
        // Reported again next time; harmless.
      }
      if (crumb && typeof crumb.step === 'string') found.push({ ...crumb, ownTab: mine });
    }
    return found;
  }
}

/** Load phases before the model goes onto the GPU: a page closed then wasn't out of memory. */
const DOWNLOAD_PHASES = new Set(['checking', 'waiting', 'downloading', 'verifying']);

/**
 * Whether a leftover note means the page ran out of memory. Not when it was in the background (phones close
 * background pages anyway, and Safari's background limit is far lower) or still downloading.
 */
export function killedForMemory(crumb: Breadcrumb): boolean {
  return !crumb.hidden && !interruptedDownload(crumb);
}

/** A load closed by the browser while it was still downloading (resumable: the files are saved as they arrive). */
export function interruptedDownload(crumb: Breadcrumb): boolean {
  return crumb.step === 'load' && !!crumb.modelId && (crumb.phase === undefined || DOWNLOAD_PHASES.has(crumb.phase));
}

function pageHidden(): boolean {
  return typeof document !== 'undefined' && document.visibilityState === 'hidden';
}

function safeStorage(get: () => Storage): Storage | null {
  try {
    const s = get();
    return s ?? null;
  } catch {
    return null;
  }
}
