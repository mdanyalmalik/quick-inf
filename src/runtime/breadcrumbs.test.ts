import { describe, expect, it, vi } from 'vitest';
import { Breadcrumbs, interruptedDownload, killedForMemory, type Breadcrumb } from './breadcrumbs';

function memoryStorage() {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    key: (i: number) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  };
}

describe('Breadcrumbs', () => {
  it('finds a step that never ended after this tab reloads, once', () => {
    const local = memoryStorage();
    const tab = memoryStorage();
    new Breadcrumbs(local, tab).begin('load', 'qwen3-0.6b'); // the page dies here: nothing ends it
    const reloaded = new Breadcrumbs(local, tab);
    expect(reloaded.takeLeftovers()).toEqual([expect.objectContaining({ step: 'load', modelId: 'qwen3-0.6b' })]);
    expect(reloaded.takeLeftovers()).toEqual([]);
  });

  it('forgets steps that ended, however they ended', () => {
    const local = memoryStorage();
    const tab = memoryStorage();
    const end = new Breadcrumbs(local, tab).begin('image', 'sd-turbo');
    end();
    expect(new Breadcrumbs(local, tab).takeLeftovers()).toEqual([]);
  });

  it('leaves another tab that is still working alone, and reports one whose heartbeat stopped', () => {
    vi.useFakeTimers();
    const local = memoryStorage();
    let now = 1_000_000;
    new Breadcrumbs(local, memoryStorage(), () => now).begin('reply', 'smollm2-360m-instruct');
    const other = new Breadcrumbs(local, memoryStorage(), () => now);
    now += 10_000;
    expect(other.takeLeftovers()).toEqual([]);
    vi.clearAllTimers(); // the first tab is gone: its heartbeat stops
    now += 60_000;
    expect(other.takeLeftovers()).toEqual([expect.objectContaining({ step: 'reply' })]);
    vi.useRealTimers();
  });

  it('does nothing without storage (private windows)', () => {
    const crumbs = new Breadcrumbs(null, null);
    expect(() => crumbs.begin('benchmark')()).not.toThrow();
    expect(crumbs.takeLeftovers()).toEqual([]);
  });

  it('records how far a load got, and whether it was this tab', () => {
    const local = memoryStorage();
    const tab = memoryStorage();
    const crumbs = new Breadcrumbs(local, tab);
    crumbs.begin('load', 'qwen3-0.6b');
    crumbs.note({ phase: 'downloading' });
    expect(new Breadcrumbs(local, tab).takeLeftovers()).toEqual([
      expect.objectContaining({ step: 'load', modelId: 'qwen3-0.6b', phase: 'downloading', ownTab: true }),
    ]);
  });
});

describe('what a closed page means', () => {
  const crumb = (patch: Partial<Breadcrumb>): Breadcrumb => ({ step: 'load', modelId: 'm', at: 0, beat: 0, ...patch });

  it('counts a page closed while the model was going onto the GPU, or replying, as out of memory', () => {
    expect(killedForMemory(crumb({ phase: 'loading' }))).toBe(true);
    expect(killedForMemory(crumb({ step: 'reply' }))).toBe(true);
    expect(killedForMemory(crumb({ step: 'benchmark', modelId: undefined }))).toBe(true);
  });

  it('does not count one closed while downloading, or while in the background', () => {
    expect(killedForMemory(crumb({ phase: 'downloading' }))).toBe(false);
    expect(killedForMemory(crumb({}))).toBe(false); // closed before the load reported any progress
    expect(killedForMemory(crumb({ phase: 'loading', hidden: true }))).toBe(false);
    expect(killedForMemory(crumb({ step: 'image', hidden: true }))).toBe(false);
  });

  it('knows which interrupted loads can carry on downloading', () => {
    expect(interruptedDownload(crumb({ phase: 'downloading' }))).toBe(true);
    expect(interruptedDownload(crumb({ phase: 'verifying' }))).toBe(true);
    expect(interruptedDownload(crumb({ phase: 'loading' }))).toBe(false);
    expect(interruptedDownload(crumb({ step: 'reply' }))).toBe(false);
  });
});
