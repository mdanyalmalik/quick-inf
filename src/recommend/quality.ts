// How good each model is, as one score that orders recommendations among the models that run well.
//
// Chat models: a blend of human preference and academic benchmarks, in LMArena rating points. Human preference is
// the model's LMArena rating (`quality.arena`); academic benchmarks are the Artificial Analysis Intelligence Index
// (`quality.intelligence`), turned into rating points by a straight line fitted to the models that have both. A model
// with both gets their mean; with one, that one. Below INTELLIGENCE_FLOOR the index no longer separates models (the
// smallest ones all score 5–6 on it), so it's left out there.
//
// Image models: human preference only, as the Artificial Analysis text-to-image arena Elo (`arena`).
//
// Models with no outside score keep their hand-set rank, put on the same scale by a straight line fitted to the
// models that have both (so a model's place among its neighbours stays as judged).

import type { ImageModelSpec } from '../catalogue/image-models';
import type { ModelSpec } from '../catalogue/models';

/** Below this, the Intelligence Index doesn't tell models apart. */
export const INTELLIGENCE_FLOOR = 8;

export interface QualityScore {
  /** In rating points of the model's kind (LMArena for chat, Artificial Analysis arena Elo for images). */
  score: number;
  /** What the score is based on. */
  basis: 'arena+benchmarks' | 'arena' | 'benchmarks' | 'estimate';
}

interface Line {
  slope: number;
  intercept: number;
}

/** Least-squares line y = slope·x + intercept (null with fewer than two distinct x). */
function fit(points: readonly [number, number][]): Line | null {
  if (points.length < 2) return null;
  const n = points.length;
  const mx = points.reduce((s, [x]) => s + x, 0) / n;
  const my = points.reduce((s, [, y]) => s + y, 0) / n;
  const sxx = points.reduce((s, [x]) => s + (x - mx) ** 2, 0);
  if (sxx === 0) return null;
  const slope = points.reduce((s, [x, y]) => s + (x - mx) * (y - my), 0) / sxx;
  return { slope, intercept: my - slope * mx };
}

const at = (line: Line, x: number) => line.slope * x + line.intercept;

/** Scores for every chat model, by id. */
export function chatQuality(models: readonly ModelSpec[]): Map<string, QualityScore> {
  const intelligence = (m: ModelSpec) => {
    const value = m.quality.intelligence;
    return value !== undefined && value >= INTELLIGENCE_FLOOR ? value : undefined;
  };
  const toArena = fit(
    models.flatMap((m) => {
      const i = intelligence(m);
      return m.quality.arena !== undefined && i !== undefined ? [[i, m.quality.arena] as [number, number]] : [];
    }),
  );
  const scored = new Map<string, QualityScore>();
  for (const m of models) {
    const arena = m.quality.arena;
    const i = intelligence(m);
    const benchmarks = i !== undefined && toArena ? at(toArena, i) : undefined;
    if (arena !== undefined && benchmarks !== undefined) scored.set(m.id, { score: (arena + benchmarks) / 2, basis: 'arena+benchmarks' });
    else if (arena !== undefined) scored.set(m.id, { score: arena, basis: 'arena' });
    else if (benchmarks !== undefined) scored.set(m.id, { score: benchmarks, basis: 'benchmarks' });
  }
  return withEstimates(models, scored, (m) => m.quality.rank);
}

/** Scores for every image model, by id. */
export function imageQuality(models: readonly ImageModelSpec[]): Map<string, QualityScore> {
  const scored = new Map<string, QualityScore>();
  for (const m of models) if (m.arena !== undefined) scored.set(m.id, { score: m.arena, basis: 'arena' });
  return withEstimates(models, scored, (m) => m.rank);
}

/** The rest from their hand rank, on the scale fitted to the scored models. */
function withEstimates<M extends { id: string }>(
  models: readonly M[],
  scored: Map<string, QualityScore>,
  rank: (m: M) => number,
): Map<string, QualityScore> {
  const line = fit(models.flatMap((m) => (scored.has(m.id) ? [[rank(m), scored.get(m.id)!.score] as [number, number]] : [])));
  for (const m of models) if (!scored.has(m.id)) scored.set(m.id, { score: line ? at(line, rank(m)) : rank(m), basis: 'estimate' });
  return scored;
}
