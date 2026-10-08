// Next-token selection from raw logits. Pure functions, unit tested.

export function argmax(logits: ArrayLike<number>): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i]!;
    if (v > bestValue) {
      bestValue = v;
      best = i;
    }
  }
  return best;
}

/**
 * Repetition penalty (as in Hugging Face / llama.cpp): every token id seen in
 * `recent` has its logit pushed away from being chosen, positive logits divided
 * by `penalty` and negative ones multiplied. Each id is penalised once,
 * however often it occurred. Modifies `logits` in place.
 */
export function applyRepetitionPenalty(logits: Float32Array, recent: Iterable<number>, penalty: number): void {
  if (penalty === 1) return;
  for (const id of new Set(recent)) {
    const v = logits[id];
    if (v === undefined) continue;
    logits[id] = v > 0 ? v / penalty : v * penalty;
  }
}

/** Deterministic PRNG (mulberry32) so sampled runs can be reproduced from a seed. */
export function createRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Temperature + nucleus (top-p) sampling. Temperature 0 is greedy.
 * Only tokens within ~30 logits of the maximum are considered: anything lower
 * has probability below e^-30 relative to the top token and cannot matter.
 */
export function sampleToken(
  logits: ArrayLike<number>,
  { temperature, topP }: { temperature: number; topP: number },
  random: () => number,
): number {
  if (temperature <= 0) return argmax(logits);

  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
  const cutoff = max - 30 * temperature;

  const candidates: { id: number; p: number }[] = [];
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i]!;
    if (v < cutoff) continue;
    const p = Math.exp((v - max) / temperature);
    candidates.push({ id: i, p });
    total += p;
  }
  candidates.sort((a, b) => b.p - a.p);

  // Keep the smallest prefix whose probability mass reaches topP.
  const limit = Math.min(Math.max(topP, 0), 1) * total;
  let kept = 0;
  let mass = 0;
  while (kept < candidates.length && (kept === 0 || mass < limit)) mass += candidates[kept++]!.p;

  let r = random() * mass;
  for (let i = 0; i < kept; i++) {
    r -= candidates[i]!.p;
    if (r <= 0) return candidates[i]!.id;
  }
  return candidates[kept - 1]!.id;
}
