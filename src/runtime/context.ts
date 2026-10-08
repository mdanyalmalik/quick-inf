// Bounded context construction: keep the newest turns that fit the budget.
// Temporary home until the conversation service exists (milestone 4); the
// policy (drop oldest non-system turns whole) is deliberately simple.

import { RuntimeError, type ChatTurn } from './contract';

export interface ContextPlan {
  turns: ChatTurn[];
  droppedTurns: number;
}

/**
 * @param turnLength token cost of a turn including its formatting
 * @param fixedCost tokens always present (generation prompt and any default system turn)
 */
export function fitTurns(
  turns: readonly ChatTurn[],
  budget: number,
  turnLength: (turn: ChatTurn) => number,
  fixedCost: number,
): ContextPlan {
  const system = turns[0]?.role === 'system' ? turns[0] : undefined;
  const rest = system ? turns.slice(1) : turns.slice();
  let used = fixedCost + (system ? turnLength(system) : 0);

  const kept: ChatTurn[] = [];
  for (let i = rest.length - 1; i >= 0; i--) {
    const cost = turnLength(rest[i]!);
    if (used + cost > budget) break;
    used += cost;
    kept.unshift(rest[i]!);
  }

  const last = rest[rest.length - 1];
  if (last && kept[kept.length - 1] !== last) {
    throw new RuntimeError('context-overflow', 'The latest message is too long for the model’s context window.');
  }
  // Never start the kept history with an assistant turn: drop it so the model sees a user turn first.
  while (kept.length > 1 && kept[0]!.role === 'assistant') kept.shift();

  return { turns: system ? [system, ...kept] : kept, droppedTurns: rest.length - kept.length };
}
