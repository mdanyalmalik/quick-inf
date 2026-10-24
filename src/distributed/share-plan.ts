// The host's side of loading a model shared across devices (info/distributed-plan.md): planning who runs which layers
// from the model's prepared graph, and the adapter options that load the host's segments here and the others' on
// their devices. Used by the runtime worker and the development page.

import type { ModelSpec, OnnxArtifact } from '../catalogue/models';
import type { LoadedArtifacts } from '../runtime/contract';
import { preparedLayerSplit, type LoadOptions } from '../runtime/onnx/onnx-adapter';
import { getOrt } from '../runtime/onnx/ort-env';
import { planSplit, type Plan, type PlanDevice } from './planner';
import { createRemoteChain, type PipelineLink } from './remote-stage';

const GB = 1e9;
/** Memory each device keeps for activations and the runtime, besides its weights and cache. */
const OVERHEAD_BYTES = 0.25 * GB;

export interface SharePlan {
  plan: Plan;
  /** splitLayers' cuts: the host's layers, each device's, then the output layer (back on the host). */
  cuts: number[];
  /** What the plan was made from, for the log. */
  summary: string;
}

/**
 * The best split of `model` across `devices` (the host first) at `contextTokens`, or null if they can't hold it
 * together. Each layer's weights come from the prepared graph cut before every layer; weights several layers use
 * (rotary tables) count once per device; the host also holds the embedding (with its first layer) and the output
 * layer, each in a session of its own.
 */
export function planShare(
  model: ModelSpec,
  artifact: OnnxArtifact,
  files: Pick<LoadedArtifacts, 'graph' | 'externalData'>,
  devices: PlanDevice[],
  contextTokens: number,
  hopMs: number,
): SharePlan | null {
  const layers = model.architecture.layers;
  const perLayer = preparedLayerSplit(artifact, files, Array.from({ length: layers }, (_, i) => i + 1)).segments;
  const users = new Map<string, number>();
  for (const s of perLayer.slice(1, layers)) for (const name of s.initializers.keys()) users.set(name, (users.get(name) ?? 0) + 1);
  const shared = new Set([...users].filter(([, n]) => n > 1).map(([name]) => name));
  const own = (s: (typeof perLayer)[number]) => [...s.initializers].reduce((n, [name, bytes]) => n + (shared.has(name) ? 0 : bytes), 0);
  const sharedBytes = [...perLayer[1]!.initializers].reduce((n, [name, bytes]) => n + (shared.has(name) ? bytes : 0), 0);
  const layerBytes = perLayer.slice(0, layers).map(own);
  const typical = [...layerBytes].sort((a, b) => a - b)[Math.floor(layers / 2)]!;
  const hostBytes = layerBytes[0]! - typical + own(perLayer[layers]!);
  layerBytes[0] = typical;
  const { kvHeads, headDim } = model.architecture;
  const kvBytesPerLayer = 2 * kvHeads * headDim * 2 * contextTokens;
  const plan = planSplit({ layerBytes, hostBytes, kvBytesPerLayer, overheadBytes: OVERHEAD_BYTES + sharedBytes, devices, hopMs });
  const summary =
    `${(layerBytes.reduce((a, b) => a + b, 0) / GB).toFixed(2)} GB of weights over ${layers} layers, ${(sharedBytes / 1e6).toFixed(0)} MB shared by them, ` +
    `the host's extra ${(hostBytes / GB).toFixed(2)} GB, KV ${(kvBytesPerLayer / 1e6).toFixed(0)} MB per layer at ${contextTokens} tokens`;
  return plan && { plan, cuts: [...plan.cuts, layers], summary };
}

/** The adapter's remoteStage: the devices' segments (all between the host's two) as one chain. */
export function shareStages(link: PipelineLink, model: ModelSpec, { plan, cuts }: SharePlan): NonNullable<LoadOptions['remoteStage']> {
  // Segment i is plan.shares[i]'s; the output layer's, after them, is the host's.
  const owner = (index: number) => (index === 0 || index === cuts.length ? null : plan.shares[index]!.id);
  return (split, index, types, sources) => {
    const peers: string[] = [];
    for (let i = index; owner(i); i++) peers.push(owner(i)!);
    if (!peers.length) return null;
    const layers = peers.map((_, i) => [plan.shares[index + i]!.from, plan.shares[index + i]!.to] as [number, number]);
    return createRemoteChain({ link, peers, model, split, sources, index, layers, types, Tensor: getOrt().Tensor });
  };
}

/** "host layers 0–10 (1.93 GB), phone layers 11–27 (1.16 GB)". */
export function describePlan(plan: Plan, label: (id: string) => string = (id) => id): string {
  return plan.shares.map((s) => `${label(s.id)} layers ${s.from}–${s.to - 1} (${(s.bytes / GB).toFixed(2)} GB)`).join(', ');
}
