// Which layers each device runs (info/distributed-plan.md): the host takes the first block (with the embedding) and
// the output layer, the other devices the blocks after it in the order they joined. Every device must fit its share
// (its layers' weights and KV cache at the planned context, plus a margin); among the plans that fit, the one with the
// least time per token wins: each device's layers' bytes over its bandwidth, plus a hop per device the step passes
// through (out to each used device and back to the host). A device can be left out (no layers). With 2–4 devices
// every plan is tried.

export interface PlanDevice {
  id: string;
  memoryBytes: number;
  bandwidthGBs: number;
}

export interface PlanInput {
  /** Weight bytes of each layer, in order. */
  layerBytes: number[];
  /** Weights only the host holds besides its layers: the embedding and the output layer. */
  hostBytes: number;
  /** KV cache bytes per layer at the planned context. */
  kvBytesPerLayer: number;
  /** Margin each device keeps for activations and the runtime. */
  overheadBytes: number;
  /** The host first, then the other devices in join order. */
  devices: PlanDevice[];
  /** Time one message takes between devices. */
  hopMs: number;
}

export interface PlanShare {
  id: string;
  /** Layers [from, to). */
  from: number;
  to: number;
  bytes: number;
  ms: number;
}

export interface Plan {
  /** Devices used, in order, with their layers. */
  shares: PlanShare[];
  /** Layer numbers where each device after the host starts (splitLayers' cuts, before the output layer's). */
  cuts: number[];
  msPerToken: number;
}

/** The best plan, or null if the devices can't hold the model together. */
export function planSplit(input: PlanInput): Plan | null {
  const { layerBytes, devices } = input;
  const layers = layerBytes.length;
  if (!devices.length || layers === 0) return null;
  const prefix = [0];
  for (const b of layerBytes) prefix.push(prefix.at(-1)! + b);
  let best: Plan | null = null;

  // Each device's block end, in device order; a device whose block is empty is left out.
  const ends: number[] = [];
  const consider = () => {
    const shares: PlanShare[] = [];
    let from = 0;
    for (const [d, to] of ends.entries()) {
      if (to === from) continue;
      const device = devices[d]!;
      const weights = prefix[to]! - prefix[from]!;
      const bytes = weights + (to - from) * input.kvBytesPerLayer + input.overheadBytes + (d === 0 ? input.hostBytes : 0);
      if (bytes > device.memoryBytes) return;
      shares.push({ id: device.id, from, to, bytes, ms: weights / (device.bandwidthGBs * 1e6) });
      from = to;
    }
    const hops = shares.length > 1 ? shares.length : 0;
    const msPerToken = shares.reduce((t, s) => t + s.ms, 0) + hops * input.hopMs;
    if (!best || msPerToken < best.msPerToken - 1e-9 || (Math.abs(msPerToken - best.msPerToken) < 1e-9 && shares.length < best.shares.length)) {
      best = { shares, cuts: shares.slice(1).map((s) => s.from), msPerToken };
    }
  };
  const choose = (d: number, from: number) => {
    if (d === devices.length - 1) {
      ends.push(layers);
      consider();
      ends.pop();
      return;
    }
    // The host runs at least one layer.
    for (let to = d === 0 ? Math.max(from, 1) : from; to <= layers; to++) {
      ends.push(to);
      choose(d + 1, to);
      ends.pop();
    }
  };
  if (devices.length === 1) {
    ends.push(layers);
    consider();
  } else choose(0, 0);
  return best;
}
