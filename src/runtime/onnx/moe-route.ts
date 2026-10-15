// Routing for streamed experts (see moe-split.ts): which experts each token uses, and how to feed QMoE only those.
//
// QMoE routes from its router input as follows: softmax over all experts, the top k, then (with
// normalize_routing_weights) those k weights scaled to sum to 1. That is recomputed here on the CPU, from the router
// values read back from the GPU, so the runtime knows which experts to load before QMoE runs.
//
// QMoE is then fed only a block of chosen experts (its "slots"). A prompt chunk's tokens can together choose more
// experts than fit at once, so the slots come in groups, each run separately and summed. For QMoE to weight each
// group's experts exactly as the full routing does, its router input for a group is each token's final weight w as
// log(w) for the token's experts in the group, and one extra "rest" slot with log of the token's weight outside the
// group; every other slot gets −∞ (in practice the float16 minimum). QMoE's softmax over that row gives w for the
// group's experts (the row's weights sum to 1), and its top k (k + 1 here) picks them and the rest slot. The rest
// slot's expert is all zeros (zero scales and biases), so it adds nothing. Normalizing changes nothing either: the
// picked weights already sum to 1.

export interface RoutingSettings {
  experts: number;
  k: number;
  normalize: boolean;
}

export interface Routing {
  tokens: number;
  k: number;
  /** tokens × k expert indices. */
  experts: Int32Array;
  /** tokens × k weights, matching. */
  weights: Float32Array;
}

/** QMoE's routing of `router` (tokens × experts values). */
export function route(router: ArrayLike<number>, { experts, k, normalize }: RoutingSettings): Routing {
  const tokens = router.length / experts;
  const chosen = new Int32Array(tokens * k);
  const weights = new Float32Array(tokens * k);
  const probs = new Float64Array(experts);
  const order = Array.from({ length: experts }, (_, i) => i);
  for (let t = 0; t < tokens; t++) {
    let max = -Infinity;
    for (let e = 0; e < experts; e++) max = Math.max(max, Number(router[t * experts + e]));
    let sum = 0;
    for (let e = 0; e < experts; e++) {
      probs[e] = Math.exp(Number(router[t * experts + e]) - max);
      sum += probs[e]!;
    }
    // Highest first; ties to the lower index, as a stable top-k does.
    order.sort((a, b) => probs[b]! - probs[a]! || a - b);
    let picked = 0;
    for (let j = 0; j < k; j++) picked += probs[order[j]!]! / sum;
    for (let j = 0; j < k; j++) {
      const e = order[j]!;
      chosen[t * k + j] = e;
      weights[t * k + j] = normalize ? probs[e]! / sum / picked : probs[e]! / sum;
    }
  }
  return { tokens, k, experts: chosen, weights };
}

export interface SlotGroup {
  /** The experts in this group's slots, in slot order (then the rest slot, and any padding slots, all zero). */
  experts: number[];
  /** Slots QMoE is given: experts, the rest slot and padding (so there are at least k + 1). */
  slots: number;
  /** tokens × slots router input (log weights, −∞ as `floor`). */
  router: Float32Array;
}

/**
 * The chosen experts in groups of at most `maxSlots` (rest slot included), each with its router input. Experts are
 * grouped in ascending order. `floor` stands for −∞ (QMoE's router is often float16).
 */
export function slotGroups(routing: Routing, maxSlots: number, floor = -65504): SlotGroup[] {
  const { tokens, k } = routing;
  const used = [...new Set(routing.experts)].sort((a, b) => a - b);
  const perGroup = Math.max(1, maxSlots - 1);
  const groups: SlotGroup[] = [];
  for (let start = 0; start < used.length; start += perGroup) {
    const experts = used.slice(start, start + perGroup);
    const slots = Math.max(experts.length + 1, k + 1);
    const slotOf = new Map(experts.map((e, i) => [e, i]));
    const router = new Float32Array(tokens * slots).fill(floor);
    for (let t = 0; t < tokens; t++) {
      let rest = 0;
      for (let j = 0; j < k; j++) {
        const e = routing.experts[t * k + j]!;
        const w = routing.weights[t * k + j]!;
        const slot = slotOf.get(e);
        if (slot === undefined) rest += w;
        else router[t * slots + slot] = w > 0 ? Math.log(w) : floor;
      }
      router[t * slots + experts.length] = rest > 0 ? Math.log(rest) : floor;
    }
    groups.push({ experts, slots, router });
  }
  return groups;
}

/** float16 bit patterns of `values` (for float16 tensors), rounding to nearest. */
export function halfBits(values: ArrayLike<number>): Uint16Array<ArrayBuffer> {
  const F16 = (globalThis as { Float16Array?: new (v: ArrayLike<number>) => { buffer: ArrayBuffer } }).Float16Array;
  if (F16) return new Uint16Array(new F16(values).buffer);
  const out = new Uint16Array(values.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < values.length; i++) {
    f32[0] = values[i]!;
    const x = u32[0]!;
    const sign = (x >>> 16) & 0x8000;
    const exp = ((x >>> 23) & 0xff) - 127 + 15;
    const mant = x & 0x7fffff;
    if (exp >= 31) out[i] = sign | (((x >>> 23) & 0xff) === 0xff && mant ? 0x7e00 : 0x7c00);
    else if (exp <= 0) out[i] = exp < -10 ? sign : sign | ((mant | 0x800000) >> (14 - exp)) + (((mant | 0x800000) >> (13 - exp)) & 1);
    else out[i] = (sign | (exp << 10) | (mant >> 13)) + ((mant >> 12) & 1);
  }
  return out;
}
