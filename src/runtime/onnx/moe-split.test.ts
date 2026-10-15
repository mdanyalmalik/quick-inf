import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { halfBits, route, slotGroups } from './moe-route';
import { elemTypeOf, opsetImports, qmoeModel, splitMoe, type MoeLayer, type ValueType } from './moe-split';

// tiny-moe.onnx: two QMoE layers (8 experts, top 2), a residual stream, a carried state, an attention-mask value used
// by both layers, and "heavy" embedding and output weights (generated with ONNX; expected outputs from ONNX Runtime's
// CPU engine in Python).
const DIR = new URL('./__fixtures__/', import.meta.url);
const model = new Uint8Array(readFileSync(new URL('tiny-moe.onnx', DIR)));
const data = new Uint8Array(readFileSync(new URL('tiny-moe.onnx.data', DIR)));
const expected = JSON.parse(readFileSync(new URL('tiny-moe.expected.json', DIR), 'utf8')) as { ids: number[]; logits: number[]; present_s: number[] };
const SMALL = 16 * 1024;
const externalData = [{ path: 'tiny-moe.onnx.data', data }];
const H = 64;

function inputs(): Record<string, ort.Tensor> {
  const n = expected.ids.length;
  return {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(expected.ids, BigInt), [1, n]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
    past_s: new ort.Tensor('float16', halfBits(new Float32Array(H).fill(0.5)), [1, 1, H]),
  };
}

const numbers = (t: ort.Tensor) => Array.from(t.data as ArrayLike<number>, Number);
const maxDiff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i]!)));

/** Runs the split model: segments in order, each QMoE layer fed only its chosen experts in groups of `maxSlots`. */
async function runStreamed(maxSlots: number) {
  const split = splitMoe(model, { smallInitializerBytes: SMALL })!;
  const feeds = inputs();
  const values = new Map<string, ort.Tensor>(Object.entries(feeds));
  const types = new Map<string, ValueType>();
  const opset = opsetImports(model);
  const outputs: Record<string, ort.Tensor> = {};
  for (const [s, segment] of split.segments.entries()) {
    const session = await ort.InferenceSession.create(segment.build(types), { executionProviders: ['wasm'], externalData });
    for (const meta of session.outputMetadata) if (meta.isTensor) types.set(meta.name, { elem: elemTypeOf(meta.type), shape: meta.shape });
    const result = await session.run(Object.fromEntries(segment.inputs.map((n) => [n, values.get(n)!])));
    for (const [name, value] of Object.entries(result)) {
      values.set(name, value as ort.Tensor);
      outputs[name] = value as ort.Tensor;
    }
    await session.release();
    const layer = split.layers[s];
    if (!layer) break;
    const y = await runExperts(layer, values.get(layer.x)!, values.get(layer.router)!, opset, types, maxSlots);
    values.set(layer.output, y);
    types.set(layer.output, types.get(layer.x)!);
  }
  return outputs;
}

async function runExperts(layer: MoeLayer, x: ort.Tensor, router: ort.Tensor, opset: Uint8Array[], types: Map<string, ValueType>, maxSlots: number) {
  const attr = { experts: layer.experts, k: 2, normalize: true };
  const routing = route(numbers(router), attr);
  const qmoe = await ort.InferenceSession.create(qmoeModel(layer, opset, { x: types.get(layer.x)!.elem, router: types.get(layer.router)!.elem }, attr.k + 1), {
    executionProviders: ['wasm'],
  });
  const sum = new Float32Array(x.size);
  for (const group of slotGroups(routing, maxSlots)) {
    const feeds: Record<string, ort.Tensor> = { x, router: new ort.Tensor('float16', halfBits(group.router), [routing.tokens, group.slots]) };
    for (const t of layer.tensors) {
      const per = t.length / layer.experts;
      const bytes = new Uint8Array(per * group.slots); // rest and padding slots stay zero
      group.experts.forEach((e, i) => bytes.set(data.subarray(t.offset + e * per, t.offset + (e + 1) * per), i * per));
      const dims = [group.slots, ...t.dims.slice(1)];
      feeds[`expert_${t.input}`] = t.dataType === 2 ? new ort.Tensor('uint8', bytes, dims) : new ort.Tensor('float16', new Uint16Array(bytes.buffer), dims);
    }
    const y = (await qmoe.run(feeds)).y as ort.Tensor;
    numbers(y).forEach((v, i) => (sum[i]! += v));
  }
  await qmoe.release();
  return new ort.Tensor('float16', halfBits(sum), x.dims);
}

describe('splitMoe', () => {
  it('cuts at each QMoE, recomputes input-only values and passes the residual stream', () => {
    const split = splitMoe(model, { smallInitializerBytes: SMALL })!;
    expect(split.layers.map((l) => [l.x, l.router, l.output, l.experts])).toEqual([
      ['x0', 'router_in0', 'moe0', 8],
      ['x1', 'router_in1', 'moe1', 8],
    ]);
    expect(split.layers[0]!.tensors.map((t) => t.input)).toEqual([2, 3, 4, 5, 6, 7]);
    const [s0, s1, s2] = split.segments;
    expect(s0!.outputs.sort()).toEqual(['h0', 'present_s', 'router_in0', 'x0']);
    // The attention-mask value is recomputed in each segment that uses it, not passed.
    expect(s1!.inputs.sort()).toEqual(['attention_mask', 'h0', 'moe0']);
    expect(s1!.passedIn.sort()).toEqual(['h0', 'moe0']);
    expect(s2!.inputs.sort()).toEqual(['attention_mask', 'h1', 'moe1']);
    expect(s2!.outputs).toEqual(['logits']);
  });

  it('returns null for a model without experts', () => {
    expect(splitMoe(new Uint8Array(readFileSync(new URL('rewrite-matmul-fp16.onnx', DIR))))).toBeNull();
  });

  it('runs the same as the whole model, with experts fed in one group or several', async () => {
    const full = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], externalData });
    const reference = await full.run(inputs());
    expect(maxDiff(numbers(reference.logits as ort.Tensor), expected.logits)).toBeLessThan(0.02);
    for (const maxSlots of [16, 3]) {
      const streamed = await runStreamed(maxSlots);
      expect(maxDiff(numbers(streamed.logits!), numbers(reference.logits as ort.Tensor))).toBeLessThan(0.05);
      expect(maxDiff(numbers(streamed.present_s!), numbers(reference.present_s as ort.Tensor))).toBe(0);
    }
  }, 60000);
});

describe('route', () => {
  it('picks the top k of the softmax and normalizes them', () => {
    const r = route([0, Math.log(3), Math.log(1), -20], { experts: 4, k: 2, normalize: true });
    expect(Array.from(r.experts)).toEqual([1, 0]);
    expect(Array.from(r.weights).map((w) => +w.toFixed(4))).toEqual([0.75, 0.25]);
  });

  it('splits the chosen experts into groups whose router rows give the full weights', () => {
    const routing = { tokens: 1, k: 2, experts: Int32Array.of(5, 1), weights: Float32Array.of(0.7, 0.3) };
    const groups = slotGroups(routing, 2);
    expect(groups.map((g) => g.experts)).toEqual([[1], [5]]);
    const [g] = groups;
    expect(g!.slots).toBe(3);
    expect(Math.exp(g!.router[0]!)).toBeCloseTo(0.3);
    expect(Math.exp(g!.router[1]!)).toBeCloseTo(0.7); // the rest slot: expert 5's weight
  });
});
