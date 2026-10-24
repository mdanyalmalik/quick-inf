// Splits a mixture-of-experts model at its expert layers, so the experts can be streamed (see moe-stream.ts and
// info/expert-streaming-plan.md).
//
// Every mixture-of-experts model in the catalogue runs its experts with one op, com.microsoft.QMoE, whose expert
// weights are stacked initializers [experts, …]: one expert is a contiguous slice of each. Loaded normally, all of them
// go onto the GPU, though each token uses only k of them per layer. Here the graph (already rewritten by
// external-weights.ts) is cut at every QMoE node into segments:
//
//   segment 0 → QMoE 0 → segment 1 → QMoE 1 → … → QMoE m−1 → segment m
//
// Segment k runs the nodes between QMoE k−1 and QMoE k and outputs what QMoE k needs (its hidden-state input and
// router input) along with anything later segments need from it (the residual stream). The QMoE nodes are left out:
// one separate QMoE session runs them all, fed only the experts the router chose (QMoE takes its weights as inputs as
// well as it does as initializers). The expert initializers are dropped from every segment, so they never load whole.
//
// Values computed only from small model inputs (attention lengths, masks, positions) are recomputed by each segment
// that needs them rather than passed along. Passed values are declared without a type in the segment that makes them;
// ONNX Runtime infers it, and the runtime reads it back (session.outputMetadata) to declare the next segment's inputs,
// so segments are built in order.
//
// ONNX protobuf fields used (onnx.proto3): ModelProto.graph = 7; GraphProto.node = 1, name = 2, initializer = 5,
// input = 11, output = 12; NodeProto.input = 1, output = 2, name = 3, op_type = 4, attribute = 5, domain = 7;
// TensorProto.dims = 1, data_type = 2, name = 8, raw_data = 9, external_data = 13 (key = 1, value = 2);
// ValueInfoProto.name = 1, type = 2: TypeProto.tensor_type = 1: elem_type = 1.

const WIRE_VARINT = 0;
const WIRE_64BIT = 1;
const WIRE_LENGTH = 2;
const WIRE_32BIT = 5;

/** ONNX element types. */
export const ELEM = { float: 1, uint8: 2, int8: 3, int32: 6, int64: 7, bool: 9, float16: 10, bfloat16: 16 } as const;
const ELEM_BYTES: Record<number, number> = { 1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2 };

/** An initializer above this many bytes is a weight: nodes reading it are never recomputed in another segment. */
const SMALL_INITIALIZER_BYTES = 1 << 20;
/** A value needing more nodes than this to recompute is passed between segments instead. */
const MAX_RECOMPUTE_NODES = 64;

/** One of a QMoE node's expert tensors, as stored: expert e is bytes [offset + e × length / experts, …). */
export interface ExpertTensor {
  /** QMoE input position (2 fc1 weights, 3 fc1 scales, 4 fc1 bias, 5 fc2 weights, 6 fc2 scales, 7 fc2 bias, …). */
  input: number;
  name: string;
  dataType: number;
  dims: number[];
  /** External data file (as the rewritten graph names it) and the tensor's bytes in it; `location` is null for a
   * tensor stored in the graph itself (`inline`). */
  location: string | null;
  offset: number;
  length: number;
  inline?: Uint8Array;
}

export interface MoeLayer {
  /** Inputs QMoE reads: hidden state and router. */
  x: string;
  router: string;
  /** QMoE's output. */
  output: string;
  experts: number;
  tensors: ExpertTensor[];
  /** The node's AttributeProto fields, unchanged (identical across layers in every export seen). */
  attributes: Uint8Array[];
  /** Number of inputs the node lists (trailing optional ones included, as empty names). */
  inputCount: number;
}

export interface Segment {
  /** Every input it needs: the model's own inputs and values from earlier segments or QMoE outputs. */
  inputs: string[];
  /** The model's outputs it makes (logits, present states) and values later steps need. */
  outputs: string[];
  /** Values passed in from earlier steps (a subset of inputs): their types must be given to `build`. */
  passedIn: string[];
  /** Bytes of the weights it uses (initializers, inline or external). */
  weightBytes: number;
  /** Those weights: name → bytes. */
  initializers: ReadonlyMap<string, number>;
  /** Where those stored in external files are (their files as the graph names them). */
  externals: { location: string; offset: number; length: number }[];
  /**
   * The segment as a model, each passed-in value declared with the given element type and, when known, shape
   * (ONNX Runtime WebGPU's fusions can fail on inputs of unknown rank).
   */
  build(types: ReadonlyMap<string, ValueType>): Uint8Array<ArrayBuffer>;
}

/** An element type, and a shape whose dimensions are sizes or symbolic names. */
export interface ValueType {
  elem: number;
  shape?: readonly (number | string)[];
}

export interface GraphSplit {
  segments: Segment[];
  /** The model's own inputs (names), for feeding each segment the ones it lists. */
  modelInputs: string[];
}

export interface MoeSplit extends GraphSplit {
  layers: MoeLayer[];
}

interface Field {
  field: number;
  wire: number;
  /** The whole field (tag included). */
  raw: Uint8Array;
  /** Length-delimited payload. */
  payload: Uint8Array;
  /** Varint value. */
  value: number;
}

interface NodeInfo {
  raw: Uint8Array;
  name: string;
  inputs: string[];
  outputs: string[];
  opType: string;
  domain: string;
  attributes: Uint8Array[];
}

interface InitializerInfo {
  raw: Uint8Array;
  name: string;
  dataType: number;
  dims: number[];
  bytes: number;
  external: Map<string, string>;
  /** raw_data, when stored in the graph. */
  rawData?: Uint8Array;
}

/**
 * Splits `model` (ModelProto bytes) at its QMoE nodes; null if it has none. `smallInitializerBytes`: initializers
 * up to this size don't count as weights (tests use tiny models).
 */
export function splitMoe(model: Uint8Array, { smallInitializerBytes = SMALL_INITIALIZER_BYTES } = {}): MoeSplit | null {
  const graph = parseModel(model);
  const { nodes, initializers } = graph;
  const moeIndices = nodes.flatMap((n, i) => (n.opType === 'QMoE' && n.domain === 'com.microsoft' ? [i] : []));
  if (moeIndices.length === 0) return null;
  const layers = moeIndices.map((i) => moeLayer(nodes[i]!, initializers));
  // Segment k holds the nodes between QMoE k−1 and QMoE k; the QMoE nodes run outside the segments.
  const stage: number[] = [];
  let segment = 0;
  nodes.forEach((_, i) => {
    if (moeIndices[segment] === i) {
      stage.push(-1);
      segment++;
    } else stage.push(segment);
  });
  const segments = cutSegments(graph, {
    stage,
    segmentCount: moeIndices.length + 1,
    // QMoE layer k reads its inputs after segment k.
    externalUses: layers.flatMap((l, k) => [l.x, l.router].map((name) => ({ name, beforeSegment: k + 1 }))),
    externalValues: new Set(layers.map((l) => l.output)),
    forbiddenInitializers: new Set(layers.flatMap((l) => l.tensors.map((t) => t.name))),
    smallInitializerBytes,
  });
  return { layers, segments, modelInputs: [...graph.graphInputs.keys()] };
}

/**
 * Splits a decoder at layer boundaries, for running its layers on several devices: segment k holds layers
 * cuts[k−1] up to cuts[k] − 1 (the first also the embedding, the last also the final norm and the output layer). A
 * node's layer comes from its name (`/model/layers.N/…`, as ONNX Runtime GenAI's builder and onnx-community name
 * them); other nodes go with the layer before them. Each segment keeps its own layers' KV inputs and outputs.
 */
export function splitLayers(model: Uint8Array, cuts: readonly number[], { smallInitializerBytes = SMALL_INITIALIZER_BYTES } = {}): GraphSplit {
  if (cuts.some((c, i) => !(c > (cuts[i - 1] ?? 0)))) throw new Error('Cuts must be increasing layer numbers above 0');
  const graph = parseModel(model);
  const segmentOf = (layer: number) => cuts.filter((c) => c <= layer).length;
  let current = 0;
  const stage = graph.nodes.map((n) => {
    const layer = LAYER_NAME.exec(n.name)?.[1];
    if (layer !== undefined) current = segmentOf(Number(layer));
    return current;
  });
  if (stage.at(-1) !== cuts.length) throw new Error(`The model has no layer ${cuts.at(-1)}`);
  const segments = cutSegments(graph, { stage, segmentCount: cuts.length + 1, smallInitializerBytes });
  return { segments, modelInputs: [...graph.graphInputs.keys()] };
}

const LAYER_NAME = /^\/model\/layers\.(\d+)\//;

interface ParsedModel {
  /** ModelProto fields other than the graph (IR version, opsets, …), kept for every segment. */
  otherModelFields: Field[];
  nodes: NodeInfo[];
  initializers: Map<string, InitializerInfo>;
  graphInputs: Map<string, Uint8Array>;
  graphOutputs: Map<string, Uint8Array>;
}

function parseModel(model: Uint8Array): ParsedModel {
  const modelFields = fields(model);
  const graphField = modelFields.find((f) => f.field === 7 && f.wire === WIRE_LENGTH);
  if (!graphField) throw new Error('No graph in the model');
  const parsed: ParsedModel = {
    otherModelFields: modelFields.filter((f) => f !== graphField),
    nodes: [],
    initializers: new Map(),
    graphInputs: new Map(),
    graphOutputs: new Map(),
  };
  for (const f of fields(graphField.payload)) {
    if (f.wire !== WIRE_LENGTH) continue;
    if (f.field === 1) parsed.nodes.push(readNode(f));
    else if (f.field === 5) {
      const init = readInitializer(f);
      parsed.initializers.set(init.name, init);
    } else if (f.field === 11) parsed.graphInputs.set(valueName(f.payload), f.raw);
    else if (f.field === 12) parsed.graphOutputs.set(valueName(f.payload), f.raw);
  }
  if (parsed.nodes.some((n) => n.attributes.some(hasSubgraph))) throw new Error('Graphs with subgraphs (If, Loop) are not supported');
  return parsed;
}

interface Placement {
  /** Each node's segment, or −1 for a node run outside the segments (QMoE). */
  stage: number[];
  segmentCount: number;
  /** Values read by steps outside the segments: each comes from a segment before `beforeSegment`. */
  externalUses?: { name: string; beforeSegment: number }[];
  /** Values made outside the segments. */
  externalValues?: ReadonlySet<string>;
  /** Initializers no segment may use. */
  forbiddenInitializers?: ReadonlySet<string>;
  smallInitializerBytes: number;
}

/** The segments for a placement of the nodes. */
function cutSegments(graph: ParsedModel, placement: Placement): Segment[] {
  const { nodes, initializers, graphInputs, graphOutputs, otherModelFields } = graph;
  const { stage, segmentCount, smallInitializerBytes } = placement;
  const externalValues = placement.externalValues ?? new Set<string>();
  const producer = new Map<string, number>();
  nodes.forEach((n, i) => n.outputs.forEach((o) => o && producer.set(o, i)));
  const isState = (name: string) => graphInputs.has(name) && /^past/.test(name);

  // Nodes that can be recomputed wherever they're needed: no weights, no state inputs, small ancestry.
  const recomputable = new Map<number, Set<number> | null>();
  const ancestry = (i: number): Set<number> | null => {
    if (recomputable.has(i)) return recomputable.get(i)!;
    recomputable.set(i, null); // until shown otherwise (also ends any cycle)
    if (stage[i] === -1) return null;
    const set = new Set<number>([i]);
    for (const input of nodes[i]!.inputs) {
      if (!input) continue;
      const init = initializers.get(input);
      if (init) {
        if (init.bytes > smallInitializerBytes) return null;
        continue;
      }
      if (graphInputs.has(input)) {
        if (isState(input)) return null;
        continue;
      }
      const p = producer.get(input);
      if (p === undefined) return null;
      const sub = ancestry(p);
      if (!sub) return null;
      for (const s of sub) set.add(s);
      if (set.size > MAX_RECOMPUTE_NODES) return null;
    }
    recomputable.set(i, set);
    return set;
  };
  const isRecomputable = nodes.map((_, i) => ancestry(i) !== null);

  // Each other node belongs to its segment.
  const owner = stage.map((s, i) => (isRecomputable[i] ? -1 : s));
  const members: number[][] = Array.from({ length: segmentCount }, () => []);
  owner.forEach((s, i) => s >= 0 && members[s]!.push(i));

  // Where each value is made: a segment, a step outside them, or the model's inputs / initializers.
  const madeIn = new Map<string, number>();
  owner.forEach((s, i) => s >= 0 && nodes[i]!.outputs.forEach((o) => o && madeIn.set(o, s)));

  // Add each segment's recomputed nodes, then work out its inputs.
  const segmentNodes: number[][] = members.map((own) => {
    const set = new Set(own);
    const queue = [...own];
    while (queue.length) {
      const i = queue.pop()!;
      for (const input of nodes[i]!.inputs) {
        const p = producer.get(input);
        if (p !== undefined && isRecomputable[p] && !set.has(p)) {
          set.add(p);
          queue.push(p);
        }
      }
    }
    return [...set].sort((a, b) => a - b);
  });
  const needed: Set<string>[] = segmentNodes.map(() => new Set());
  segmentNodes.forEach((list, s) => {
    const made = new Set(list.flatMap((i) => nodes[i]!.outputs));
    for (const i of list) for (const input of nodes[i]!.inputs) if (input && !made.has(input) && !initializers.has(input)) needed[s]!.add(input);
  });

  const outputsOf: Set<string>[] = segmentNodes.map(() => new Set());
  const recomputedValue = (name: string) => {
    const p = producer.get(name);
    return p !== undefined && isRecomputable[p];
  };
  // A value made in segment s and used later (by a segment or a step outside them) is an output of s.
  const requireFrom = (name: string, consumerSegment: number) => {
    const s = madeIn.get(name);
    if (s !== undefined && s < consumerSegment) outputsOf[s]!.add(name);
  };
  needed.forEach((names, s) => names.forEach((n) => requireFrom(n, s)));
  for (const { name, beforeSegment } of placement.externalUses ?? []) {
    if (recomputedValue(name)) throw new Error(`${name} has no hidden-state source`);
    requireFrom(name, beforeSegment);
  }
  segmentNodes.forEach((list, s) => {
    for (const i of list) for (const o of nodes[i]!.outputs) if (graphOutputs.has(o)) outputsOf[s]!.add(o);
  });

  return segmentNodes.map((list, s) => {
    const inputs = [...needed[s]!];
    const passedIn = inputs.filter((n) => !graphInputs.has(n));
    for (const n of passedIn) {
      const from = madeIn.get(n);
      if (from === undefined ? !externalValues.has(n) : from >= s) throw new Error(`Segment ${s} needs ${n}, which nothing before it makes`);
    }
    const outputs = [...outputsOf[s]!];
    const usedInitializers = new Set(list.flatMap((i) => nodes[i]!.inputs.filter((n) => initializers.has(n))));
    for (const n of usedInitializers) if (placement.forbiddenInitializers?.has(n)) throw new Error(`Expert weights ${n} are used outside QMoE`);
    return {
      inputs,
      outputs,
      passedIn,
      weightBytes: [...usedInitializers].reduce((n, name) => n + initializers.get(name)!.bytes, 0),
      initializers: new Map([...usedInitializers].map((name) => [name, initializers.get(name)!.bytes])),
      externals: [...usedInitializers].flatMap((name) => {
        const { external, bytes } = initializers.get(name)!;
        const location = external.get('location');
        return location ? [{ location, offset: Number(external.get('offset') ?? 0), length: Number(external.get('length') ?? bytes) }] : [];
      }),
      build(types) {
        const graph: Uint8Array[] = [];
        for (const i of list) graph.push(nodes[i]!.raw);
        graph.push(lengthField(2, utf8(`segment ${s}`)));
        for (const n of usedInitializers) graph.push(initializers.get(n)!.raw);
        for (const n of inputs) {
          if (graphInputs.has(n)) graph.push(graphInputs.get(n)!);
          else {
            const type = types.get(n);
            if (type === undefined) throw new Error(`No type for ${n}`);
            graph.push(lengthField(11, valueInfo(n, type.elem, type.shape)));
          }
        }
        for (const n of outputs) graph.push(graphOutputs.get(n) ?? lengthField(12, valueInfo(n, null)));
        return concat([...otherModelFields.map((f) => f.raw), lengthField(7, concat(graph))]);
      },
    };
  });
}

/**
 * One QMoE node as a model of its own, with every input a model input: x, router, then the expert tensors as
 * `layer`'s, at any number of experts each. `k` overrides the node's top-k (the streaming runtime adds a "rest"
 * slot). With `accumulate`, the output is QMoE's plus an input `acc` (for adding up groups of experts).
 */
export function qmoeModel(
  layer: MoeLayer,
  opset: Uint8Array[],
  types: { x: number; router: number },
  k: number,
  accumulate = false,
): Uint8Array<ArrayBuffer> {
  const inputs = new Array<string>(layer.inputCount).fill('');
  inputs[0] = 'x';
  inputs[1] = 'router';
  for (const t of layer.tensors) inputs[t.input] = `expert_${t.input}`;
  const node: Uint8Array[] = inputs.map((n) => lengthField(1, utf8(n)));
  const moeOut = accumulate ? 'moe' : 'y';
  node.push(lengthField(2, utf8(moeOut)), lengthField(3, utf8('QMoE')), lengthField(4, utf8('QMoE')), lengthField(7, utf8('com.microsoft')));
  for (const a of layer.attributes) {
    node.push(attributeName(a) === 'k' ? lengthField(5, intAttribute('k', k)) : lengthField(5, a));
  }
  const graph: Uint8Array[] = [lengthField(1, concat(node))];
  if (accumulate) {
    graph.push(lengthField(1, concat([lengthField(1, utf8('moe')), lengthField(1, utf8('acc')), lengthField(2, utf8('y')), lengthField(4, utf8('Add'))])));
  }
  graph.push(lengthField(2, utf8('qmoe')));
  graph.push(lengthField(11, valueInfo('x', types.x)), lengthField(11, valueInfo('router', types.router)));
  for (const t of layer.tensors) graph.push(lengthField(11, valueInfo(`expert_${t.input}`, t.dataType)));
  if (accumulate) graph.push(lengthField(11, valueInfo('acc', types.x)));
  graph.push(lengthField(12, valueInfo('y', types.x)));
  return concat([varintField(1, 10), ...opset, lengthField(7, concat(graph))]);
}

/** A layer's integer attribute (AttributeProto.i = 3), or `fallback`. */
export function intAttributeOf(layer: MoeLayer, name: string, fallback: number): number {
  for (const a of layer.attributes) {
    if (attributeName(a) !== name) continue;
    const f = fields(a).find((g) => g.field === 3 && g.wire === WIRE_VARINT);
    if (f) return f.value;
  }
  return fallback;
}

/** The model's opset_import fields (ModelProto field 8), for building the QMoE model. */
export function opsetImports(model: Uint8Array): Uint8Array[] {
  return fields(model)
    .filter((f) => f.field === 8 && f.wire === WIRE_LENGTH)
    .map((f) => f.raw);
}

function moeLayer(node: NodeInfo, initializers: Map<string, InitializerInfo>): MoeLayer {
  const tensors: ExpertTensor[] = [];
  let experts = 0;
  node.inputs.forEach((name, input) => {
    if (input < 2 || !name) return;
    const init = initializers.get(name);
    if (!init) throw new Error(`QMoE input ${name} isn't stored in the model`);
    const location = init.external.get('location') ?? null;
    if (location === null && !init.rawData) throw new Error(`Expert weights ${name} are stored in a form not supported here`);
    experts ||= init.dims[0]!;
    if (init.dims[0] !== experts) throw new Error(`Expert weights ${name} have ${init.dims[0]} experts, not ${experts}`);
    tensors.push({
      input,
      name,
      dataType: init.dataType,
      dims: init.dims,
      location,
      offset: Number(init.external.get('offset') ?? 0),
      length: location === null ? init.rawData!.length : Number(init.external.get('length') ?? init.bytes),
      ...(location === null && { inline: init.rawData }),
    });
  });
  if (!tensors.length) throw new Error('QMoE node without expert weights');
  return { x: node.inputs[0]!, router: node.inputs[1]!, output: node.outputs[0]!, experts, tensors, attributes: node.attributes, inputCount: node.inputs.length };
}

function readNode(f: Field): NodeInfo {
  const node: NodeInfo = { raw: f.raw, name: '', inputs: [], outputs: [], opType: '', domain: '', attributes: [] };
  for (const g of fields(f.payload)) {
    if (g.wire !== WIRE_LENGTH) continue;
    if (g.field === 1) node.inputs.push(decode(g.payload));
    else if (g.field === 2) node.outputs.push(decode(g.payload));
    else if (g.field === 3) node.name = decode(g.payload);
    else if (g.field === 4) node.opType = decode(g.payload);
    else if (g.field === 5) node.attributes.push(g.payload);
    else if (g.field === 7) node.domain = decode(g.payload);
  }
  return node;
}

function readInitializer(f: Field): InitializerInfo {
  const init: InitializerInfo = { raw: f.raw, name: '', dataType: 0, dims: [], bytes: 0, external: new Map() };
  let raw = 0;
  for (const g of fields(f.payload)) {
    if (g.field === 1 && g.wire === WIRE_VARINT) init.dims.push(g.value);
    else if (g.field === 1 && g.wire === WIRE_LENGTH) init.dims.push(...packedVarints(g.payload));
    else if (g.field === 2 && g.wire === WIRE_VARINT) init.dataType = g.value;
    else if (g.field === 8 && g.wire === WIRE_LENGTH) init.name = decode(g.payload);
    else if (g.field === 9 && g.wire === WIRE_LENGTH) {
      raw = g.payload.length;
      init.rawData = g.payload;
    }
    else if (g.field === 13 && g.wire === WIRE_LENGTH) {
      let key = '';
      let value = '';
      for (const e of fields(g.payload)) {
        if (e.field === 1) key = decode(e.payload);
        if (e.field === 2) value = decode(e.payload);
      }
      init.external.set(key, value);
    }
  }
  init.bytes = raw || init.dims.reduce((a, b) => a * b, 1) * (ELEM_BYTES[init.dataType] ?? 4);
  return init;
}

/** AttributeProto.g = 6 or graphs = 11: an op with a subgraph. */
function hasSubgraph(attribute: Uint8Array): boolean {
  return fields(attribute).some((f) => (f.field === 6 || f.field === 11) && f.wire === WIRE_LENGTH);
}

function attributeName(attribute: Uint8Array): string {
  const f = fields(attribute).find((g) => g.field === 1 && g.wire === WIRE_LENGTH);
  return f ? decode(f.payload) : '';
}

/** AttributeProto: name = 1, i = 3, type = 20 (INT = 2). */
function intAttribute(name: string, value: number): Uint8Array {
  return concat([lengthField(1, utf8(name)), varintField(3, value), varintField(20, 2)]);
}

function valueName(info: Uint8Array): string {
  const f = fields(info).find((g) => g.field === 1 && g.wire === WIRE_LENGTH);
  return f ? decode(f.payload) : '';
}

/**
 * ValueInfoProto with an element type (and shape, if given: TensorShapeProto.dim = 1, Dimension.dim_value = 1,
 * dim_param = 2), or with no type at all (ONNX Runtime infers it).
 */
function valueInfo(name: string, elemType: number | null, shape?: readonly (number | string)[]): Uint8Array {
  const parts = [lengthField(1, utf8(name))];
  if (elemType !== null) {
    const tensor = [varintField(1, elemType)];
    if (shape) {
      // ONNX Runtime Web reports an unknown dimension as 2³² − 1: it stays unknown (an empty Dimension).
      const dims = shape.map((d) =>
        lengthField(1, typeof d === 'string' ? lengthField(2, utf8(d)) : d >= 0 && d < 2 ** 31 ? varintField(1, d) : new Uint8Array(0)),
      );
      tensor.push(lengthField(2, concat(dims)));
    }
    parts.push(lengthField(2, lengthField(1, concat(tensor))));
  }
  return concat(parts);
}

/** ONNX Runtime's tensor type names as ONNX element types. */
export function elemTypeOf(type: string): number {
  const map: Record<string, number> = { float32: 1, uint8: 2, int8: 3, int32: 6, int64: 7, bool: 9, float16: 10, bfloat16: 16 };
  const t = map[type];
  if (t === undefined) throw new Error(`Unsupported tensor type ${type}`);
  return t;
}

function fields(bytes: Uint8Array): Field[] {
  const out: Field[] = [];
  let pos = 0;
  while (pos < bytes.length) {
    const start = pos;
    const [key, afterKey] = varint(bytes, pos);
    const field = Math.floor(key / 8);
    const wire = key % 8;
    let payload = bytes.subarray(0, 0);
    let value = 0;
    if (wire === WIRE_VARINT) {
      const [v, next] = varint(bytes, afterKey);
      value = v;
      pos = next;
    } else if (wire === WIRE_LENGTH) {
      const [length, afterLength] = varint(bytes, afterKey);
      payload = bytes.subarray(afterLength, afterLength + length);
      pos = afterLength + length;
    } else if (wire === WIRE_64BIT) pos = afterKey + 8;
    else if (wire === WIRE_32BIT) pos = afterKey + 4;
    else throw new Error(`Unsupported protobuf wire type ${wire}`);
    if (pos > bytes.length) throw new Error('Malformed model file');
    out.push({ field, wire, raw: bytes.subarray(start, pos), payload, value });
  }
  return out;
}

function varint(bytes: Uint8Array, pos: number): [number, number] {
  let value = 0;
  let scale = 1;
  for (let i = 0; i < 10; i++) {
    const byte = bytes[pos + i]!;
    value += (byte & 0x7f) * scale;
    if (byte < 0x80) return [value, pos + i + 1];
    scale *= 128;
  }
  throw new Error('Malformed varint');
}

function packedVarints(bytes: Uint8Array): number[] {
  const out: number[] = [];
  for (let pos = 0; pos < bytes.length;) {
    const [v, next] = varint(bytes, pos);
    out.push(v);
    pos = next;
  }
  return out;
}

function encodeVarint(value: number): number[] {
  const out: number[] = [];
  let v = value;
  while (v >= 0x80) {
    out.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return out;
}

function varintField(field: number, value: number): Uint8Array {
  return Uint8Array.from([...encodeVarint(field * 8 + WIRE_VARINT), ...encodeVarint(value)]);
}

function lengthField(field: number, payload: Uint8Array): Uint8Array {
  return concat([Uint8Array.from([...encodeVarint(field * 8 + WIRE_LENGTH), ...encodeVarint(payload.length)]), payload]);
}

function concat(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();
const decode = (bytes: Uint8Array) => textDecoder.decode(bytes);
const utf8 = (text: string) => textEncoder.encode(text);
