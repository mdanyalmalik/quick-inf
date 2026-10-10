// Loads a single-file ONNX model without ever holding the whole file in memory.
//
// Normally ONNX Runtime Web copies the model bytes into its WebAssembly memory
// and then parses them (building more copies of the weights), which made a
// 1.1 GB model take 4+ GB of memory while loading and crashed phone tabs.
//
// Instead, `externalizeWeights` rewrites only the model's structure: every
// large weight tensor is changed from "data inline" to "external data at offset
// X of file W", where W is the original model file itself. Nothing is copied:
// the offsets point at where each tensor's bytes already sit in the file. The
// small rewritten structure goes to ONNX Runtime as the model, and the original
// file is mounted as the external data file through `lazyBytes`, which reads
// each tensor from disk only when ONNX Runtime asks for it (it then uploads that
// tensor straight to the GPU). Peak memory is roughly the largest single tensor.
//
// It also removes one costly pattern these exports share: the output layer
// multiplies by the transposed token-embedding table, `MatMul(x, Transpose(E))`.
// ONNX Runtime pre-computes that transpose at load time, building a second copy
// of the table (311 MB for Qwen3 0.6B, 545 MB for Qwen2.5 0.5B) in WebAssembly
// memory, which never shrinks, and then on the GPU too. `Gemm(x, E, transB = 1)`
// computes the same product reading E as it is; Gemm takes 2-D inputs, so x is
// reshaped to [tokens, hidden] before and the result back to [1, tokens, vocab]
// after (the runtime always runs one sequence at a time). ONNX Runtime's own
// fusion to FusedMatMul isn't usable: its WebGPU backend lacks FusedMatMul, so
// that runs on the CPU, several times slower.
//
// Finally, transformers.js's fused-attention exports (e.g. Qwen2.5 "-ONNX-MHA")
// build their causal mask in a subgraph (/model/attn_mask_reformat/) that only
// works with no KV cache or one new token: with several new tokens after a cache
// (chunked prompts, reusing the previous reply's cache) it fails, because it
// builds a tokens × tokens triangle where tokens × (past + tokens) is needed.
// The subgraph is replaced by a model input, ATTENTION_BIAS_INPUT, that the
// runtime fills with the right mask.
//
// ONNX protobuf fields used (onnx.proto3): ModelProto.graph = 7;
// GraphProto.node = 1, initializer = 5, input = 11, output = 12 (ValueInfoProto.name = 1,
// type = 2: TypeProto.tensor_type = 1: elem_type = 1, shape = 2: dim = 1: dim_value = 1, dim_param = 2);
// NodeProto.input = 1, output = 2, name = 3, op_type = 4, attribute = 5, domain = 7;
// AttributeProto.name = 1, i = 3, ints = 8, type = 20 (INT = 2);
// TensorProto.dims = 1, data_type = 2 (INT64 = 7), name = 8, raw_data = 9, external_data = 13
// (StringStringEntryProto: key = 1, value = 2), data_location = 14 (EXTERNAL = 1).

/** Random-access reading of a file (or an in-memory buffer). */
export interface ByteSource {
  readonly size: number;
  /** Returns `length` bytes starting at `offset`. */
  read(offset: number, length: number): Uint8Array<ArrayBuffer>;
}

export function bytesSource(bytes: Uint8Array<ArrayBuffer>): ByteSource {
  return { size: bytes.byteLength, read: (offset, length) => bytes.subarray(offset, offset + length) };
}

/** Tensors smaller than this stay inline (not worth a separate read). */
const MIN_EXTERNAL_BYTES = 1024;

const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const GRAPH_INPUT = 11;
const GRAPH_OUTPUT = 12;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_NAME = 3;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const NODE_DOMAIN = 7;
const ATTRIBUTE_NAME = 1;
const ATTRIBUTE_I = 3;
const ATTRIBUTE_T = 5;
const ATTRIBUTE_INTS = 8;
const ATTRIBUTE_TYPE = 20;
const ATTRIBUTE_TYPE_INT = 2;
const TENSOR_DIMS = 1;
const TENSOR_DATA_TYPE = 2;
const TENSOR_TYPE_INT64 = 7;
const TENSOR_TYPE_FLOAT = 1;
const TENSOR_TYPE_FLOAT16 = 10;
const TENSOR_NAME = 8;
const TENSOR_RAW_DATA = 9;
const TENSOR_EXTERNAL_DATA = 13;
const TENSOR_DATA_LOCATION = 14;
const WIRE_VARINT = 0;
const WIRE_64BIT = 1;
const WIRE_LENGTH = 2;
const WIRE_32BIT = 5;

export interface ExternalizedModel {
  /** The model structure, with large tensors pointing into the external data file. */
  graph: Uint8Array<ArrayBuffer>;
  /** Tensors moved to external data, and their total bytes. */
  tensors: number;
  bytes: number;
  /** MatMuls by a transposed constant rewritten as Gemm (see above). */
  fusedTransposes: number;
  /** Whether the causal-mask subgraph was replaced by ATTENTION_BIAS_INPUT (see above). */
  attentionBias: boolean;
}

/** The causal mask input added in place of a broken mask subgraph: [1, 1, tokens, past + tokens], 0 or a large negative. */
export const ATTENTION_BIAS_INPUT = 'attention_bias';
const MASK_SUBGRAPH = '/model/attn_mask_reformat/';

/**
 * Rewrites a single-file ONNX model so its large initializers refer to
 * `location` (which must be mounted as the original file) instead of being inline.
 */
export function externalizeWeights(source: ByteSource, location: string): ExternalizedModel {
  const reader = new BufferedReader(source);
  const out = new Writer();
  const stats = { tensors: 0, bytes: 0, fusedTransposes: 0, attentionBias: false };
  forEachField(reader, 0, source.size, (field, wire, start, payloadStart, end) => {
    if (field === MODEL_GRAPH && wire === WIRE_LENGTH) {
      const graph = rewriteGraph(reader, payloadStart, end, location, stats);
      out.tag(MODEL_GRAPH, WIRE_LENGTH);
      out.lengthDelimited(graph);
    } else {
      out.bytes(reader.slice(start, end));
    }
  });
  return { graph: out.finish(), ...stats };
}

type Stats = { tensors: number; bytes: number; fusedTransposes: number; attentionBias: boolean };

interface NodeInfo {
  start: number;
  end: number;
  inputs: string[];
  outputs: string[];
  name: string;
  opType: string;
  domain: string;
  /** perm attribute, if any (Transpose). */
  perm: number[] | null;
  attributes: number;
}

function rewriteGraph(reader: BufferedReader, start: number, end: number, location: string, stats: Stats): Uint8Array {
  const fused = planTransposeFusions(reader, start, end);
  const mask = planMaskInput(reader, start, end);
  if (mask) stats.attentionBias = true;
  const out = new Writer();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field === GRAPH_NODE && wire === WIRE_LENGTH && mask?.replace.has(fieldStart)) {
      const replacement = mask.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && fused.replace.has(fieldStart)) {
      const replacement = fused.replace.get(fieldStart)!;
      if (replacement) {
        out.bytes(replacement);
        stats.fusedTransposes++;
      }
    } else if (field === GRAPH_INITIALIZER && wire === WIRE_LENGTH) {
      const tensor = rewriteTensor(reader, payloadStart, fieldEnd, location, stats);
      out.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
      out.lengthDelimited(tensor);
    } else {
      out.bytes(reader.slice(fieldStart, fieldEnd));
    }
  });
  out.bytes(fused.extra);
  if (mask) out.bytes(mask.input);
  return out.finish();
}

/**
 * Finds the transformers.js causal-mask subgraph (see the top of this file). Returns
 * the node edits (null drops a node; the final Tile is rewritten to read the new
 * input) and the new graph input, or null if the graph doesn't have it.
 */
function planMaskInput(
  reader: BufferedReader,
  start: number,
  end: number,
): { replace: Map<number, Uint8Array | null>; input: Uint8Array } | null {
  const nodes: NodeInfo[] = [];
  const initializerTypes = new Map<string, number>();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_NODE) nodes.push(readNode(reader, fieldStart, payloadStart, fieldEnd));
    else if (field === GRAPH_INITIALIZER) {
      let name = '';
      let type = 0;
      forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
        if (f === TENSOR_NAME && w === WIRE_LENGTH) name = text(reader, p, e);
        if (f === TENSOR_DATA_TYPE && w === WIRE_VARINT) type = reader.varint(p)[0];
      });
      initializerTypes.set(name, type);
    }
  });
  // Constants can also come from Constant nodes (Phi-3.5's export): their `value` tensor's type.
  for (const n of nodes) {
    if (n.opType !== 'Constant' || n.outputs.length !== 1) continue;
    forEachField(reader, n.start, n.end, (f, w, _s, p, e) => {
      if (f !== GRAPH_NODE || w !== WIRE_LENGTH) return;
      forEachField(reader, p, e, (nf, nw, _ns, np, ne) => {
        if (nf !== NODE_ATTRIBUTE || nw !== WIRE_LENGTH) return;
        forEachField(reader, np, ne, (af, aw, _as, ap, ae) => {
          if (af !== ATTRIBUTE_T || aw !== WIRE_LENGTH) return;
          forEachField(reader, ap, ae, (tf, tw, _ts, tp) => {
            if (tf === TENSOR_DATA_TYPE && tw === WIRE_VARINT) initializerTypes.set(n.outputs[0]!, reader.varint(tp)[0]);
          });
        });
      });
    });
  }
  const subgraph = nodes.filter((n) => n.name.startsWith(MASK_SUBGRAPH));
  const tiles = subgraph.filter((n) => n.opType === 'Tile');
  if (tiles.length !== 1) return null;
  const tile = tiles[0]!;
  // Nodes whose results something outside the subgraph still reads (e.g. Phi-3.5's rotary
  // cache reads the total length) stay, with everything they depend on; the rest go.
  const producer = new Map(subgraph.flatMap((n) => n.outputs.map((o) => [o, n] as const)));
  const keep = new Set<NodeInfo>();
  const need = (name: string) => {
    const node = producer.get(name);
    // The Tile is rewritten to read the new input, so what it reads isn't needed.
    if (!node || node === tile || keep.has(node)) return;
    keep.add(node);
    node.inputs.forEach(need);
  };
  for (const n of nodes) if (!n.name.startsWith(MASK_SUBGRAPH)) n.inputs.forEach(need);
  // The mask's element type: that of the large negative constant it fills masked places with.
  const isFloat = (t: number) => t === TENSOR_TYPE_FLOAT || t === TENSOR_TYPE_FLOAT16;
  const elemType = subgraph
    .filter((n) => n.opType === 'Where')
    .flatMap((n) => n.inputs.map((i) => initializerTypes.get(i) ?? 0))
    .find(isFloat);
  if (!elemType) return null;

  const replace = new Map<number, Uint8Array | null>();
  for (const node of subgraph) if (!keep.has(node)) replace.set(node.start, null);
  // The Tile, reading the new input instead: [1, 1, tokens, total] → [1, heads, tokens, total].
  replace.set(tile.start, nodeField('Tile', tile.name, [ATTENTION_BIAS_INPUT, ...tile.inputs.slice(1)], tile.outputs));

  const shape = new Writer();
  for (const dim of [1, 1, 'sequence_length', 'total_sequence_length'] as const) {
    const d = new Writer();
    if (typeof dim === 'number') {
      d.tag(1, WIRE_VARINT);
      d.varint(dim);
    } else {
      d.tag(2, WIRE_LENGTH);
      d.lengthDelimited(utf8(dim));
    }
    shape.tag(1, WIRE_LENGTH);
    shape.lengthDelimited(d.finish());
  }
  const tensorType = new Writer();
  tensorType.tag(1, WIRE_VARINT);
  tensorType.varint(elemType);
  tensorType.tag(2, WIRE_LENGTH);
  tensorType.lengthDelimited(shape.finish());
  const type = new Writer();
  type.tag(1, WIRE_LENGTH);
  type.lengthDelimited(tensorType.finish());
  const info = new Writer();
  info.tag(1, WIRE_LENGTH);
  info.lengthDelimited(utf8(ATTENTION_BIAS_INPUT));
  info.tag(2, WIRE_LENGTH);
  info.lengthDelimited(type.finish());
  const input = new Writer();
  input.tag(GRAPH_INPUT, WIRE_LENGTH);
  input.lengthDelimited(info.finish());
  return { replace, input: input.finish() };
}

/**
 * Finds `MatMul(x, Transpose(W))` where W is a 2-D initializer and the Transpose
 * feeds only MatMuls (as their second input). Returns, by node start offset, the
 * graph fields to write instead (Reshape, Gemm, Reshape) or null for a Transpose
 * to drop, plus fields to add to the graph (the reshape shapes).
 */
function planTransposeFusions(
  reader: BufferedReader,
  start: number,
  end: number,
): { replace: Map<number, Uint8Array | null>; extra: Uint8Array } {
  const nodes: NodeInfo[] = [];
  const twoDimInitializers = new Map<string, [number, number]>();
  const graphOutputs = new Set<string>();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_NODE) nodes.push(readNode(reader, fieldStart, payloadStart, fieldEnd));
    else if (field === GRAPH_OUTPUT) {
      forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
        if (f === 1 && w === WIRE_LENGTH) graphOutputs.add(text(reader, p, e));
      });
    } else if (field === GRAPH_INITIALIZER) {
      let name = '';
      const dims: number[] = [];
      forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
        if (f === TENSOR_NAME && w === WIRE_LENGTH) name = text(reader, p, e);
        if (f === TENSOR_DIMS) {
          if (w === WIRE_VARINT) dims.push(reader.varint(p)[0]);
          else if (w === WIRE_LENGTH) {
            for (let pos = p; pos < e; ) {
              const [v, next] = reader.varint(pos);
              dims.push(v);
              pos = next;
            }
          }
        }
      });
      if (dims.length === 2) twoDimInitializers.set(name, [dims[0]!, dims[1]!]);
    }
  });

  const consumers = new Map<string, { node: NodeInfo; index: number }[]>();
  for (const node of nodes) {
    node.inputs.forEach((input, index) => {
      const list = consumers.get(input) ?? [];
      list.push({ node, index });
      consumers.set(input, list);
    });
  }

  const plan = new Map<number, Uint8Array | null>();
  const extra = new Writer();
  for (const node of nodes) {
    if (node.opType !== 'Transpose' || node.domain !== '' || node.inputs.length !== 1 || node.outputs.length !== 1) continue;
    if (node.perm && !(node.perm.length === 2 && node.perm[0] === 1 && node.perm[1] === 0)) continue;
    const weight = node.inputs[0]!;
    const transposed = node.outputs[0]!;
    if (!twoDimInitializers.has(weight) || graphOutputs.has(transposed)) continue;
    const uses = consumers.get(transposed) ?? [];
    const fusable = (u: { node: NodeInfo; index: number }) =>
      u.node.opType === 'MatMul' && u.node.domain === '' && u.index === 1 && u.node.inputs.length === 2 && u.node.attributes === 0;
    if (uses.length === 0 || !uses.every(fusable)) continue;
    // W is [out, in]: x [1, tokens, in] → [tokens, in] → Gemm → [tokens, out] → [1, tokens, out].
    const [outFeatures, inFeatures] = twoDimInitializers.get(weight)!;
    const inShape = `${weight}/gemm_in_shape`;
    const outShape = `${weight}/gemm_out_shape`;
    extra.bytes(int64Initializer(inShape, [-1, inFeatures]));
    extra.bytes(int64Initializer(outShape, [1, -1, outFeatures]));
    plan.set(node.start, null);
    uses.forEach(({ node: matmul }, i) => plan.set(matmul.start, gemmNodes(matmul, weight, inShape, outShape, `${weight}/gemm${i}`)));
  }
  return { replace: plan, extra: extra.finish() };
}

function readNode(reader: BufferedReader, start: number, payloadStart: number, end: number): NodeInfo {
  const node: NodeInfo = { start, end, inputs: [], outputs: [], name: '', opType: '', domain: '', perm: null, attributes: 0 };
  forEachField(reader, payloadStart, end, (field, wire, _s, p, e) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === NODE_INPUT) node.inputs.push(text(reader, p, e));
    else if (field === NODE_OUTPUT) node.outputs.push(text(reader, p, e));
    else if (field === NODE_NAME) node.name = text(reader, p, e);
    else if (field === NODE_OP_TYPE) node.opType = text(reader, p, e);
    else if (field === NODE_DOMAIN) node.domain = text(reader, p, e);
    else if (field === NODE_ATTRIBUTE) {
      node.attributes++;
      let name = '';
      const ints: number[] = [];
      forEachField(reader, p, e, (f, w, _s2, p2, e2) => {
        if (f === ATTRIBUTE_NAME && w === WIRE_LENGTH) name = text(reader, p2, e2);
        if (f === ATTRIBUTE_INTS) {
          if (w === WIRE_VARINT) ints.push(reader.varint(p2)[0]);
          else if (w === WIRE_LENGTH) for (let pos = p2; pos < e2; ) {
            const [v, next] = reader.varint(pos);
            ints.push(v);
            pos = next;
          }
        }
      });
      if (name === 'perm') node.perm = ints;
    }
  });
  return node;
}

/** Graph node fields computing the MatMul's output as Reshape → Gemm(transB = 1) → Reshape. */
function gemmNodes(matmul: NodeInfo, weight: string, inShape: string, outShape: string, prefix: string): Uint8Array {
  const flat = `${prefix}/input_2d`;
  const product = `${prefix}/output_2d`;
  const out = new Writer();
  out.bytes(nodeField('Reshape', `${prefix}/reshape_in`, [matmul.inputs[0]!, inShape], [flat]));
  out.bytes(nodeField('Gemm', matmul.name || `${prefix}/gemm`, [flat, weight], [product], { transB: 1 }));
  out.bytes(nodeField('Reshape', `${prefix}/reshape_out`, [product, outShape], matmul.outputs));
  return out.finish();
}

/** A GraphProto.node field (tag included) for a default-domain op with INT attributes. */
function nodeField(opType: string, name: string, inputs: string[], outputs: string[], ints: Record<string, number> = {}): Uint8Array {
  const node = new Writer();
  const str = (field: number, value: string) => {
    node.tag(field, WIRE_LENGTH);
    node.lengthDelimited(utf8(value));
  };
  for (const input of inputs) str(NODE_INPUT, input);
  for (const output of outputs) str(NODE_OUTPUT, output);
  str(NODE_NAME, name);
  str(NODE_OP_TYPE, opType);
  for (const [key, value] of Object.entries(ints)) {
    const attribute = new Writer();
    attribute.tag(ATTRIBUTE_NAME, WIRE_LENGTH);
    attribute.lengthDelimited(utf8(key));
    attribute.tag(ATTRIBUTE_I, WIRE_VARINT);
    attribute.varint(value);
    attribute.tag(ATTRIBUTE_TYPE, WIRE_VARINT);
    attribute.varint(ATTRIBUTE_TYPE_INT);
    node.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
    node.lengthDelimited(attribute.finish());
  }
  const field = new Writer();
  field.tag(GRAPH_NODE, WIRE_LENGTH);
  field.lengthDelimited(node.finish());
  return field.finish();
}

/** A GraphProto.initializer field (tag included): a 1-D int64 tensor. */
function int64Initializer(name: string, values: number[]): Uint8Array {
  const raw = new DataView(new ArrayBuffer(values.length * 8));
  values.forEach((v, i) => raw.setBigInt64(i * 8, BigInt(v), true));
  const tensor = new Writer();
  tensor.tag(TENSOR_DIMS, WIRE_VARINT);
  tensor.varint(values.length);
  tensor.tag(TENSOR_DATA_TYPE, WIRE_VARINT);
  tensor.varint(TENSOR_TYPE_INT64);
  tensor.tag(TENSOR_NAME, WIRE_LENGTH);
  tensor.lengthDelimited(utf8(name));
  tensor.tag(TENSOR_RAW_DATA, WIRE_LENGTH);
  tensor.lengthDelimited(new Uint8Array(raw.buffer));
  const field = new Writer();
  field.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
  field.lengthDelimited(tensor.finish());
  return field.finish();
}

const decoder = new TextDecoder();
function text(reader: BufferedReader, start: number, end: number): string {
  return decoder.decode(reader.slice(start, end));
}

function rewriteTensor(reader: BufferedReader, start: number, end: number, location: string, stats: Stats): Uint8Array {
  let raw: { start: number; length: number } | null = null;
  let alreadyExternal = false;
  forEachField(reader, start, end, (field, wire, _fieldStart, payloadStart, fieldEnd) => {
    if (field === TENSOR_RAW_DATA && wire === WIRE_LENGTH) raw = { start: payloadStart, length: fieldEnd - payloadStart };
    if (field === TENSOR_EXTERNAL_DATA) alreadyExternal = true;
    // Exporters may write data_location = DEFAULT (0) explicitly; only EXTERNAL (1) means external.
    if (field === TENSOR_DATA_LOCATION && reader.varint(payloadStart)[0] === 1) alreadyExternal = true;
  });
  const inline = raw as { start: number; length: number } | null;
  if (!inline || alreadyExternal || inline.length < MIN_EXTERNAL_BYTES) return reader.slice(start, end);

  const out = new Writer();
  forEachField(reader, start, end, (field, _wire, fieldStart, _payloadStart, fieldEnd) => {
    if (field !== TENSOR_RAW_DATA && field !== TENSOR_DATA_LOCATION) out.bytes(reader.slice(fieldStart, fieldEnd));
  });
  for (const [key, value] of [
    ['location', location],
    ['offset', String(inline.start)],
    ['length', String(inline.length)],
  ] as const) {
    const entry = new Writer();
    entry.tag(1, WIRE_LENGTH);
    entry.lengthDelimited(utf8(key));
    entry.tag(2, WIRE_LENGTH);
    entry.lengthDelimited(utf8(value));
    out.tag(TENSOR_EXTERNAL_DATA, WIRE_LENGTH);
    out.lengthDelimited(entry.finish());
  }
  out.tag(TENSOR_DATA_LOCATION, WIRE_VARINT);
  out.varint(1);
  stats.tensors++;
  stats.bytes += inline.length;
  return out.finish();
}

/**
 * Wraps a ByteSource as the Uint8Array ONNX Runtime expects for an external
 * data file. Its loader only reads `byteLength` and calls `subarray(begin, end)`
 * once per tensor, so those two are answered from the source on demand and the
 * array itself stays empty.
 */
export function lazyBytes(source: ByteSource): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(0);
  Object.defineProperty(bytes, 'byteLength', { get: () => source.size });
  Object.defineProperty(bytes, 'length', { get: () => source.size });
  Object.defineProperty(bytes, 'subarray', {
    value: (begin = 0, end = source.size) => source.read(begin, end - begin),
  });
  return bytes;
}

/** Calls `visit` for each protobuf field in [start, end) with its tag/payload/end positions. */
function forEachField(
  reader: BufferedReader,
  start: number,
  end: number,
  visit: (field: number, wire: number, fieldStart: number, payloadStart: number, fieldEnd: number) => void,
): void {
  let pos = start;
  while (pos < end) {
    const fieldStart = pos;
    const [key, afterKey] = reader.varint(pos);
    const field = Math.floor(key / 8);
    const wire = key % 8;
    let payloadStart = afterKey;
    let fieldEnd: number;
    if (wire === WIRE_VARINT) fieldEnd = reader.varint(afterKey)[1];
    else if (wire === WIRE_64BIT) fieldEnd = afterKey + 8;
    else if (wire === WIRE_32BIT) fieldEnd = afterKey + 4;
    else if (wire === WIRE_LENGTH) {
      const [length, afterLength] = reader.varint(afterKey);
      payloadStart = afterLength;
      fieldEnd = afterLength + length;
    } else {
      throw new Error(`Unsupported protobuf wire type ${wire} at byte ${pos}`);
    }
    if (fieldEnd > end) throw new Error(`Malformed model file: field ${field} at byte ${pos} runs past its parent`);
    visit(field, wire, fieldStart, payloadStart, fieldEnd);
    pos = fieldEnd;
  }
}

/** Reads a ByteSource through a small window, so many tiny reads don't each hit the disk. */
class BufferedReader {
  private window = new Uint8Array(0);
  private windowStart = 0;

  constructor(private readonly source: ByteSource) {}

  /** A varint at `pos`: its value and the position after it. */
  varint(pos: number): [number, number] {
    let value = 0;
    let scale = 1;
    for (let i = 0; i < 10; i++) {
      const byte = this.byte(pos + i);
      value += (byte & 0x7f) * scale;
      if (byte < 0x80) return [value, pos + i + 1];
      scale *= 128;
    }
    throw new Error(`Malformed varint at byte ${pos}`);
  }

  slice(start: number, end: number): Uint8Array {
    if (start >= this.windowStart && end <= this.windowStart + this.window.length) {
      return this.window.slice(start - this.windowStart, end - this.windowStart);
    }
    return this.source.read(start, end - start).slice();
  }

  private byte(pos: number): number {
    if (pos < this.windowStart || pos >= this.windowStart + this.window.length) {
      this.windowStart = pos;
      this.window = this.source.read(pos, Math.min(64 * 1024, this.source.size - pos));
    }
    return this.window[pos - this.windowStart]!;
  }
}

class Writer {
  private buf = new Uint8Array(1024);
  private len = 0;

  bytes(data: Uint8Array): void {
    this.reserve(data.length);
    this.buf.set(data, this.len);
    this.len += data.length;
  }

  varint(value: number): void {
    this.reserve(10);
    let v = value;
    while (v >= 0x80) {
      this.buf[this.len++] = (v % 128) | 0x80;
      v = Math.floor(v / 128);
    }
    this.buf[this.len++] = v;
  }

  tag(field: number, wire: number): void {
    this.varint(field * 8 + wire);
  }

  lengthDelimited(data: Uint8Array): void {
    this.varint(data.length);
    this.bytes(data);
  }

  finish(): Uint8Array<ArrayBuffer> {
    return this.buf.slice(0, this.len);
  }

  private reserve(extra: number): void {
    if (this.len + extra <= this.buf.length) return;
    let size = this.buf.length * 2;
    while (size < this.len + extra) size *= 2;
    const next = new Uint8Array(size);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
  }
}

const encoder = new TextEncoder();
function utf8(text: string): Uint8Array {
  return encoder.encode(text);
}
