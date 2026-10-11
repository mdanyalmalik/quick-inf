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
// With `quantizeOutput`, the product instead becomes `MatMulNBits` over a 4- or
// 8-bit copy of E (blocks of 32 values, one scale each, no zero points), as MLC's
// compiled models do. The table is otherwise read in full precision for every generated token:
// 525 MB for Llama 3.2 1B (fp16), 1.2 GB for Qwen3 1.7B (fp32), about as much as
// all the other layers together. The copy is computed when ONNX Runtime reads it,
// from an extra external data file (`extraFiles`), so it never sits in memory whole
// alongside the original. The embedding lookup then reads the same copy
// (GatherBlockQuantized), so the full-precision table isn't loaded at all.
//
// The output layer also gets only the last position's hidden state: logits are
// read for the last token only, but a prompt chunk of n tokens would otherwise
// multiply all n by the whole vocabulary (110 × 152K for Qwen), a fifth or more of
// the prompt's work for small models. `logits` is then [1, 1, vocab].
//
// Weight files over 2 GB (ONNX Runtime GenAI builds keep a whole 7B model in one
// `model.onnx.data`) are mounted as overlapping 2 GB windows, `<file>#<k>` covering
// [k GiB, k GiB + 2 GiB), and each tensor points into the window it starts in, so
// ONNX Runtime never sees an offset or file size past 2^31.
//
// The graph's optional shape hints (value_info) are dropped; some exports' are wrong.
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
// GraphProto.node = 1, initializer = 5, input = 11, output = 12, value_info = 13 (ValueInfoProto.name = 1,
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
  /** Fills `target` with the bytes starting at `offset`, without allocating (optional). */
  readInto?(offset: number, target: Uint8Array): void;
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
const GRAPH_VALUE_INFO = 13;
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
  /** Whether the output layer reads only the last position (see above). */
  lastTokenLogits: boolean;
  /** External data files the rewritten structure refers to besides the original (quantized output layers). */
  extraFiles: { path: string; source: ByteSource }[];
}

export interface RewriteOptions {
  /** Quantize the output layer's transposed embedding table to this many bits (see above). */
  quantizeOutput?: 4 | 8;
  /** The model's other external data files, by name (for a table stored outside the graph file). */
  externalData?: Record<string, ByteSource>;
}

/** The causal mask input added in place of a broken mask subgraph: [1, 1, tokens, past + tokens], 0 or a large negative. */
export const ATTENTION_BIAS_INPUT = 'attention_bias';
const MASK_SUBGRAPH = '/model/attn_mask_reformat/';

/**
 * Rewrites a single-file ONNX model so its large initializers refer to
 * `location` (which must be mounted as the original file) instead of being inline.
 */
export function externalizeWeights(source: ByteSource, location: string, options: RewriteOptions = {}): ExternalizedModel {
  const reader = new BufferedReader(source);
  const out = new Writer();
  const stats: Stats = { tensors: 0, bytes: 0, fusedTransposes: 0, attentionBias: false, lastTokenLogits: false, extraFiles: [] };
  const context: RewriteContext = { source, location, ...options };
  forEachField(reader, 0, source.size, (field, wire, start, payloadStart, end) => {
    if (field === MODEL_GRAPH && wire === WIRE_LENGTH) {
      const graph = rewriteGraph(reader, payloadStart, end, context, stats);
      out.tag(MODEL_GRAPH, WIRE_LENGTH);
      out.lengthDelimited(graph);
    } else {
      out.bytes(reader.slice(start, end));
    }
  });
  return { graph: out.finish(), ...stats };
}

type Stats = Omit<ExternalizedModel, 'graph'>;
type RewriteContext = RewriteOptions & { source: ByteSource; location: string };

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
  /** axis attribute, if any (Gather). */
  axis: number | null;
  attributes: number;
}

function rewriteGraph(reader: BufferedReader, start: number, end: number, context: RewriteContext, stats: Stats): Uint8Array {
  const { location } = context;
  const last = planLastTokenLogits(reader, start, end);
  if (last) stats.lastTokenLogits = true;
  const fused = planTransposeFusions(reader, start, end, context, last);
  stats.extraFiles.push(...fused.files);
  stats.fusedTransposes += fused.matmuls;
  const mask = planMaskInput(reader, start, end);
  if (mask) stats.attentionBias = true;
  const out = new Writer();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field === GRAPH_NODE && wire === WIRE_LENGTH && mask?.replace.has(fieldStart)) {
      const replacement = mask.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && fused.replace.has(fieldStart)) {
      const replacement = fused.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && last?.head.start === fieldStart) {
      out.bytes(renameFirstInput(reader, last.head, last.input));
    } else if (field === GRAPH_VALUE_INFO) {
      // Optional shape hints, dropped: some are wrong (Phi-4-mini-reasoning's GenAI build declares a
      // scalar as [1]), which ONNX Runtime's CPU backend tolerates but session creation here doesn't.
    } else if (field === GRAPH_INITIALIZER && wire === WIRE_LENGTH) {
      const tensor = rewriteTensor(reader, payloadStart, fieldEnd, location, stats, context);
      out.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
      out.lengthDelimited(tensor);
    } else {
      out.bytes(reader.slice(fieldStart, fieldEnd));
    }
  });
  out.bytes(fused.extra);
  if (last) out.bytes(last.extra);
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
  context: RewriteContext,
  last: LastTokenPlan | null,
): { replace: Map<number, Uint8Array | null>; extra: Uint8Array; files: { path: string; source: ByteSource }[]; matmuls: number } {
  const nodes: NodeInfo[] = [];
  const twoDimInitializers = new Map<string, [number, number]>();
  const tensorData = new Map<string, TensorData>();
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
      const data = readTensorData(reader, payloadStart, fieldEnd, context);
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
      if (data) tensorData.set(name, data);
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
  const files: { path: string; source: ByteSource }[] = [];
  let matmuls = 0;
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
    matmuls += uses.length;
    const [outFeatures, inFeatures] = twoDimInitializers.get(weight)!;
    const data = tensorData.get(weight);
    if (context.quantizeOutput && data && inFeatures % QUANT_BLOCK === 0) {
      // W is [out, in], already the row-per-output layout MatMulNBits takes.
      const bits = context.quantizeOutput;
      const path = `${weight.replace(/[^\w.-]/g, '_')}.q${bits}`;
      // Lookups of single rows (the input embedding) read the quantized copy too, when every other use is one.
      const others = (consumers.get(weight) ?? []).filter((u) => u.node !== node);
      const lookups = others.every((u) => u.node.opType === 'Gather' && u.node.domain === '' && u.index === 0 && (u.node.axis ?? 0) === 0)
        ? others.map((u) => u.node)
        : [];
      const quantized = quantizedSource(data, outFeatures, inFeatures, bits, lookups.length ? 3 : 2);
      files.push({ path, source: quantized.source });
      const qName = `${weight}/q${bits}`;
      const scalesName = `${weight}/q${bits}_scales`;
      const blocks = inFeatures / QUANT_BLOCK;
      extra.bytes(externalInitializer(qName, [outFeatures, blocks, (QUANT_BLOCK * bits) / 8], TENSOR_TYPE_UINT8, path, 0, quantized.weightBytes));
      extra.bytes(externalInitializer(scalesName, [outFeatures * blocks], data.type, path, quantized.weightBytes, quantized.scaleBytes));
      if (lookups.length) {
        // Same weights as [rows, blocks, 32] (that's MatMulNBits' layout too), scales as [rows, blocks, 1].
        const lookupScales = `${scalesName}/lookup`;
        const rowShape = `${weight}/row_shape`;
        extra.bytes(externalInitializer(lookupScales, [outFeatures, blocks, 1], data.type, path, quantized.weightBytes, quantized.scaleBytes));
        extra.bytes(int64Initializer(rowShape, [0, 0, inFeatures]));
        lookups.forEach((gather, i) => {
          const blocked = `${weight}/lookup${i}_blocks`;
          plan.set(
            gather.start,
            concat(
              nodeField('GatherBlockQuantized', gather.name || `${weight}/lookup${i}`, [qName, gather.inputs[1]!, lookupScales], [blocked], {
                bits,
                block_size: QUANT_BLOCK,
                gather_axis: 0,
                quantize_axis: 2,
              }, MICROSOFT_DOMAIN),
              // [1, tokens, blocks, 32] → [1, tokens, in].
              nodeField('Reshape', `${weight}/lookup${i}_reshape`, [blocked, rowShape], gather.outputs),
            ),
          );
        });
      }
      plan.set(node.start, null);
      uses.forEach(({ node: matmul }, i) =>
        plan.set(
          matmul.start,
          nodeField('MatMulNBits', matmul.name || `${weight}/matmul_nbits${i}`, [inputOf(matmul, last), qName, scalesName], matmul.outputs, {
            K: inFeatures,
            N: outFeatures,
            bits,
            block_size: QUANT_BLOCK,
          }, MICROSOFT_DOMAIN),
        ),
      );
      continue;
    }
    // W is [out, in]: x [1, tokens, in] → [tokens, in] → Gemm → [tokens, out] → [1, tokens, out].
    const inShape = `${weight}/gemm_in_shape`;
    const outShape = `${weight}/gemm_out_shape`;
    extra.bytes(int64Initializer(inShape, [-1, inFeatures]));
    extra.bytes(int64Initializer(outShape, [1, -1, outFeatures]));
    plan.set(node.start, null);
    uses.forEach(({ node: matmul }, i) =>
      plan.set(matmul.start, gemmNodes({ ...matmul, inputs: [inputOf(matmul, last), ...matmul.inputs.slice(1)] }, weight, inShape, outShape, `${weight}/gemm${i}`)),
    );
  }
  return { replace: plan, extra: extra.finish(), files, matmuls };
}

interface LastTokenPlan {
  /** The output layer's node (a matrix product producing the logits, maybe through Casts). */
  head: NodeInfo;
  /** Its new first input: the last position of the hidden states. */
  input: string;
  /** The Slice node and its constants. */
  extra: Uint8Array;
}

/** Finds the output layer (see the top of this file) and plans slicing its input to the last position. */
function planLastTokenLogits(reader: BufferedReader, start: number, end: number): LastTokenPlan | null {
  const nodes: NodeInfo[] = [];
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field === GRAPH_NODE && wire === WIRE_LENGTH) nodes.push(readNode(reader, fieldStart, payloadStart, fieldEnd));
  });
  const producer = new Map(nodes.flatMap((n) => n.outputs.map((o) => [o, n] as const)));
  let head = producer.get(LOGITS_OUTPUT);
  while (head?.opType === 'Cast' && head.inputs.length === 1) head = producer.get(head.inputs[0]!);
  if (!head || !['MatMul', 'MatMulNBits'].includes(head.opType) || head.inputs.length < 2) return null;
  const hidden = head.inputs[0]!;
  const input = `${hidden}/last_position`;
  const extra = new Writer();
  const [starts, ends, axes] = ['starts', 'ends', 'axes'].map((k) => `${input}/${k}`);
  extra.bytes(int64Initializer(starts!, [-1]));
  extra.bytes(int64Initializer(ends!, [Number.MAX_SAFE_INTEGER]));
  extra.bytes(int64Initializer(axes!, [1]));
  extra.bytes(nodeField('Slice', `${input}/slice`, [hidden, starts!, ends!, axes!], [input]));
  return { head, input, extra: extra.finish() };
}

const LOGITS_OUTPUT = 'logits';

function inputOf(node: NodeInfo, last: LastTokenPlan | null): string {
  return last?.head.start === node.start ? last.input : node.inputs[0]!;
}

/** The node's GraphProto.node field with its first input renamed. */
function renameFirstInput(reader: BufferedReader, node: NodeInfo, input: string): Uint8Array {
  const payload = new Writer();
  let renamed = false;
  forEachField(reader, node.start, node.end, (field, wire, _s, payloadStart, fieldEnd) => {
    // The node field itself: copy its payload's fields, swapping the first input.
    if (field !== GRAPH_NODE || wire !== WIRE_LENGTH) return;
    forEachField(reader, payloadStart, fieldEnd, (f, w, fs, _p, fe) => {
      if (f === NODE_INPUT && w === WIRE_LENGTH && !renamed) {
        payload.tag(NODE_INPUT, WIRE_LENGTH);
        payload.lengthDelimited(utf8(input));
        renamed = true;
      } else {
        payload.bytes(reader.slice(fs, fe));
      }
    });
  });
  const field = new Writer();
  field.tag(GRAPH_NODE, WIRE_LENGTH);
  field.lengthDelimited(payload.finish());
  return field.finish();
}

function concat(...parts: Uint8Array[]): Uint8Array {
  const out = new Writer();
  parts.forEach((p) => out.bytes(p));
  return out.finish();
}

/** Values per quantization block (one scale each). */
const QUANT_BLOCK = 32;
const MICROSOFT_DOMAIN = 'com.microsoft';
const TENSOR_TYPE_UINT8 = 2;

/** Where a float tensor's values are: a byte range of the graph file or of one of its external data files. */
interface TensorData {
  type: typeof TENSOR_TYPE_FLOAT | typeof TENSOR_TYPE_FLOAT16;
  source: ByteSource;
  offset: number;
  length: number;
}

function readTensorData(reader: BufferedReader, start: number, end: number, context: RewriteContext): TensorData | null {
  let type = 0;
  let raw: { offset: number; length: number } | null = null;
  const external: Record<string, string> = {};
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === TENSOR_DATA_TYPE && w === WIRE_VARINT) type = reader.varint(p)[0];
    else if (f === TENSOR_RAW_DATA && w === WIRE_LENGTH) raw = { offset: p, length: e - p };
    else if (f === TENSOR_EXTERNAL_DATA && w === WIRE_LENGTH) {
      let key = '';
      let value = '';
      forEachField(reader, p, e, (ef, ew, _es, ep, ee) => {
        if (ew !== WIRE_LENGTH) return;
        if (ef === 1) key = text(reader, ep, ee);
        if (ef === 2) value = text(reader, ep, ee);
      });
      external[key] = value;
    }
  });
  if (type !== TENSOR_TYPE_FLOAT && type !== TENSOR_TYPE_FLOAT16) return null;
  const inline = raw as { offset: number; length: number } | null;
  if (inline) return { type, source: context.source, ...inline };
  const file = external.location && context.externalData?.[external.location];
  if (!file || external.length === undefined) return null;
  return { type, source: file, offset: Number(external.offset ?? 0), length: Number(external.length) };
}

/**
 * A virtual file holding W quantized for MatMulNBits: the packed weights ([rows, blocks, 32 × bits / 8],
 * 4-bit values two per byte, low nibble first), then one scale per block (in W's type). Symmetric:
 * value = (q − 2^(bits−1)) × scale, scale = largest |w| in the block / (2^(bits−1) − 1).
 * Computed in full on the first read and dropped once every tensor using it has been read.
 */
function quantizedSource(
  data: TensorData,
  rows: number,
  cols: number,
  bits: 4 | 8,
  /** How many tensors read from it (each is read once). */
  readers: number,
): { source: ByteSource; weightBytes: number; scaleBytes: number } {
  const blocks = cols / QUANT_BLOCK;
  const weightBytes = (rows * cols * bits) / 8;
  const scaleBytes = rows * blocks * (data.type === TENSOR_TYPE_FLOAT16 ? 2 : 4);
  const size = weightBytes + scaleBytes;
  let built: Uint8Array<ArrayBuffer> | null = null;
  let reads = 0;
  const source: ByteSource = {
    size,
    read(offset, length) {
      built ??= quantize(data, rows, cols, bits, weightBytes, scaleBytes);
      // A view: the loader copies it out before the next read (see lazyBytes).
      const part = built.subarray(offset, offset + length);
      // Dropped after the last expected read (rebuilt if read again, e.g. by a second session).
      if (++reads % readers === 0) built = null;
      return part;
    },
  };
  return { source, weightBytes, scaleBytes };
}

function quantize(data: TensorData, rows: number, cols: number, bits: 4 | 8, weightBytes: number, scaleBytes: number): Uint8Array<ArrayBuffer> {
  const half = data.type === TENSOR_TYPE_FLOAT16;
  const elementBytes = half ? 2 : 4;
  const out = new Uint8Array(weightBytes + scaleBytes);
  const scales16 = half ? new Uint16Array(out.buffer, weightBytes, scaleBytes / 2) : null;
  const scales32 = half ? null : new Float32Array(out.buffer, weightBytes, scaleBytes / 4);
  const zero = 1 << (bits - 1);
  const maxQ = zero - 1;
  const top = (1 << bits) - 1;
  const blocks = cols / QUANT_BLOCK;
  const row = new Float32Array(cols);
  // Rows are read in batches of about 8 MB.
  const batch = Math.max(1, Math.floor((8 * 1024 * 1024) / (cols * elementBytes)));
  for (let r0 = 0; r0 < rows; r0 += batch) {
    const count = Math.min(batch, rows - r0);
    const at = data.offset + r0 * cols * elementBytes;
    const length = count * cols * elementBytes;
    let bytes: Uint8Array<ArrayBuffer>;
    if (data.source.readInto) {
      batchBuffer = batchBuffer.byteLength >= length ? batchBuffer : new Uint8Array(length);
      bytes = batchBuffer.subarray(0, length);
      data.source.readInto(at, bytes);
    } else {
      const read = data.source.read(at, length);
      bytes = read.byteOffset % 4 === 0 ? read : read.slice();
    }
    const values = half
      ? new Uint16Array(bytes.buffer, bytes.byteOffset, count * cols)
      : new Float32Array(bytes.buffer, bytes.byteOffset, count * cols);
    for (let r = 0; r < count; r++) {
      const base = r * cols;
      if (half) for (let c = 0; c < cols; c++) row[c] = HALF_TO_FLOAT[(values as Uint16Array)[base + c]!]!;
      else row.set((values as Float32Array).subarray(base, base + cols));
      const rowIndex = r0 + r;
      for (let b = 0; b < blocks; b++) {
        let maxAbs = 0;
        for (let i = b * QUANT_BLOCK; i < (b + 1) * QUANT_BLOCK; i++) maxAbs = Math.max(maxAbs, Math.abs(row[i]!));
        // Rounded to the stored type first, so values are quantized with the scale the GPU will use.
        const scaleIndex = rowIndex * blocks + b;
        let scale: number;
        if (scales16) {
          scales16[scaleIndex] = floatToHalf(maxAbs / maxQ);
          scale = HALF_TO_FLOAT[scales16[scaleIndex]!]!;
        } else {
          scale = Math.fround(maxAbs / maxQ);
          scales32![scaleIndex] = scale;
        }
        const inverse = scale > 0 ? 1 / scale : 0;
        const blobStart = scaleIndex * ((QUANT_BLOCK * bits) / 8);
        for (let i = 0; i < QUANT_BLOCK; i++) {
          const q = Math.min(top, Math.max(0, Math.round(row[b * QUANT_BLOCK + i]! * inverse) + zero));
          if (bits === 8) out[blobStart + i] = q;
          else if (i % 2 === 0) out[blobStart + i / 2] = q;
          else out[blobStart + (i - 1) / 2]! |= q << 4;
        }
      }
    }
  }
  batchBuffer = new Uint8Array(0);
  return out;
}

let batchBuffer = new Uint8Array(0);

const HALF_TO_FLOAT = (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    table[h] =
      exponent === 0 ? sign * 2 ** -14 * (fraction / 1024) : exponent === 31 ? (fraction ? NaN : sign * Infinity) : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return table;
})();

/** Nearest float16 (round half to even) of a finite value. */
function floatToHalf(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent >= 31) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exponent;
    const halfMantissa = mantissa >> shift;
    const rest = mantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (halfMantissa + (rest > halfway || (rest === halfway && halfMantissa & 1) ? 1 : 0));
  }
  const halfMantissa = mantissa >> 13;
  const rest = mantissa & 0x1fff;
  const rounded = (exponent << 10) | halfMantissa;
  return sign | (rounded + (rest > 0x1000 || (rest === 0x1000 && halfMantissa & 1) ? 1 : 0));
}

/** A GraphProto.initializer field (tag included) whose data is a byte range of an external data file. */
function externalInitializer(name: string, dims: number[], type: number, location: string, offset: number, length: number): Uint8Array {
  const tensor = new Writer();
  for (const d of dims) {
    tensor.tag(TENSOR_DIMS, WIRE_VARINT);
    tensor.varint(d);
  }
  tensor.tag(TENSOR_DATA_TYPE, WIRE_VARINT);
  tensor.varint(type);
  tensor.tag(TENSOR_NAME, WIRE_LENGTH);
  tensor.lengthDelimited(utf8(name));
  for (const [key, value] of [
    ['location', location],
    ['offset', String(offset)],
    ['length', String(length)],
  ] as const) {
    const entry = new Writer();
    entry.tag(1, WIRE_LENGTH);
    entry.lengthDelimited(utf8(key));
    entry.tag(2, WIRE_LENGTH);
    entry.lengthDelimited(utf8(value));
    tensor.tag(TENSOR_EXTERNAL_DATA, WIRE_LENGTH);
    tensor.lengthDelimited(entry.finish());
  }
  tensor.tag(TENSOR_DATA_LOCATION, WIRE_VARINT);
  tensor.varint(1);
  const field = new Writer();
  field.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
  field.lengthDelimited(tensor.finish());
  return field.finish();
}

function readNode(reader: BufferedReader, start: number, payloadStart: number, end: number): NodeInfo {
  const node: NodeInfo = { start, end, inputs: [], outputs: [], name: '', opType: '', domain: '', perm: null, axis: null, attributes: 0 };
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
      let int: number | null = null;
      forEachField(reader, p, e, (f, w, _s2, p2, e2) => {
        if (f === ATTRIBUTE_NAME && w === WIRE_LENGTH) name = text(reader, p2, e2);
        if (f === ATTRIBUTE_I && w === WIRE_VARINT) int = reader.varint(p2)[0];
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
      if (name === 'axis') node.axis = int;
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

/** A GraphProto.node field (tag included) for an op with INT attributes. */
function nodeField(opType: string, name: string, inputs: string[], outputs: string[], ints: Record<string, number> = {}, domain = ''): Uint8Array {
  const node = new Writer();
  const str = (field: number, value: string) => {
    node.tag(field, WIRE_LENGTH);
    node.lengthDelimited(utf8(value));
  };
  for (const input of inputs) str(NODE_INPUT, input);
  for (const output of outputs) str(NODE_OUTPUT, output);
  str(NODE_NAME, name);
  str(NODE_OP_TYPE, opType);
  if (domain) str(NODE_DOMAIN, domain);
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

/** Window size for weight files over WINDOW_LIMIT (see above); each window spans two. */
const WINDOW_BYTES = 2 ** 30;
const WINDOW_LIMIT = 2 ** 31 - 1;

function rewriteTensor(reader: BufferedReader, start: number, end: number, location: string, stats: Stats, context: RewriteContext): Uint8Array {
  let raw: { start: number; length: number } | null = null;
  let alreadyExternal = false;
  const external: Record<string, string> = {};
  forEachField(reader, start, end, (field, wire, _fieldStart, payloadStart, fieldEnd) => {
    if (field === TENSOR_RAW_DATA && wire === WIRE_LENGTH) raw = { start: payloadStart, length: fieldEnd - payloadStart };
    if (field === TENSOR_EXTERNAL_DATA && wire === WIRE_LENGTH) {
      alreadyExternal = true;
      let key = '';
      let value = '';
      forEachField(reader, payloadStart, fieldEnd, (ef, ew, _es, ep, ee) => {
        if (ew !== WIRE_LENGTH) return;
        if (ef === 1) key = text(reader, ep, ee);
        if (ef === 2) value = text(reader, ep, ee);
      });
      external[key] = value;
    }
    // Exporters may write data_location = DEFAULT (0) explicitly; only EXTERNAL (1) means external.
    if (field === TENSOR_DATA_LOCATION && reader.varint(payloadStart)[0] === 1) alreadyExternal = true;
  });
  const inline = raw as { start: number; length: number } | null;
  let target: { location: string; offset: number; length: number };
  if (alreadyExternal) {
    const file = external.location ? context.externalData?.[external.location] : undefined;
    if (!file || file.size <= WINDOW_LIMIT || external.length === undefined) return reader.slice(start, end);
    // A tensor in a file too big to mount whole: point it into its window.
    const offset = Number(external.offset ?? 0);
    const k = Math.floor(offset / WINDOW_BYTES);
    const path = `${external.location}#${k}`;
    if (!stats.extraFiles.some((f) => f.path === path)) {
      const base = k * WINDOW_BYTES;
      stats.extraFiles.push({ path, source: windowOf(file, base, Math.min(2 * WINDOW_BYTES, file.size - base)) });
    }
    target = { location: path, offset: offset - k * WINDOW_BYTES, length: Number(external.length) };
  } else {
    if (!inline || inline.length < MIN_EXTERNAL_BYTES) return reader.slice(start, end);
    target = { location, offset: inline.start, length: inline.length };
  }

  const out = new Writer();
  forEachField(reader, start, end, (field, _wire, fieldStart, _payloadStart, fieldEnd) => {
    if (field !== TENSOR_RAW_DATA && field !== TENSOR_DATA_LOCATION && field !== TENSOR_EXTERNAL_DATA) out.bytes(reader.slice(fieldStart, fieldEnd));
  });
  for (const [key, value] of [
    ['location', target.location],
    ['offset', String(target.offset)],
    ['length', String(target.length)],
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
  if (!alreadyExternal) {
    stats.tensors++;
    stats.bytes += target.length;
  }
  return out.finish();
}

/** `length` bytes of `file` from `base`, as a file of its own. */
function windowOf(file: ByteSource, base: number, length: number): ByteSource {
  return {
    size: length,
    read: (offset, n) => file.read(base + offset, n),
    readInto: file.readInto && ((offset, target) => file.readInto!(base + offset, target)),
  };
}

/**
 * Wraps a ByteSource as the Uint8Array ONNX Runtime expects for an external
 * data file. Its loader only reads `byteLength` and calls `subarray(begin, end)`
 * once per tensor, so those two are answered from the source on demand and the
 * array itself stays empty.
 *
 * The loader copies each tensor out (into WebAssembly memory or a GPU buffer)
 * before it asks for the next, so a source that can `readInto` fills one shared
 * buffer instead of allocating a new array per tensor. Otherwise a load leaves
 * the whole model's bytes behind as garbage (2.9 GB for Llama 3.2 3B), which the
 * browser frees only later and which can push a tab on a smaller device over its
 * limit. Call `releaseReadBuffer` once the session exists.
 */
export function lazyBytes(source: ByteSource): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(0);
  Object.defineProperty(bytes, 'byteLength', { get: () => source.size });
  Object.defineProperty(bytes, 'length', { get: () => source.size });
  Object.defineProperty(bytes, 'subarray', {
    value: (begin = 0, end = source.size) => {
      if (!source.readInto) return source.read(begin, end - begin);
      const target = readBuffer(end - begin);
      source.readInto(begin, target);
      return target;
    },
  });
  return bytes;
}

let shared = new Uint8Array(0);

/** A view of `length` bytes of the shared read buffer (grown as needed; its contents are overwritten by the next call). */
function readBuffer(length: number): Uint8Array<ArrayBuffer> {
  if (shared.byteLength < length) shared = new Uint8Array(length);
  return shared.subarray(0, length);
}

/** Drops the shared read buffer (after a model's sessions are created). */
export function releaseReadBuffer(): void {
  shared = new Uint8Array(0);
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
