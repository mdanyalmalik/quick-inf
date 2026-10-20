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
// 4-bit weights stored as DequantizeLinear (signed int4, blocks along K, no zero point) feeding a MatMul —
// NVIDIA's and DirectML's INT4 exports (Mistral Nemo) — become MatMulNBits: ONNX Runtime WebGPU doesn't fuse the
// pair, and dequantizes every weight in full each step instead (1 token/s for a 12B model). The same nibbles are
// repacked as MatMulNBits wants them: [N, K / block, block / 2] bytes, unsigned with the default zero point of 8
// (so each nibble just flips its top bit), and the scales transposed to [N, K / block]. Each repacked tensor is
// computed when ONNX Runtime reads it, from extra external data files (`extraFiles`).
//
// With `quantizeWeights`, every MatMul by a float weight matrix [K, N] (and every plain Gemm, by [K, N] or, with
// transB, [N, K], its bias added after) becomes MatMulNBits over an 8- or 4-bit copy (blocks of 32 along K, one scale
// each, symmetric), computed as ONNX Runtime reads it: for float16-only exports (Granite 4.1 8B, SD 3.5 Large's
// transformer), whose full-precision weights take twice the GPU memory and time of 8-bit ones.
//
// ONNX Runtime Web is 32-bit WebAssembly: a single tensor over 2 GiB fails at load ("Integer overflow"). A float
// lookup table that big (Qwen3.8 27B's 2.5 GB token embeddings) becomes an 8-bit copy read by GatherBlockQuantized,
// computed as ONNX Runtime reads it (blocks of 32, as the output layer's above), about half the size.
//
// ONNX's own Attention (opset 23+, Gemma 4 26B's export) has no WebGPU kernel and would run on the CPU, moving
// every layer's activations and KV cache back and forth. With an explicit mask and is_causal = 0 (the graph builds
// its own causal and sliding-window masks) it becomes standard ops: the past and new keys and values concatenated
// (the present outputs), queries grouped by key head ([B, kv heads, group × S, H], so grouped-query attention
// needs no copies of K and V), scores scaled, soft-capped if asked, masked, softmaxed and applied to V.
//
// MultiHeadAttention with packed inputs (Olive's SDXL exports: query as [B, L, heads, 3, H] for self-attention, key
// as [B, L, heads, 2, H] for cross-attention) isn't implemented on WebGPU: each part is taken out (Gather on axis 3)
// and flattened to [B, L, heads × H], and the same node gets separate query, key and value.
//
// With `computeFloat32`, a float16 graph computes in float32 instead (Stable Diffusion 3.5's transformer, whose
// attention outputs pass float16's range above 512 × 512): float16 initializers and constants become float32 (the
// large ones converted as ONNX Runtime reads them, from extra files), Casts to float16 cast to float32, and the
// graph's float16 inputs and outputs are declared float32. Weights quantized by `quantizeWeights` get float32 scales.
//
// IsInf (T5's float16 overflow clamp, in AMD's FLUX.1 export) has no kernel here: it becomes Equal(Abs(x), +∞) (or
// Equal(x, ±∞) for one sign only). The clamp casts its input to float64 first, which nothing here supports either: a
// Cast to float64 read only by IsInf casts to float32 instead (an infinity stays one).
//
// ONNX Runtime's GroupNorm (com.microsoft, from Olive's Stable Diffusion exports) has no WebGPU or CPU kernel
// here, so each one becomes standard ops: Reshape to [N, groups, rest] → InstanceNormalization (in float32:
// the variance of an fp16 UNet's activations can overflow fp16) → Reshape back → × gamma + beta per channel
// → SiLU if the node asks for it. Its gamma's type is taken as the input's (true of every export seen).
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
const ATTRIBUTE_F = 2;
const ATTRIBUTE_I = 3;
const ATTRIBUTE_S = 4;
const ATTRIBUTE_T = 5;
const ATTRIBUTE_G = 6;
const ATTRIBUTE_INTS = 8;
const ATTRIBUTE_TYPE = 20;
const ATTRIBUTE_TYPE_FLOAT = 1;
const ATTRIBUTE_TYPE_INT = 2;
const ATTRIBUTE_TYPE_STRING = 3;
const ATTRIBUTE_TYPE_INTS = 7;
const TENSOR_DIMS = 1;
const TENSOR_DATA_TYPE = 2;
const TENSOR_TYPE_INT64 = 7;
const TENSOR_TYPE_FLOAT = 1;
const TENSOR_TYPE_FLOAT16 = 10;
const TENSOR_TYPE_DOUBLE = 11;
const TENSOR_NAME = 8;
const TENSOR_INT32_DATA = 5;
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
  /**
   * Only move the weights out (no output-layer, last-token or mask rewrites): for graphs that aren't
   * language models, such as an image model's UNet, whose shapes those rewrites assume.
   */
  externalizeOnly?: boolean;
  /**
   * Sets every MatMulNBits node's accuracy_level. FLUX.2 klein's export asks for 4 (int8 maths), which ONNX
   * Runtime WebGPU's kernel fails on with its block size of 16 ("divide by zero"); 0 uses the float path.
   */
  matMulNBitsAccuracy?: number;
  /** Quantize every MatMul's float weight matrix to this many bits (see above). */
  quantizeWeights?: 4 | 8;
  /** Compute a float16 graph in float32 (see above). */
  computeFloat32?: boolean;
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
  const rewrite = !context.externalizeOnly;
  const last = rewrite ? planLastTokenLogits(reader, start, end) : null;
  if (last) stats.lastTokenLogits = true;
  const fused = rewrite
    ? planTransposeFusions(reader, start, end, context, last)
    : { replace: new Map<number, Uint8Array | null>(), extra: new Uint8Array(0), files: [], matmuls: 0 };
  stats.extraFiles.push(...fused.files);
  stats.fusedTransposes += fused.matmuls;
  const mask = rewrite ? planMaskInput(reader, start, end) : null;
  if (mask) stats.attentionBias = true;
  const groupNorms = planGroupNorms(reader, start, end, !!context.computeFloat32);
  const attentions = planAttentions(reader, start, end);
  const packed = planPackedAttention(reader, start, end);
  const isInf = planIsInf(reader, start, end);
  const linear = rewrite ? planLinearAttention(reader, start, end) : null;
  const lookups = planLargeLookups(reader, start, end, context);
  stats.extraFiles.push(...lookups.files);
  const weights = context.quantizeWeights ? planQuantizedMatMuls(reader, start, end, context, context.quantizeWeights, last, fused.replace) : null;
  if (weights) stats.extraFiles.push(...weights.files);
  const upcast = context.computeFloat32 ? new Float32Files() : null;
  const dequant = planDequantMatMuls(reader, start, end, context, last);
  stats.extraFiles.push(...dequant.files);
  const out = new Writer();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field === GRAPH_NODE && wire === WIRE_LENGTH && mask?.replace.has(fieldStart)) {
      const replacement = mask.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && fused.replace.has(fieldStart)) {
      const replacement = fused.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && dequant.replace.has(fieldStart)) {
      const replacement = dequant.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && weights?.replace.has(fieldStart)) {
      out.bytes(weights.replace.get(fieldStart)!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && lookups.replace.has(fieldStart)) {
      out.bytes(lookups.replace.get(fieldStart)!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && linear?.replace.has(fieldStart)) {
      const replacement = linear.replace.get(fieldStart)!;
      if (replacement) out.bytes(replacement);
    } else if ((field === GRAPH_INPUT || field === GRAPH_OUTPUT) && wire === WIRE_LENGTH && linear?.valueInfos.has(valueInfoName(reader, payloadStart, fieldEnd))) {
      out.bytes(linear.valueInfos.get(valueInfoName(reader, payloadStart, fieldEnd))!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && isInf.replace.has(fieldStart)) {
      out.bytes(isInf.replace.get(fieldStart)!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && packed.replace.has(fieldStart)) {
      out.bytes(packed.replace.get(fieldStart)!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && attentions.replace.has(fieldStart)) {
      out.bytes(attentions.replace.get(fieldStart)!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && groupNorms.replace.has(fieldStart)) {
      out.bytes(groupNorms.replace.get(fieldStart)!);
    } else if (field === GRAPH_NODE && wire === WIRE_LENGTH && last?.head.start === fieldStart) {
      out.bytes(renameFirstInput(reader, last.head, last.input));
    } else if (
      field === GRAPH_NODE &&
      wire === WIRE_LENGTH &&
      context.matMulNBitsAccuracy !== undefined &&
      readNode(reader, fieldStart, payloadStart, fieldEnd).opType === 'MatMulNBits'
    ) {
      out.bytes(withIntAttribute(reader, payloadStart, fieldEnd, 'accuracy_level', context.matMulNBitsAccuracy));
    } else if (upcast && field === GRAPH_NODE && wire === WIRE_LENGTH) {
      out.bytes(upcastNode(reader, payloadStart, fieldEnd));
    } else if (upcast && (field === GRAPH_INPUT || field === GRAPH_OUTPUT) && wire === WIRE_LENGTH) {
      out.tag(field, WIRE_LENGTH);
      out.lengthDelimited(upcastValueInfo(reader, payloadStart, fieldEnd));
    } else if (field === GRAPH_VALUE_INFO) {
      // Optional shape hints, dropped: some are wrong (Phi-4-mini-reasoning's GenAI build declares a
      // scalar as [1]), which ONNX Runtime's CPU backend tolerates but session creation here doesn't.
    } else if (
      field === GRAPH_INITIALIZER &&
      wire === WIRE_LENGTH &&
      (dequant.dropped.size > 0 || lookups.dropped.size > 0 || (weights?.dropped.size ?? 0) > 0) &&
      [dequant.dropped, lookups.dropped, weights?.dropped].some((set) => set?.has(tensorName(reader, payloadStart, fieldEnd)))
    ) {
      // Replaced by its repacked copy.
    } else if (upcast && field === GRAPH_INITIALIZER && wire === WIRE_LENGTH && tensorType(reader, payloadStart, fieldEnd) === TENSOR_TYPE_FLOAT16) {
      out.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
      out.lengthDelimited(upcast.tensor(reader, payloadStart, fieldEnd, context));
    } else if (field === GRAPH_INITIALIZER && wire === WIRE_LENGTH) {
      const tensor = rewriteTensor(reader, payloadStart, fieldEnd, location, stats, context);
      out.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
      out.lengthDelimited(tensor);
    } else {
      out.bytes(reader.slice(fieldStart, fieldEnd));
    }
  });
  if (upcast) stats.extraFiles.push(...upcast.files());
  out.bytes(fused.extra);
  out.bytes(groupNorms.extra);
  out.bytes(attentions.extra);
  out.bytes(packed.extra);
  out.bytes(isInf.extra);
  if (linear) out.bytes(linear.extra);
  out.bytes(lookups.extra);
  if (weights) out.bytes(weights.extra);
  out.bytes(dequant.extra);
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
            for (let pos = p; pos < e;) {
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
      extra.bytes(
        externalInitializer(qName, [outFeatures, blocks, (QUANT_BLOCK * bits) / 8], TENSOR_TYPE_UINT8, path, 0, quantized.weightBytes),
      );
      extra.bytes(externalInitializer(scalesName, [outFeatures * blocks], data.type, path, quantized.weightBytes, quantized.scaleBytes));
      if (lookups.length) {
        // Same weights as [rows, blocks, 32] (that's MatMulNBits' layout too), scales as [rows, blocks, 1].
        const lookupScales = `${scalesName}/lookup`;
        const rowShape = `${weight}/row_shape`;
        extra.bytes(
          externalInitializer(lookupScales, [outFeatures, blocks, 1], data.type, path, quantized.weightBytes, quantized.scaleBytes),
        );
        extra.bytes(int64Initializer(rowShape, [0, 0, inFeatures]));
        lookups.forEach((gather, i) => {
          const blocked = `${weight}/lookup${i}_blocks`;
          plan.set(
            gather.start,
            concat(
              nodeField(
                'GatherBlockQuantized',
                gather.name || `${weight}/lookup${i}`,
                [qName, gather.inputs[1]!, lookupScales],
                [blocked],
                {
                  bits,
                  block_size: QUANT_BLOCK,
                  gather_axis: 0,
                  quantize_axis: 2,
                },
                MICROSOFT_DOMAIN,
              ),
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
          nodeField(
            'MatMulNBits',
            matmul.name || `${weight}/matmul_nbits${i}`,
            [inputOf(matmul, last), qName, scalesName],
            matmul.outputs,
            {
              K: inFeatures,
              N: outFeatures,
              bits,
              block_size: QUANT_BLOCK,
            },
            MICROSOFT_DOMAIN,
          ),
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
      plan.set(
        matmul.start,
        gemmNodes(
          { ...matmul, inputs: [inputOf(matmul, last), ...matmul.inputs.slice(1)] },
          weight,
          inShape,
          outShape,
          `${weight}/gemm${i}`,
        ),
      ),
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
  while ((head?.opType === 'Cast' || head?.opType === 'Identity') && head.inputs.length === 1) head = producer.get(head.inputs[0]!);
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

function quantize(
  data: TensorData,
  rows: number,
  cols: number,
  bits: 4 | 8,
  weightBytes: number,
  scaleBytes: number,
  /** Scales as float32 even for a float16 tensor (computeFloat32). */
  floatScales = false,
): Uint8Array<ArrayBuffer> {
  const half = data.type === TENSOR_TYPE_FLOAT16;
  const elementBytes = half ? 2 : 4;
  const out = new Uint8Array(weightBytes + scaleBytes);
  const scales16 = half && !floatScales ? new Uint16Array(out.buffer, weightBytes, scaleBytes / 2) : null;
  const scales32 = half && !floatScales ? null : new Float32Array(out.buffer, weightBytes, scaleBytes / 4);
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
      exponent === 0
        ? sign * 2 ** -14 * (fraction / 1024)
        : exponent === 31
          ? fraction
            ? NaN
            : sign * Infinity
          : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
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
          else if (w === WIRE_LENGTH)
            for (let pos = p2; pos < e2;) {
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

/** A GraphProto.node field: the node with its INT attribute `name` set to `value` (replacing any it had). */
function withIntAttribute(reader: BufferedReader, payloadStart: number, end: number, name: string, value: number): Uint8Array {
  const node = new Writer();
  forEachField(reader, payloadStart, end, (field, wire, fieldStart, p, e) => {
    if (field === NODE_ATTRIBUTE && wire === WIRE_LENGTH) {
      let attributeName = '';
      forEachField(reader, p, e, (f, w, _s, p2, e2) => {
        if (f === ATTRIBUTE_NAME && w === WIRE_LENGTH) attributeName = text(reader, p2, e2);
      });
      if (attributeName === name) return;
    }
    node.bytes(reader.slice(fieldStart, e));
  });
  const attribute = new Writer();
  attribute.tag(ATTRIBUTE_NAME, WIRE_LENGTH);
  attribute.lengthDelimited(utf8(name));
  attribute.tag(ATTRIBUTE_I, WIRE_VARINT);
  attribute.varint(value);
  attribute.tag(ATTRIBUTE_TYPE, WIRE_VARINT);
  attribute.varint(ATTRIBUTE_TYPE_INT);
  node.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
  node.lengthDelimited(attribute.finish());
  const field = new Writer();
  field.tag(GRAPH_NODE, WIRE_LENGTH);
  field.lengthDelimited(node.finish());
  return field.finish();
}

/**
 * GroupNorm nodes rewritten as standard ops (see the top of this file): the replacement node fields by the
 * original node's position, and the initializers they share.
 */
function planGroupNorms(
  reader: BufferedReader,
  start: number,
  end: number,
  /** The graph computes in float32 (computeFloat32): its float16 gammas become float32. */
  float32: boolean,
): { replace: Map<number, Uint8Array>; extra: Uint8Array } {
  const nodes: { node: NodeInfo; groups: number; epsilon: number; silu: boolean; channelsLast: boolean }[] = [];
  const types = new Map<string, number>();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_NODE) {
      const node = readNode(reader, fieldStart, payloadStart, fieldEnd);
      if (node.opType !== 'GroupNorm' || node.domain !== MICROSOFT_DOMAIN) return;
      const ints = new Map<string, number>();
      let epsilon = 1e-5;
      forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
        if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH) return;
        const name = attributeName(reader, p, e);
        forEachField(reader, p, e, (af, aw, _s2, p2) => {
          if (af === ATTRIBUTE_I && aw === WIRE_VARINT) ints.set(name, reader.varint(p2)[0]);
          if (af === ATTRIBUTE_F && aw === WIRE_32BIT && name === 'epsilon')
            epsilon = new DataView(reader.slice(p2, p2 + 4).buffer).getFloat32(0, true);
        });
      });
      nodes.push({
        node,
        groups: ints.get('groups') ?? 32,
        epsilon,
        silu: ints.get('activation') === 1,
        channelsLast: ints.get('channels_last') !== 0,
      });
    } else if (field === GRAPH_INITIALIZER) {
      let name = '';
      let type = 0;
      forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
        if (f === TENSOR_NAME && w === WIRE_LENGTH) name = text(reader, p, e);
        if (f === TENSOR_DATA_TYPE && w === WIRE_VARINT) type = reader.varint(p)[0];
      });
      types.set(name, type);
    }
  });
  const replace = new Map<number, Uint8Array>();
  const extra = new Writer();
  const added = new Set<string>();
  const constant = (name: string, bytes: () => Uint8Array) => {
    if (!added.has(name)) extra.bytes(bytes());
    added.add(name);
    return name;
  };
  for (const { node, groups, epsilon, silu, channelsLast } of nodes) {
    const [input, gamma, beta] = node.inputs as [string, string, string];
    const stored = types.get(gamma);
    if (stored !== TENSOR_TYPE_FLOAT && stored !== TENSOR_TYPE_FLOAT16)
      throw new Error(`GroupNorm ${node.name}: gamma isn't a float initializer`);
    const type = float32 ? TENSOR_TYPE_FLOAT : stored;
    const p = `${node.name || node.outputs[0]}/group_norm`;
    const out = new Writer();
    let x = input;
    if (channelsLast) {
      out.bytes(nodeField('Transpose', `${p}/to_nchw`, [x], [`${p}/nchw`], {}, '', {}, { perm: [0, 3, 1, 2] }));
      x = `${p}/nchw`;
    }
    const grouped = constant(`/group_norm/shape_${groups}`, () => int64Initializer(`/group_norm/shape_${groups}`, [0, groups, -1]));
    const ones = constant(`/group_norm/ones_${groups}`, () =>
      floatInitializer(`/group_norm/ones_${groups}`, new Array<number>(groups).fill(1)),
    );
    const zeros = constant(`/group_norm/zeros_${groups}`, () =>
      floatInitializer(`/group_norm/zeros_${groups}`, new Array<number>(groups).fill(0)),
    );
    const perChannel = constant('/group_norm/per_channel', () => int64Initializer('/group_norm/per_channel', [-1, 1, 1]));
    out.bytes(nodeField('Reshape', `${p}/group`, [x, grouped], [`${p}/grouped`]));
    let normIn = `${p}/grouped`;
    if (type === TENSOR_TYPE_FLOAT16) {
      out.bytes(nodeField('Cast', `${p}/to_float`, [normIn], [`${p}/grouped_f32`], { to: TENSOR_TYPE_FLOAT }));
      normIn = `${p}/grouped_f32`;
    }
    const normOut = type === TENSOR_TYPE_FLOAT16 ? `${p}/normalized_f32` : `${p}/normalized`;
    out.bytes(nodeField('InstanceNormalization', `${p}/normalize`, [normIn, ones, zeros], [normOut], {}, '', { epsilon }));
    if (type === TENSOR_TYPE_FLOAT16)
      out.bytes(nodeField('Cast', `${p}/to_half`, [normOut], [`${p}/normalized`], { to: TENSOR_TYPE_FLOAT16 }));
    out.bytes(nodeField('Shape', `${p}/shape`, [x], [`${p}/shape_out`]));
    out.bytes(nodeField('Reshape', `${p}/ungroup`, [`${p}/normalized`, `${p}/shape_out`], [`${p}/ungrouped`]));
    out.bytes(nodeField('Reshape', `${p}/gamma`, [gamma, perChannel], [`${p}/gamma_c11`]));
    out.bytes(nodeField('Reshape', `${p}/beta`, [beta, perChannel], [`${p}/beta_c11`]));
    out.bytes(nodeField('Mul', `${p}/scale`, [`${p}/ungrouped`, `${p}/gamma_c11`], [`${p}/scaled`]));
    const affine = silu || channelsLast ? `${p}/affine` : node.outputs[0]!;
    out.bytes(nodeField('Add', `${p}/shift`, [`${p}/scaled`, `${p}/beta_c11`], [affine]));
    let y = affine;
    if (silu) {
      const activated = channelsLast ? `${p}/activated` : node.outputs[0]!;
      out.bytes(nodeField('Sigmoid', `${p}/sigmoid`, [affine], [`${p}/sigmoid_out`]));
      out.bytes(nodeField('Mul', `${p}/silu`, [affine, `${p}/sigmoid_out`], [activated]));
      y = activated;
    }
    if (channelsLast) out.bytes(nodeField('Transpose', `${p}/to_nhwc`, [y], [node.outputs[0]!], {}, '', {}, { perm: [0, 2, 3, 1] }));
    replace.set(node.start, out.finish());
  }
  return { replace, extra: extra.finish() };
}

const TENSOR_TYPE_INT4 = 22;
/** Repacked tensors per extra file: each file stays far below the 2 GB ONNX Runtime can address. */
const REPACK_FILE_BYTES = 2 ** 30;

/** An initializer's type, shape and where its bytes are. */
interface Initializer {
  type: number;
  dims: number[];
  source: ByteSource;
  offset: number;
  length: number;
}

function readInitializer(reader: BufferedReader, start: number, end: number, context: RewriteContext): Initializer | null {
  let type = 0;
  const dims: number[] = [];
  let raw: { offset: number; length: number } | null = null;
  const external: Record<string, string> = {};
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === TENSOR_DATA_TYPE && w === WIRE_VARINT) type = reader.varint(p)[0];
    else if (f === TENSOR_DIMS && w === WIRE_VARINT) dims.push(reader.varint(p)[0]);
    else if (f === TENSOR_DIMS && w === WIRE_LENGTH)
      for (let pos = p; pos < e; ) {
        const [v, next] = reader.varint(pos);
        dims.push(v);
        pos = next;
      }
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
  const inline = raw as { offset: number; length: number } | null;
  if (inline) return { type, dims, source: context.source, ...inline };
  const file = external.location && context.externalData?.[external.location];
  if (!file || external.length === undefined) return null;
  return { type, dims, source: file, offset: Number(external.offset ?? 0), length: Number(external.length) };
}

function tensorName(reader: BufferedReader, start: number, end: number): string {
  let name = '';
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === TENSOR_NAME && w === WIRE_LENGTH) name = text(reader, p, e);
  });
  return name;
}

/**
 * DequantizeLinear(int4) → MatMul pairs rewritten as MatMulNBits (see the top of this file): node edits by position
 * (null drops the DequantizeLinear), the initializers dropped, the repacked ones added, and the files they read.
 */
function planDequantMatMuls(
  reader: BufferedReader,
  start: number,
  end: number,
  context: RewriteContext,
  last: LastTokenPlan | null,
): { replace: Map<number, Uint8Array | null>; dropped: Set<string>; extra: Uint8Array; files: { path: string; source: ByteSource }[] } {
  const none = { replace: new Map<number, Uint8Array | null>(), dropped: new Set<string>(), extra: new Uint8Array(0), files: [] };
  const dequants = new Map<string, { node: NodeInfo; axis: number; block: number }>();
  const consumers = new Map<string, NodeInfo[]>();
  const initializers = new Map<string, { start: number; end: number }>();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_NODE) {
      const node = readNode(reader, fieldStart, payloadStart, fieldEnd);
      for (const input of node.inputs) consumers.set(input, [...(consumers.get(input) ?? []), node]);
      if (node.opType !== 'DequantizeLinear' || node.domain) return;
      const ints = new Map<string, number>();
      forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
        if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH) return;
        const name = attributeName(reader, p, e);
        forEachField(reader, p, e, (af, aw, _s2, p2) => {
          if (af === ATTRIBUTE_I && aw === WIRE_VARINT) ints.set(name, reader.varint(p2)[0]);
        });
      });
      dequants.set(node.outputs[0]!, { node, axis: ints.get('axis') ?? 1, block: ints.get('block_size') ?? 0 });
    } else if (field === GRAPH_INITIALIZER) {
      initializers.set(tensorName(reader, payloadStart, fieldEnd), { start: payloadStart, end: fieldEnd });
    }
  });
  if (dequants.size === 0) return none;

  const replace = new Map<number, Uint8Array | null>();
  const dropped = new Set<string>();
  const extra = new Writer();
  const files: { path: string; source: ByteSource }[] = [];
  // Repacked tensors, laid out in files of up to REPACK_FILE_BYTES.
  let pending: { offset: number; length: number; build: () => Uint8Array }[] = [];
  let fileBytes = 0;
  const flush = () => {
    if (pending.length === 0) return;
    files.push({ path: `repacked-${files.length}.bin`, source: repackedFile(pending, fileBytes) });
    pending = [];
    fileBytes = 0;
  };
  const place = (length: number, build: () => Uint8Array) => {
    if (fileBytes + length > REPACK_FILE_BYTES) flush();
    const entry = { offset: fileBytes, length, build };
    pending.push(entry);
    fileBytes += length;
    return { path: `repacked-${files.length}.bin`, offset: entry.offset };
  };
  for (const [output, { node, axis, block }] of dequants) {
    const users = consumers.get(output) ?? [];
    const [weightName, scaleName, zeroName] = node.inputs as [string, string, string | undefined];
    if (axis !== 0 || ![16, 32, 64, 128, 256].includes(block) || zeroName) continue;
    if (users.length === 0 || users.some((u) => u.opType !== 'MatMul' || u.inputs[1] !== output || u.start === last?.head.start)) continue;
    const w = initializers.get(weightName);
    const sc = initializers.get(scaleName);
    if (!w || !sc) continue;
    const weight = readInitializer(reader, w.start, w.end, context);
    const scale = readInitializer(reader, sc.start, sc.end, context);
    if (!weight || !scale || weight.type !== TENSOR_TYPE_INT4 || weight.dims.length !== 2) continue;
    if (scale.type !== TENSOR_TYPE_FLOAT16 && scale.type !== TENSOR_TYPE_FLOAT) continue;
    const [k, n] = weight.dims as [number, number];
    if (k % block !== 0 || n % 2 !== 0 || scale.dims.join() !== `${k / block},${n}`) continue;
    const blocks = k / block;
    const scaleBytes = scale.type === TENSOR_TYPE_FLOAT16 ? 2 : 4;
    const packed = place((n * k) / 2, () => repackInt4(weight, k, n));
    const packedName = `${weightName}/nbits`;
    extra.bytes(externalInitializer(packedName, [n, blocks, block / 2], TENSOR_TYPE_UINT8, packed.path, packed.offset, (n * k) / 2));
    const scales = place(n * blocks * scaleBytes, () => transposeScales(scale, blocks, n, scaleBytes));
    const scalesName = `${scaleName}/nbits`;
    extra.bytes(externalInitializer(scalesName, [n, blocks], scale.type, scales.path, scales.offset, n * blocks * scaleBytes));
    replace.set(node.start, null);
    for (const user of users)
      replace.set(
        user.start,
        nodeField('MatMulNBits', user.name || `${output}/MatMulNBits`, [user.inputs[0]!, packedName, scalesName], user.outputs, { K: k, N: n, bits: 4, block_size: block }, MICROSOFT_DOMAIN),
      );
    dropped.add(weightName);
    dropped.add(scaleName);
  }
  flush();
  return { replace, dropped, extra: extra.finish(), files };
}

/** A virtual file of repacked tensors, each computed when it is read (ONNX Runtime reads each tensor once). */
function repackedFile(entries: { offset: number; length: number; build: () => Uint8Array }[], size: number): ByteSource {
  return {
    size,
    read(offset, length) {
      const entry = entries.find((e) => offset >= e.offset && offset < e.offset + e.length);
      if (!entry) return new Uint8Array(length);
      const bytes = entry.build();
      const from = offset - entry.offset;
      const part = new Uint8Array(length);
      part.set(bytes.subarray(from, Math.min(bytes.length, from + length)));
      return part;
    },
  };
}

/**
 * int4 [K, N] (row-major, two values per byte, low nibble first, signed) → MatMulNBits' [N, K / block, block / 2]
 * (column n's K values in order, two per byte, low nibble first, unsigned with zero point 8: the signed value's
 * nibble with its top bit flipped).
 */
function repackInt4(weight: Initializer, k: number, n: number): Uint8Array {
  const src = weight.source.read(weight.offset, weight.length);
  const out = new Uint8Array((n * k) / 2);
  const rowBytes = n / 2;
  for (let row = 0; row < k; row += 2) {
    const a = row * rowBytes;
    const b = a + rowBytes;
    const at = row / 2;
    for (let pair = 0; pair < rowBytes; pair++) {
      const top = src[a + pair]!;
      const bottom = src[b + pair]!;
      const col = pair * 2;
      // Column col: (row, row + 1) low nibbles; column col + 1: their high nibbles.
      out[col * (k / 2) + at] = ((top & 0x0f) ^ 0x08) | (((bottom & 0x0f) ^ 0x08) << 4);
      out[(col + 1) * (k / 2) + at] = ((top >> 4) ^ 0x08) | (((bottom >> 4) ^ 0x08) << 4);
    }
  }
  return out;
}

/** Scales [K / block, N] → [N, K / block]. */
function transposeScales(scale: Initializer, blocks: number, n: number, bytes: number): Uint8Array {
  const src = scale.source.read(scale.offset, scale.length);
  const out = new Uint8Array(blocks * n * bytes);
  for (let b = 0; b < blocks; b++)
    for (let c = 0; c < n; c++) for (let i = 0; i < bytes; i++) out[(c * blocks + b) * bytes + i] = src[(b * n + c) * bytes + i]!;
  return out;
}

/**
 * ONNX Attention nodes (opset 23+) with 3-D inputs, a mask and a past KV cache, as standard ops (see the top of
 * this file): the replacement node fields by the original node's position, and the shape constants they share.
 */
function planAttentions(reader: BufferedReader, start: number, end: number): { replace: Map<number, Uint8Array>; extra: Uint8Array } {
  const replace = new Map<number, Uint8Array>();
  const extra = new Writer();
  const added = new Set<string>();
  const shape = (name: string, values: number[]) => {
    if (!added.has(name)) extra.bytes(int64Initializer(name, values));
    added.add(name);
    return name;
  };
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field !== GRAPH_NODE || wire !== WIRE_LENGTH) return;
    const node = readNode(reader, fieldStart, payloadStart, fieldEnd);
    if (node.opType !== 'Attention' || node.domain) return;
    const ints = new Map<string, number>();
    const floats = new Map<string, number>();
    forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
      if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH) return;
      const name = attributeName(reader, p, e);
      forEachField(reader, p, e, (af, aw, _s2, p2) => {
        if (af === ATTRIBUTE_I && aw === WIRE_VARINT) ints.set(name, reader.varint(p2)[0]);
        if (af === ATTRIBUTE_F && aw === WIRE_32BIT) floats.set(name, new DataView(reader.slice(p2, p2 + 4).buffer).getFloat32(0, true));
      });
    });
    const [q, k, v, mask, pastK, pastV] = node.inputs;
    const [y, presentK, presentV] = node.outputs;
    const heads = ints.get('q_num_heads');
    const kvHeads = ints.get('kv_num_heads');
    // Only the form described above: 3-D inputs (head counts given), an explicit mask, a past cache, its outputs.
    if (!heads || !kvHeads || heads % kvHeads !== 0 || ints.get('is_causal') || !mask || !pastK || !pastV || !presentK || !presentV) return;
    if (node.inputs.length > 6 || node.outputs.length > 3 || (ints.get('qk_matmul_output_mode') ?? 0) !== 0) return;
    const p = `${node.name || y}/decomposed`;
    const out = new Writer();
    const n = (op: string, suffix: string, inputs: string[], outputs: string[], i: Record<string, number> = {}, f: Record<string, number> = {}, l: Record<string, number[]> = {}) =>
      out.bytes(nodeField(op, `${p}/${suffix}`, inputs, outputs, i, '', f, l));
    // [B, S, N × H] → [B, N, S, H].
    const split = (x: string, count: number, name: string) => {
      n('Reshape', `${name}_4d`, [x, shape(`/attention/heads_${count}`, [0, 0, count, -1])], [`${p}/${name}_bsnh`]);
      n('Transpose', `${name}_bnsh`, [`${p}/${name}_bsnh`], [`${p}/${name}`], {}, {}, { perm: [0, 2, 1, 3] });
      return `${p}/${name}`;
    };
    const query = split(q!, heads, 'q');
    n('Concat', 'present_k', [pastK, split(k!, kvHeads, 'k')], [presentK], { axis: 2 });
    n('Concat', 'present_v', [pastV, split(v!, kvHeads, 'v')], [presentV], { axis: 2 });
    // Queries by key head: [B, kv, group × S, H] × [B, kv, H, T] → [B, kv, group × S, T] → [B, N, S, T].
    n('Reshape', 'q_grouped', [query, shape(`/attention/heads_${kvHeads}_keep_last`, [0, kvHeads, -1, 0])], [`${p}/q_grouped`]);
    n('Transpose', 'k_t', [presentK], [`${p}/k_t`], {}, {}, { perm: [0, 1, 3, 2] });
    n('MatMul', 'qk', [`${p}/q_grouped`, `${p}/k_t`], [`${p}/qk_grouped`]);
    n('Reshape', 'scores', [`${p}/qk_grouped`, shape(`/attention/heads_${heads}_keep_last`, [0, heads, -1, 0])], [`${p}/scores_raw`]);
    let scores = `${p}/scores_raw`;
    const scale = floats.get('scale');
    if (scale === undefined) {
      // 1 / √H, from the query's head size.
      n('Shape', 'head_size', [query], [`${p}/head_size_i64`], { start: 3, end: 4 });
      n('Cast', 'head_size_f', [`${p}/head_size_i64`], [`${p}/head_size_f`], { to: TENSOR_TYPE_FLOAT });
      n('Sqrt', 'sqrt_h', [`${p}/head_size_f`], [`${p}/sqrt_h`]);
      n('Reciprocal', 'inv_sqrt_h', [`${p}/sqrt_h`], [`${p}/scale_f`]);
      n('CastLike', 'scale', [`${p}/scale_f`, scores], [`${p}/scale`]);
      n('Mul', 'scaled', [scores, `${p}/scale`], [`${p}/scaled`]);
      scores = `${p}/scaled`;
    } else if (scale !== 1) {
      extra.bytes(floatInitializer(`${p}/scale_f`, [scale]));
      n('CastLike', 'scale', [`${p}/scale_f`, scores], [`${p}/scale`]);
      n('Mul', 'scaled', [scores, `${p}/scale`], [`${p}/scaled`]);
      scores = `${p}/scaled`;
    }
    const softcap = floats.get('softcap') ?? 0;
    if (softcap > 0) {
      extra.bytes(floatInitializer(`${p}/softcap_f`, [softcap]));
      n('CastLike', 'softcap', [`${p}/softcap_f`, scores], [`${p}/softcap`]);
      n('Div', 'capped_in', [scores, `${p}/softcap`], [`${p}/capped_in`]);
      n('Tanh', 'capped_tanh', [`${p}/capped_in`], [`${p}/capped_tanh`]);
      n('Mul', 'capped', [`${p}/capped_tanh`, `${p}/softcap`], [`${p}/capped`]);
      scores = `${p}/capped`;
    }
    n('Add', 'masked', [scores, mask], [`${p}/masked`]);
    n('Softmax', 'probs', [`${p}/masked`], [`${p}/probs`], { axis: -1 });
    // [B, N, S, T] → [B, kv, group × S, T] × [B, kv, T, H] → [B, N, S, H] → [B, S, N × H].
    n('Reshape', 'probs_grouped', [`${p}/probs`, shape(`/attention/heads_${kvHeads}_keep_last`, [0, kvHeads, -1, 0])], [`${p}/probs_grouped`]);
    n('MatMul', 'pv_matmul', [`${p}/probs_grouped`, presentV], [`${p}/pv_grouped`]);
    n('Reshape', 'pv', [`${p}/pv_grouped`, shape(`/attention/heads_${heads}_keep_last`, [0, heads, -1, 0])], [`${p}/pv_bnsh`]);
    n('Transpose', 'pv_bsnh', [`${p}/pv_bnsh`], [`${p}/pv_bsnh`], {}, {}, { perm: [0, 2, 1, 3] });
    n('Reshape', 'output', [`${p}/pv_bsnh`, shape('/attention/merge_heads', [0, 0, -1])], [y!]);
    replace.set(fieldStart, out.finish());
  });
  return { replace, extra: extra.finish() };
}

/** MatMuls by float weight matrices as MatMulNBits over quantized copies (`quantizeWeights`, see the top of this file). */
function planQuantizedMatMuls(
  reader: BufferedReader,
  start: number,
  end: number,
  context: RewriteContext,
  bits: 4 | 8,
  last: LastTokenPlan | null,
  taken: Map<number, Uint8Array | null>,
): { replace: Map<number, Uint8Array>; dropped: Set<string>; extra: Uint8Array; files: { path: string; source: ByteSource }[] } {
  const weights = new Map<string, Initializer>();
  const consumers = new Map<string, { node: NodeInfo; index: number }[]>();
  /** Gemm nodes that are plain products (transA 0, alpha and beta 1), by position: whether B is transposed. */
  const gemms = new Map<number, boolean>();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_INITIALIZER) {
      const tensor = readInitializer(reader, payloadStart, fieldEnd, context);
      if (tensor && tensor.dims.length === 2 && (tensor.type === TENSOR_TYPE_FLOAT16 || tensor.type === TENSOR_TYPE_FLOAT))
        weights.set(tensorName(reader, payloadStart, fieldEnd), tensor);
    } else if (field === GRAPH_NODE) {
      const node = readNode(reader, fieldStart, payloadStart, fieldEnd);
      node.inputs.forEach((input, index) => consumers.set(input, [...(consumers.get(input) ?? []), { node, index }]));
      if (node.opType === 'Gemm' && !node.domain) {
        const ints = new Map<string, number>();
        const floats = new Map<string, number>();
        forEachField(reader, payloadStart, fieldEnd, (f, w, _s, p, e) => {
          if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH) return;
          const name = attributeName(reader, p, e);
          forEachField(reader, p, e, (af, aw, _s2, p2) => {
            if (af === ATTRIBUTE_I && aw === WIRE_VARINT) ints.set(name, reader.varint(p2)[0]);
            if (af === ATTRIBUTE_F && aw === WIRE_32BIT) floats.set(name, new DataView(reader.slice(p2, p2 + 4).buffer).getFloat32(0, true));
          });
        });
        if (!ints.get('transA') && (floats.get('alpha') ?? 1) === 1 && (floats.get('beta') ?? 1) === 1) gemms.set(node.start, ints.get('transB') === 1);
      }
    }
  });
  const replace = new Map<number, Uint8Array>();
  const dropped = new Set<string>();
  const extra = new Writer();
  const files: { path: string; source: ByteSource }[] = [];
  for (const [name, weight] of weights) {
    const uses = consumers.get(name) ?? [];
    // [K, N] for MatMul and Gemm; [N, K] for Gemm with transB. Every use must read it the same way.
    const transposed = (u: { node: NodeInfo }) => u.node.opType === 'Gemm' && gemms.get(u.node.start) === true;
    const plain = (u: { node: NodeInfo; index: number }) =>
      u.index === 1 &&
      !taken.has(u.node.start) &&
      ((u.node.opType === 'MatMul' && !u.node.domain && u.node.inputs.length === 2) || (u.node.opType === 'Gemm' && gemms.has(u.node.start)));
    if (uses.length === 0 || !uses.every(plain) || new Set(uses.map(transposed)).size !== 1) continue;
    const rowsFirst = transposed(uses[0]!);
    const [k, n] = (rowsFirst ? [...weight.dims].reverse() : weight.dims) as [number, number];
    if (k % QUANT_BLOCK !== 0) continue;
    const blocks = k / QUANT_BLOCK;
    const weightBytes = (n * k * bits) / 8;
    const floatScales = !!context.computeFloat32;
    const scaleType = floatScales ? TENSOR_TYPE_FLOAT : weight.type;
    const scaleBytes = n * blocks * (scaleType === TENSOR_TYPE_FLOAT16 ? 2 : 4);
    const path = `${name.replace(/[^\w.-]/g, '_')}.q${bits}`;
    let built: Uint8Array | null = null;
    let reads = 0;
    files.push({
      path,
      source: {
        size: weightBytes + scaleBytes,
        read(offset, length) {
          built ??= rowsFirst
            ? quantize(
                { type: weight.type as TensorData['type'], source: weight.source, offset: weight.offset, length: weight.length },
                n,
                k,
                bits,
                weightBytes,
                scaleBytes,
                floatScales,
              )
            : quantizeColumns(weight, k, n, bits, floatScales);
          const part = built.slice(offset, offset + length);
          if (++reads % 2 === 0) built = null;
          return part;
        },
      },
    });
    const qName = `${name}/q${bits}`;
    const scales = `${name}/q${bits}_scales`;
    extra.bytes(externalInitializer(qName, [n, blocks, (QUANT_BLOCK * bits) / 8], TENSOR_TYPE_UINT8, path, 0, weightBytes));
    extra.bytes(externalInitializer(scales, [n * blocks], scaleType, path, weightBytes, scaleBytes));
    uses.forEach(({ node }, i) => {
      const bias = node.opType === 'Gemm' ? node.inputs[2] : undefined;
      const product = bias ? `${name}/product${i}` : node.outputs[0]!;
      const matmul = nodeField(
        'MatMulNBits',
        node.name || `${name}/matmul_nbits${i}`,
        [inputOf(node, last), qName, scales],
        [product],
        { K: k, N: n, bits, block_size: QUANT_BLOCK },
        MICROSOFT_DOMAIN,
      );
      replace.set(node.start, bias ? concat(matmul, nodeField('Add', `${name}/bias${i}`, [product, bias], node.outputs)) : matmul);
    });
    dropped.add(name);
  }
  return { replace, dropped, extra: extra.finish(), files };
}

/**
 * W [K, N] (row-major, float16 or float32) quantized per column for MatMulNBits: [N, K / 32, 32 × bits / 8] bytes
 * (4-bit values two per byte, low nibble first; unsigned around 2^(bits−1)), then [N × K / 32] scales in W's type,
 * each the block's largest |w| / (2^(bits−1) − 1). Reads 32 rows at a time.
 */
function quantizeColumns(weight: Initializer, k: number, n: number, bits: 4 | 8, floatScales = false): Uint8Array {
  const half = weight.type === TENSOR_TYPE_FLOAT16;
  const elementBytes = half ? 2 : 4;
  const blocks = k / QUANT_BLOCK;
  const weightBytes = (n * k * bits) / 8;
  const scaleBytes = half && !floatScales ? 2 : 4;
  const out = new Uint8Array(weightBytes + n * blocks * scaleBytes);
  const scales16 = half && !floatScales ? new Uint16Array(out.buffer, weightBytes, n * blocks) : null;
  const scales32 = half && !floatScales ? null : new Float32Array(out.buffer, weightBytes, n * blocks);
  const zero = 1 << (bits - 1);
  const maxQ = zero - 1;
  const top = (1 << bits) - 1;
  const blockBytes = (QUANT_BLOCK * bits) / 8;
  const rows = new Float32Array(QUANT_BLOCK * n);
  for (let b = 0; b < blocks; b++) {
    const raw = weight.source.read(weight.offset + b * QUANT_BLOCK * n * elementBytes, QUANT_BLOCK * n * elementBytes);
    const bytes = raw.byteOffset % elementBytes === 0 ? raw : raw.slice();
    if (half) {
      const v = new Uint16Array(bytes.buffer, bytes.byteOffset, QUANT_BLOCK * n);
      for (let i = 0; i < v.length; i++) rows[i] = HALF_TO_FLOAT[v[i]!]!;
    } else rows.set(new Float32Array(bytes.buffer, bytes.byteOffset, QUANT_BLOCK * n));
    for (let c = 0; c < n; c++) {
      let maxAbs = 0;
      for (let r = 0; r < QUANT_BLOCK; r++) maxAbs = Math.max(maxAbs, Math.abs(rows[r * n + c]!));
      const index = c * blocks + b;
      let scale: number;
      if (scales16) {
        scales16[index] = floatToHalf(maxAbs / maxQ);
        scale = HALF_TO_FLOAT[scales16[index]!]!;
      } else {
        scale = Math.fround(maxAbs / maxQ);
        scales32![index] = scale;
      }
      const inverse = scale > 0 ? 1 / scale : 0;
      const at = index * blockBytes;
      for (let r = 0; r < QUANT_BLOCK; r++) {
        const q = Math.min(top, Math.max(0, Math.round(rows[r * n + c]! * inverse) + zero));
        if (bits === 8) out[at + r] = q;
        else if (r % 2 === 0) out[at + r / 2] = q;
        else out[at + (r - 1) / 2]! |= q << 4;
      }
    }
  }
  return out;
}

function tensorType(reader: BufferedReader, start: number, end: number): number {
  let type = 0;
  forEachField(reader, start, end, (f, w, _s, p) => {
    if (f === TENSOR_DATA_TYPE && w === WIRE_VARINT) type = reader.varint(p)[0];
  });
  return type;
}

/** float16 bits → float32 values: the browser's own conversion where it has Float16Array, else a lookup table. */
export function halfToFloat32(bits: Uint16Array): Float32Array<ArrayBuffer> {
  const Half = (globalThis as { Float16Array?: new (buffer: ArrayBufferLike, offset: number, length: number) => ArrayLike<number> }).Float16Array;
  if (Half) return new Float32Array(new Half(bits.buffer, bits.byteOffset, bits.length));
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) out[i] = HALF_TO_FLOAT[bits[i]!]!;
  return out;
}

/**
 * A float16 TensorProto's values: raw bytes (inline or external), or int32_data (one half per varint), as half bits.
 * Returns null for an external tensor whose file isn't given.
 */
function halfBits(reader: BufferedReader, start: number, end: number, context: RewriteContext): (() => Uint16Array) | null {
  let raw: { offset: number; length: number } | null = null;
  const ints: number[] = [];
  const external: Record<string, string> = {};
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === TENSOR_RAW_DATA && w === WIRE_LENGTH) raw = { offset: p, length: e - p };
    else if (f === TENSOR_INT32_DATA && w === WIRE_VARINT) ints.push(reader.varint(p)[0]);
    else if (f === TENSOR_INT32_DATA && w === WIRE_LENGTH)
      for (let pos = p; pos < e; ) {
        const [v, next] = reader.varint(pos);
        ints.push(v);
        pos = next;
      }
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
  const inline = raw as { offset: number; length: number } | null;
  const asHalf = (bytes: Uint8Array) => new Uint16Array(bytes.slice().buffer);
  if (inline) return () => asHalf(context.source.read(inline.offset, inline.length));
  if (external.location) {
    const file = context.externalData?.[external.location];
    if (!file) return null;
    const offset = Number(external.offset ?? 0);
    const length = Number(external.length ?? file.size - offset);
    return () => asHalf(file.read(offset, length));
  }
  return () => Uint16Array.from(ints);
}

/** float32 copies of float16 initializers too big to keep inline, laid out in extra files (computeFloat32). */
class Float32Files {
  private readonly done: { path: string; source: ByteSource }[] = [];
  private pending: { offset: number; length: number; build: () => Uint8Array }[] = [];
  private bytes = 0;

  /** The tensor (TensorProto bytes) as float32: inline when small, else pointing into an extra file. */
  tensor(reader: BufferedReader, start: number, end: number, context: RewriteContext): Uint8Array {
    const values = halfBits(reader, start, end, context);
    if (!values) throw new Error('computeFloat32: a float16 tensor’s data file is missing');
    const out = new Writer();
    let count = 1;
    forEachField(reader, start, end, (f, w, fieldStart, p, e) => {
      if (f === TENSOR_DIMS) {
        if (w === WIRE_VARINT) count *= reader.varint(p)[0];
        else for (let pos = p; pos < e; ) {
          const [v, next] = reader.varint(pos);
          count *= v;
          pos = next;
        }
        out.bytes(reader.slice(fieldStart, e));
      } else if (f === TENSOR_DATA_TYPE) {
        out.tag(TENSOR_DATA_TYPE, WIRE_VARINT);
        out.varint(TENSOR_TYPE_FLOAT);
      } else if (f !== TENSOR_RAW_DATA && f !== TENSOR_INT32_DATA && f !== TENSOR_EXTERNAL_DATA && f !== TENSOR_DATA_LOCATION) {
        out.bytes(reader.slice(fieldStart, e));
      }
    });
    const length = count * 4;
    if (length < MIN_EXTERNAL_BYTES) {
      out.tag(TENSOR_RAW_DATA, WIRE_LENGTH);
      out.lengthDelimited(new Uint8Array(halfToFloat32(values()).buffer));
      return out.finish();
    }
    if (this.bytes + length > REPACK_FILE_BYTES) this.flush();
    const offset = this.bytes;
    this.pending.push({ offset, length, build: () => new Uint8Array(halfToFloat32(values()).buffer) });
    this.bytes += length;
    for (const [key, value] of [
      ['location', `float32-${this.done.length}.bin`],
      ['offset', String(offset)],
      ['length', String(length)],
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
    return out.finish();
  }

  files(): { path: string; source: ByteSource }[] {
    this.flush();
    return this.done;
  }

  private flush(): void {
    if (this.pending.length === 0) return;
    this.done.push({ path: `float32-${this.done.length}.bin`, source: repackedFile(this.pending, this.bytes) });
    this.pending = [];
    this.bytes = 0;
  }
}

/** A GraphProto.node field with Cast-to-float16 made Cast-to-float32 and float16 tensor attributes made float32. */
function upcastNode(reader: BufferedReader, payloadStart: number, end: number): Uint8Array {
  const node = new Writer();
  let isCast = false;
  forEachField(reader, payloadStart, end, (field, wire, _fs, p, e) => {
    if (field === NODE_OP_TYPE && wire === WIRE_LENGTH && text(reader, p, e) === 'Cast') isCast = true;
  });
  forEachField(reader, payloadStart, end, (field, wire, fieldStart, p, e) => {
    if (field !== NODE_ATTRIBUTE || wire !== WIRE_LENGTH) {
      node.bytes(reader.slice(fieldStart, e));
      return;
    }
    const name = attributeName(reader, p, e);
    const attribute = new Writer();
    forEachField(reader, p, e, (af, aw, afs, ap, ae) => {
      if (isCast && name === 'to' && af === ATTRIBUTE_I && aw === WIRE_VARINT && reader.varint(ap)[0] === TENSOR_TYPE_FLOAT16) {
        attribute.tag(ATTRIBUTE_I, WIRE_VARINT);
        attribute.varint(TENSOR_TYPE_FLOAT);
      } else if (af === ATTRIBUTE_T && aw === WIRE_LENGTH && tensorType(reader, ap, ae) === TENSOR_TYPE_FLOAT16) {
        attribute.tag(ATTRIBUTE_T, WIRE_LENGTH);
        attribute.lengthDelimited(inlineFloat32(reader, ap, ae));
      } else {
        attribute.bytes(reader.slice(afs, ae));
      }
    });
    node.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
    node.lengthDelimited(attribute.finish());
  });
  const out = new Writer();
  out.tag(GRAPH_NODE, WIRE_LENGTH);
  out.lengthDelimited(node.finish());
  return out.finish();
}

/** A small inline float16 TensorProto (a Constant's value) as float32. */
function inlineFloat32(reader: BufferedReader, start: number, end: number): Uint8Array {
  const out = new Writer();
  forEachField(reader, start, end, (f, _w, fieldStart, _p, e) => {
    if (f === TENSOR_DATA_TYPE) {
      out.tag(TENSOR_DATA_TYPE, WIRE_VARINT);
      out.varint(TENSOR_TYPE_FLOAT);
    } else if (f !== TENSOR_RAW_DATA && f !== TENSOR_INT32_DATA) out.bytes(reader.slice(fieldStart, e));
  });
  out.tag(TENSOR_RAW_DATA, WIRE_LENGTH);
  out.lengthDelimited(new Uint8Array(halfToFloat32(rawOrInts(reader, start, end) ?? new Uint16Array(0)).buffer));
  return out.finish();
}

/** An inline TensorProto's float16 bits (raw_data or int32_data), read from the graph itself. */
function rawOrInts(reader: BufferedReader, start: number, end: number): Uint16Array | null {
  let raw: Uint8Array | null = null;
  const ints: number[] = [];
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === TENSOR_RAW_DATA && w === WIRE_LENGTH) raw = reader.slice(p, e);
    else if (f === TENSOR_INT32_DATA && w === WIRE_VARINT) ints.push(reader.varint(p)[0]);
    else if (f === TENSOR_INT32_DATA && w === WIRE_LENGTH)
      for (let pos = p; pos < e; ) {
        const [v, next] = reader.varint(pos);
        ints.push(v);
        pos = next;
      }
  });
  const bytes = raw as Uint8Array | null;
  return bytes ? new Uint16Array(bytes.slice().buffer) : ints.length ? Uint16Array.from(ints) : null;
}

/** A graph input or output (ValueInfoProto) with float16 changed to float32. */
function upcastValueInfo(reader: BufferedReader, start: number, end: number): Uint8Array {
  const rewrite = (s: number, e: number, path: number[]): Uint8Array => {
    const out = new Writer();
    forEachField(reader, s, e, (f, w, fieldStart, p, fe) => {
      const here = [...path, f];
      // ValueInfoProto.type (2) → TypeProto.tensor_type (1) → elem_type (1).
      if (here.join() === '2,1,1' && w === WIRE_VARINT && reader.varint(p)[0] === TENSOR_TYPE_FLOAT16) {
        out.tag(1, WIRE_VARINT);
        out.varint(TENSOR_TYPE_FLOAT);
      } else if ((here.join() === '2' || here.join() === '2,1') && w === WIRE_LENGTH) {
        out.tag(f, WIRE_LENGTH);
        out.lengthDelimited(rewrite(p, fe, here));
      } else out.bytes(reader.slice(fieldStart, fe));
    });
    return out.finish();
  };
  return rewrite(start, end, []);
}

/** Largest tensor ONNX Runtime Web can hold (see the top of this file). */
const TENSOR_LIMIT = 2 ** 31 - 1;

/** Gathers from float tables over TENSOR_LIMIT, as 8-bit GatherBlockQuantized lookups (see the top of this file). */
function planLargeLookups(
  reader: BufferedReader,
  start: number,
  end: number,
  context: RewriteContext,
): { replace: Map<number, Uint8Array>; dropped: Set<string>; extra: Uint8Array; files: { path: string; source: ByteSource }[] } {
  const tables = new Map<string, Initializer>();
  const gathers: NodeInfo[] = [];
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_INITIALIZER) {
      const tensor = readInitializer(reader, payloadStart, fieldEnd, context);
      if (tensor && tensor.length > TENSOR_LIMIT && tensor.dims.length === 2 && (tensor.type === TENSOR_TYPE_FLOAT16 || tensor.type === TENSOR_TYPE_FLOAT))
        tables.set(tensorName(reader, payloadStart, fieldEnd), tensor);
    } else if (field === GRAPH_NODE) {
      const node = readNode(reader, fieldStart, payloadStart, fieldEnd);
      if (node.opType === 'Gather' && !node.domain && (node.axis ?? 0) === 0) gathers.push(node);
    }
  });
  const replace = new Map<number, Uint8Array>();
  const dropped = new Set<string>();
  const extra = new Writer();
  const files: { path: string; source: ByteSource }[] = [];
  const bits = 8;
  for (const [name, table] of tables) {
    const users = gathers.filter((g) => g.inputs[0] === name);
    const [rows, cols] = table.dims as [number, number];
    if (users.length === 0 || cols % QUANT_BLOCK !== 0) continue;
    const data: TensorData = { type: table.type as TensorData['type'], source: table.source, offset: table.offset, length: table.length };
    const quantized = quantizedSource(data, rows, cols, bits, 2);
    const path = `${name.replace(/[^\w.-]/g, '_')}.q${bits}`;
    files.push({ path, source: quantized.source });
    const blocks = cols / QUANT_BLOCK;
    const qName = `${name}/q${bits}`;
    const scales = `${name}/q${bits}_scales`;
    const rowShape = `${name}/row_shape`;
    // [rows, blocks, 32] values and [rows, blocks, 1] scales, as the output layer's lookups above.
    extra.bytes(externalInitializer(qName, [rows, blocks, QUANT_BLOCK], TENSOR_TYPE_UINT8, path, 0, quantized.weightBytes));
    extra.bytes(externalInitializer(scales, [rows, blocks, 1], table.type, path, quantized.weightBytes, quantized.scaleBytes));
    extra.bytes(int64Initializer(rowShape, [0, 0, cols]));
    users.forEach((gather, i) => {
      const blocked = `${name}/lookup${i}_blocks`;
      replace.set(
        gather.start,
        concat(
          nodeField(
            'GatherBlockQuantized',
            gather.name || `${name}/lookup${i}`,
            [qName, gather.inputs[1]!, scales],
            [blocked],
            { bits, block_size: QUANT_BLOCK, gather_axis: 0, quantize_axis: 2 },
            MICROSOFT_DOMAIN,
          ),
          nodeField('Reshape', `${name}/lookup${i}_reshape`, [blocked, rowShape], gather.outputs),
        ),
      );
    });
    dropped.add(name);
  }
  return { replace, dropped, extra: extra.finish(), files };
}

/** IsInf nodes as comparisons with infinity (see the top of this file). */
/** What planLinearAttention changes: node edits by position (null drops one), new initializers, and redeclared graph inputs/outputs. */
interface LinearAttentionPlan {
  replace: Map<number, Uint8Array | null>;
  extra: Uint8Array;
  /** Graph input or output fields (tag included) by value name. */
  valueInfos: Map<string, Uint8Array>;
}

/**
 * Qwen3.5's linear-attention layers in the transformers.js exports (onnx-community/Qwen3.5-*-ONNX) as ONNX Runtime's
 * fused ops (see the top of this file): each layer's `If` on "one new token?" (a decode branch of ~76 small ops,
 * a prompt branch looping over tokens with Scan) becomes CausalConvWithState + LinearAttention, as ONNX Runtime
 * GenAI builds the same model. Both branches are read for nothing but names and shapes: the decode branch says which
 * outer values are the projections, states and weights, and how q, k and v split into heads. The conv state is
 * redeclared as the fused op keeps it, the last kernel − 1 positions, sequence-major: [batch, kernel − 1, channels]
 * instead of the export's [batch, channels, kernel]. Layers that don't match are left as they are.
 */
function planLinearAttention(reader: BufferedReader, start: number, end: number): LinearAttentionPlan | null {
  const nodes: { node: NodeInfo; payloadStart: number }[] = [];
  const inits = new Map<string, { payloadStart: number; end: number }>();
  const values = new Map<string, { field: number; payloadStart: number; end: number }>();
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (wire !== WIRE_LENGTH) return;
    if (field === GRAPH_NODE) nodes.push({ node: readNode(reader, fieldStart, payloadStart, fieldEnd), payloadStart });
    else if (field === GRAPH_INITIALIZER) inits.set(tensorName(reader, payloadStart, fieldEnd), { payloadStart, end: fieldEnd });
    else if (field === GRAPH_INPUT || field === GRAPH_OUTPUT) values.set(valueInfoName(reader, payloadStart, fieldEnd), { field, payloadStart, end: fieldEnd });
  });
  const producers = new Map<string, NodeInfo>();
  const consumers = new Map<string, NodeInfo[]>();
  for (const { node } of nodes) {
    for (const output of node.outputs) producers.set(output, node);
    for (const input of node.inputs) consumers.set(input, [...(consumers.get(input) ?? []), node]);
  }
  const dims = (name: string) => {
    const at = inits.get(name);
    if (!at) return null;
    const found: number[] = [];
    forEachField(reader, at.payloadStart, at.end, (f, w, _s, p, e) => {
      if (f === TENSOR_DIMS && w === WIRE_VARINT) found.push(reader.varint(p)[0]);
      else if (f === TENSOR_DIMS && w === WIRE_LENGTH)
        for (let pos = p; pos < e; ) {
          const [v, next] = reader.varint(pos);
          found.push(v);
          pos = next;
        }
    });
    return found;
  };
  const ints = (name: string) => {
    const at = inits.get(name);
    if (!at || tensorType(reader, at.payloadStart, at.end) !== TENSOR_TYPE_INT64) return null;
    let raw: Uint8Array | null = null;
    forEachField(reader, at.payloadStart, at.end, (f, w, _s, p, e) => {
      if (f === TENSOR_RAW_DATA && w === WIRE_LENGTH) raw = reader.slice(p, e);
    });
    const bytes = raw as Uint8Array | null;
    if (!bytes) return null;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return Array.from({ length: bytes.byteLength / 8 }, (_, i) => Number(view.getBigInt64(i * 8, true)));
  };

  const replace = new Map<number, Uint8Array | null>();
  const extra = new Writer();
  const valueInfos = new Map<string, Uint8Array>();
  const added = new Set<string>();
  const constant = (name: string, values: number[]) => {
    if (!added.has(name) && !inits.has(name)) extra.bytes(int64Initializer(name, values));
    added.add(name);
    return name;
  };
  for (const { node, payloadStart } of nodes) {
    if (node.opType !== 'If' || node.outputs.length !== 3) continue;
    const [gatedOut, presentConv, presentRecurrent] = node.outputs as [string, string, string];
    if (!presentConv.startsWith('present_conv.') || !presentRecurrent.startsWith('present_recurrent.')) continue;
    const branch = subgraphNodes(reader, payloadStart, node.end, 'then_branch');
    if (!branch) continue;
    const inner = new Set(branch.flatMap((n) => n.outputs));
    const outer = (name: string) => !inner.has(name) && !inits.has(name);
    const find = (op: string, test: (n: NodeInfo) => boolean = () => true) => branch.find((n) => n.opType === op && test(n));
    const qkv = find('Transpose', (n) => outer(n.inputs[0]!))?.inputs[0];
    const pastConv = find('Slice', (n) => n.inputs[0]!.startsWith('past_conv.'))?.inputs[0];
    const pastRecurrent = find('Cast', (n) => n.inputs[0]!.startsWith('past_recurrent.'))?.inputs[0];
    const convWeight = find('Mul', (n) => n.inputs.includes(presentConv))?.inputs.find((i) => i !== presentConv);
    const split = find('Split');
    const beta = find('Sigmoid', (n) => outer(n.inputs[0]!))?.inputs[0];
    const dtAdd = find('Add', (n) => n.inputs.some((i) => inits.has(i) && dims(i)?.length === 1) && n.inputs.some((i) => producers.get(i) === undefined && inner.has(i)));
    const dtBias = dtAdd?.inputs.find((i) => inits.has(i));
    const aCast = dtAdd && branch.find((n) => n.outputs.includes(dtAdd.inputs.find((i) => !inits.has(i))!));
    const a = aCast?.opType === 'Cast' ? aCast.inputs[0] : undefined;
    const softplus = find('Softplus');
    const decayScale = softplus && find('Mul', (n) => n.inputs.includes(softplus.outputs[0]!))?.inputs.find((i) => i !== softplus.outputs[0]);
    const zReshape = find('Reshape', (n) => outer(n.inputs[0]!) && n.inputs[0] !== qkv);
    const normMul = find('Mul', (n) => n.inputs.some((i) => inits.has(i) && dims(i)?.length === 1 && i !== decayScale));
    const normWeight = normMul?.inputs.find((i) => inits.has(i));
    const sizes = split?.inputs[1] ? ints(split.inputs[1]) : null;
    const headShape = (output: string | undefined) => {
      const reshape = output && find('Reshape', (n) => n.inputs[0] === output);
      return reshape ? ints(reshape.inputs[1]!) : null;
    };
    const qShape = headShape(split?.outputs[0]);
    const vShape = headShape(split?.outputs[2]);
    const weightDims = convWeight ? dims(convWeight) : null;
    const pastConvInfo = pastConv ? values.get(pastConv) : undefined;
    const presentConvInfo = values.get(presentConv);
    if (
      !qkv || !pastConv || !pastRecurrent || !convWeight || !split || !beta || !dtBias || !a || !decayScale || !zReshape || !normWeight ||
      !sizes || sizes.length !== 3 || !qShape || qShape.length !== 4 || !vShape || vShape.length !== 4 ||
      !weightDims || weightDims.length !== 2 || !pastConvInfo || !presentConvInfo
    ) {
      continue;
    }
    const [, , kHeads, dk] = qShape as [number, number, number, number];
    const [, , vHeads, dv] = vShape as [number, number, number, number];
    const [channels, kernel] = weightDims as [number, number];
    if (sizes[0] !== kHeads * dk || sizes[1] !== kHeads * dk || sizes[2] !== vHeads * dv || channels !== sizes[0] + sizes[1] + sizes[2]) continue;

    const p = (node.name || gatedOut).replace(/\/If$/, '') + '/fused';
    const out = new Writer();
    const n = (op: string, name: string, inputs: string[], outputs: string[], attrs: { ints?: Record<string, number>; floats?: Record<string, number>; strings?: Record<string, string>; domain?: string } = {}) =>
      out.bytes(nodeField(op, `${p}/${name}`, inputs, outputs, attrs.ints, attrs.domain, attrs.floats, {}, attrs.strings));
    // Causal convolution over [tokens, channels] with the last kernel − 1 positions as state, SiLU fused.
    n('Reshape', 'conv_weight', [convWeight, constant(`/linear_attention/conv_weight_shape/${channels}x${kernel}`, [channels, 1, kernel])], [`${p}/conv_weight`]);
    n('CausalConvWithState', 'conv', [qkv, `${p}/conv_weight`, '', pastConv], [`${p}/conv`, presentConv], {
      domain: MICROSOFT_DOMAIN,
      ints: { ndim: 1, channels_last: 1 },
      strings: { activation: 'silu' },
    });
    n('Split', 'split', [`${p}/conv`, split.inputs[1]!], [`${p}/q`, `${p}/k`, `${p}/v`], { ints: { axis: -1 } });
    // q and k L2-normalized per head.
    for (const part of ['q', 'k']) {
      n('Reshape', `${part}_heads`, [`${p}/${part}`, constant(`/linear_attention/heads/${kHeads}x${dk}`, [0, 0, kHeads, dk])], [`${p}/${part}_heads`]);
      n('LpNormalization', `${part}_norm`, [`${p}/${part}_heads`], [`${p}/${part}_norm`], { ints: { axis: -1, p: 2 } });
      n('Reshape', `${part}_flat`, [`${p}/${part}_norm`, constant(`/linear_attention/flat/${kHeads * dk}`, [0, 0, kHeads * dk])], [`${p}/${part}_flat`]);
    }
    // β = sigmoid(b); log decay g = A · softplus(a + dt_bias), in float32 as the export computes it.
    n('Sigmoid', 'beta', [beta], [`${p}/beta`]);
    n('Cast', 'a_float', [a], [`${p}/a_float`], { ints: { to: TENSOR_TYPE_FLOAT } });
    n('Add', 'a_dt', [`${p}/a_float`, dtBias], [`${p}/a_dt`]);
    n('Softplus', 'softplus', [`${p}/a_dt`], [`${p}/softplus`]);
    n('Mul', 'decay_float', [decayScale, `${p}/softplus`], [`${p}/decay_float`]);
    n('Cast', 'decay', [`${p}/decay_float`], [`${p}/decay`], { ints: { to: TENSOR_TYPE_FLOAT16 } });
    n('LinearAttention', 'attention', [`${p}/q_flat`, `${p}/k_flat`, `${p}/v`, pastRecurrent, `${p}/decay`, `${p}/beta`], [`${p}/attention`, presentRecurrent], {
      domain: MICROSOFT_DOMAIN,
      ints: { q_num_heads: kHeads, kv_num_heads: vHeads },
      floats: { scale: 1 / Math.sqrt(dk) },
      strings: { update_rule: 'gated_delta' },
    });
    // Gated RMS norm per head, in float32: norm(o) · weight · silu(z).
    n('Reshape', 'o_heads', [`${p}/attention`, constant(`/linear_attention/heads/${vHeads}x${dv}`, [0, 0, vHeads, dv])], [`${p}/o_heads`]);
    n('Cast', 'o_float', [`${p}/o_heads`], [`${p}/o_float`], { ints: { to: TENSOR_TYPE_FLOAT } });
    const weight = tensorType(reader, inits.get(normWeight)!.payloadStart, inits.get(normWeight)!.end) === TENSOR_TYPE_FLOAT ? normWeight : `${p}/norm_weight`;
    if (weight !== normWeight) n('Cast', 'norm_weight', [normWeight], [weight], { ints: { to: TENSOR_TYPE_FLOAT } });
    n('SimplifiedLayerNormalization', 'norm', [`${p}/o_float`, weight], [`${p}/normed_heads`], { ints: { axis: -1, stash_type: 1 }, floats: { epsilon: 1e-6 } });
    n('Reshape', 'normed', [`${p}/normed_heads`, constant(`/linear_attention/flat/${vHeads * dv}`, [0, 0, vHeads * dv])], [`${p}/normed`]);
    n('Cast', 'z_float', [zReshape.inputs[0]!], [`${p}/z_float`], { ints: { to: TENSOR_TYPE_FLOAT } });
    n('Sigmoid', 'z_sigmoid', [`${p}/z_float`], [`${p}/z_sigmoid`]);
    n('Mul', 'z_silu', [`${p}/z_float`, `${p}/z_sigmoid`], [`${p}/z_silu`]);
    n('Mul', 'gated', [`${p}/normed`, `${p}/z_silu`], [`${p}/gated`]);
    n('Cast', 'out', [`${p}/gated`], [gatedOut], { ints: { to: TENSOR_TYPE_FLOAT16 } });
    replace.set(node.start, out.finish());
    // The sequence-length test that fed the If, if nothing else reads it.
    for (let value = node.inputs[0]; value; ) {
      const producer = producers.get(value);
      if (!producer || producer.outputs.some((o) => (consumers.get(o) ?? []).some((c) => c !== node && !replace.has(c.start) && c.start !== node.start))) break;
      if ((consumers.get(value) ?? []).length !== 1) break;
      replace.set(producer.start, null);
      value = producer.inputs.find((i) => producers.has(i)) ?? '';
    }
    for (const [name, info] of [[pastConv, pastConvInfo], [presentConv, presentConvInfo]] as const) {
      const declared = valueInfoTensor(reader, info.payloadStart, info.end);
      valueInfos.set(name, tensorValueInfo(info.field, name, declared.type, [declared.dims[0] ?? 1, kernel - 1, channels]));
    }
  }
  return replace.size > 0 ? { replace, extra: extra.finish(), valueInfos } : null;
}

/** The nodes of a node's graph attribute (an If's then_branch), or null. */
function subgraphNodes(reader: BufferedReader, payloadStart: number, end: number, attribute: string): NodeInfo[] | null {
  let found: NodeInfo[] | null = null;
  forEachField(reader, payloadStart, end, (field, wire, _s, p, e) => {
    if (field !== NODE_ATTRIBUTE || wire !== WIRE_LENGTH || attributeName(reader, p, e) !== attribute) return;
    forEachField(reader, p, e, (af, aw, _s2, gp, ge) => {
      if (af !== ATTRIBUTE_G || aw !== WIRE_LENGTH) return;
      const list: NodeInfo[] = [];
      forEachField(reader, gp, ge, (gf, gw, gs, np, ne) => {
        if (gf === GRAPH_NODE && gw === WIRE_LENGTH) list.push(readNode(reader, gs, np, ne));
      });
      found = list;
    });
  });
  return found;
}

const VALUE_INFO_NAME = 1;
const VALUE_INFO_TYPE = 2;
const TYPE_TENSOR = 1;
const TYPE_TENSOR_ELEM = 1;
const TYPE_TENSOR_SHAPE = 2;
const SHAPE_DIM = 1;
const DIM_VALUE = 1;
const DIM_PARAM = 2;

function valueInfoName(reader: BufferedReader, start: number, end: number): string {
  let name = '';
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === VALUE_INFO_NAME && w === WIRE_LENGTH) name = text(reader, p, e);
  });
  return name;
}

/** A ValueInfoProto's tensor element type and dimensions (numbers, or names for symbolic ones). */
function valueInfoTensor(reader: BufferedReader, start: number, end: number): { type: number; dims: (number | string)[] } {
  let type = 0;
  const dims: (number | string)[] = [];
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f !== VALUE_INFO_TYPE || w !== WIRE_LENGTH) return;
    forEachField(reader, p, e, (tf, tw, _s2, tp, te) => {
      if (tf !== TYPE_TENSOR || tw !== WIRE_LENGTH) return;
      forEachField(reader, tp, te, (xf, xw, _s3, xp, xe) => {
        if (xf === TYPE_TENSOR_ELEM && xw === WIRE_VARINT) type = reader.varint(xp)[0];
        if (xf !== TYPE_TENSOR_SHAPE || xw !== WIRE_LENGTH) return;
        forEachField(reader, xp, xe, (sf, sw, _s4, sp, se) => {
          if (sf !== SHAPE_DIM || sw !== WIRE_LENGTH) return;
          let dim: number | string = 0;
          forEachField(reader, sp, se, (df, dw, _s5, dp, de) => {
            if (df === DIM_VALUE && dw === WIRE_VARINT) dim = reader.varint(dp)[0];
            if (df === DIM_PARAM && dw === WIRE_LENGTH) dim = text(reader, dp, de);
          });
          dims.push(dim);
        });
      });
    });
  });
  return { type, dims };
}

/** A GraphProto input or output field (tag included): a tensor of `type` with these dimensions. */
function tensorValueInfo(field: number, name: string, type: number, dims: (number | string)[]): Uint8Array {
  const shape = new Writer();
  for (const dim of dims) {
    const d = new Writer();
    if (typeof dim === 'number') {
      d.tag(DIM_VALUE, WIRE_VARINT);
      d.varint(dim);
    } else {
      d.tag(DIM_PARAM, WIRE_LENGTH);
      d.lengthDelimited(utf8(dim));
    }
    shape.tag(SHAPE_DIM, WIRE_LENGTH);
    shape.lengthDelimited(d.finish());
  }
  const tensor = new Writer();
  tensor.tag(TYPE_TENSOR_ELEM, WIRE_VARINT);
  tensor.varint(type);
  tensor.tag(TYPE_TENSOR_SHAPE, WIRE_LENGTH);
  tensor.lengthDelimited(shape.finish());
  const typeProto = new Writer();
  typeProto.tag(TYPE_TENSOR, WIRE_LENGTH);
  typeProto.lengthDelimited(tensor.finish());
  const info = new Writer();
  info.tag(VALUE_INFO_NAME, WIRE_LENGTH);
  info.lengthDelimited(utf8(name));
  info.tag(VALUE_INFO_TYPE, WIRE_LENGTH);
  info.lengthDelimited(typeProto.finish());
  const out = new Writer();
  out.tag(field, WIRE_LENGTH);
  out.lengthDelimited(info.finish());
  return out.finish();
}

function planIsInf(reader: BufferedReader, start: number, end: number): { replace: Map<number, Uint8Array>; extra: Uint8Array } {
  const replace = new Map<number, Uint8Array>();
  const extra = new Writer();
  const added = new Set<string>();
  const constant = (name: string, value: number) => {
    if (!added.has(name)) extra.bytes(floatInitializer(name, [value]));
    added.add(name);
    return name;
  };
  const nodes: { node: NodeInfo; payloadStart: number }[] = [];
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field === GRAPH_NODE && wire === WIRE_LENGTH) nodes.push({ node: readNode(reader, fieldStart, payloadStart, fieldEnd), payloadStart });
  });
  const isInfInputs = new Set(nodes.filter(({ node }) => node.opType === 'IsInf' && !node.domain).map(({ node }) => node.inputs[0]!));
  const readers = new Map<string, NodeInfo[]>();
  for (const { node } of nodes) for (const input of node.inputs) readers.set(input, [...(readers.get(input) ?? []), node]);
  for (const { node, payloadStart } of nodes) {
    if (node.opType !== 'Cast' || node.domain || !isInfInputs.has(node.outputs[0]!)) continue;
    if (!(readers.get(node.outputs[0]!) ?? []).every((r) => r.opType === 'IsInf')) continue;
    let to = 0;
    forEachField(reader, payloadStart, node.end, (f, w, _s, p, e) => {
      if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH || attributeName(reader, p, e) !== 'to') return;
      forEachField(reader, p, e, (af, aw, _s2, p2) => {
        if (af === ATTRIBUTE_I && aw === WIRE_VARINT) to = reader.varint(p2)[0];
      });
    });
    if (to === TENSOR_TYPE_DOUBLE) replace.set(node.start, withIntAttribute(reader, payloadStart, node.end, 'to', TENSOR_TYPE_FLOAT));
  }
  for (const { node, payloadStart } of nodes) {
    if (node.opType !== 'IsInf' || node.domain) continue;
    const ints = new Map<string, number>();
    forEachField(reader, payloadStart, node.end, (f, w, _s, p, e) => {
      if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH) return;
      const name = attributeName(reader, p, e);
      forEachField(reader, p, e, (af, aw, _s2, p2) => {
        if (af === ATTRIBUTE_I && aw === WIRE_VARINT) ints.set(name, reader.varint(p2)[0]);
      });
    });
    const positive = ints.get('detect_positive') ?? 1;
    const negative = ints.get('detect_negative') ?? 1;
    const x = node.inputs[0]!;
    const p = `${node.name || node.outputs[0]}/is_inf`;
    const out = new Writer();
    // Both signs: |x| = +∞. One sign: x = +∞ or x = −∞. Neither: |x| = −∞, never true (in x's shape).
    const both = positive === negative;
    const target = positive && !negative ? Infinity : -Infinity;
    const value = both && positive ? Infinity : target;
    const name = value > 0 ? '/is_inf/inf' : '/is_inf/neg_inf';
    out.bytes(nodeField('CastLike', `${p}/inf`, [constant(name, value), x], [`${p}/inf`]));
    if (both) out.bytes(nodeField('Abs', `${p}/abs`, [x], [`${p}/abs`]));
    out.bytes(nodeField('Equal', `${p}/equal`, [both ? `${p}/abs` : x, `${p}/inf`], node.outputs));
    replace.set(node.start, out.finish());
  }
  return { replace, extra: extra.finish() };
}

/** Inputs a static-cache graph takes for GroupQueryAttention (see staticAttentionInputs). */
export const SEQLENS_INPUT = 'qi_seqlens_k';
export const TOTAL_LENGTH_INPUT = 'qi_total_sequence_length';
/** Rotary inputs a static-cache graph may take instead of computing them from position_ids (see staticAttentionInputs). */
export const ROTARY_ROWS_INPUT = 'qi_rotary_rows';
export const ROTARY_COS_INPUT = 'qi_rotary_cos';
export const ROTARY_SIN_INPUT = 'qi_rotary_sin';
const TENSOR_TYPE_INT32 = 6;

/** What staticAttentionInputs made of a graph. */
export interface StaticGraph {
  graph: Uint8Array<ArrayBuffer>;
  /**
   * The rotary frequencies, when RotaryEmbedding takes its cos/sin rows as inputs: [tokens, frequencies] float32,
   * cos(p · f) and sin(p · f) for each new token's position p, and the rows input [1, tokens] int64 (0, 1, …).
   */
  rotaryFrequencies: Float32Array | null;
}

/** Reads a tensor's bytes from external data: (location, offset, length) as the graph names them. */
export type ExternalReader = (location: string, offset: number, length: number) => Uint8Array | null;

/** What an edit can see: the graph's initializers, and reading bytes of one (inline or external). */
interface EditContext {
  inits: Map<string, { payloadStart: number; end: number }>;
  /** Bytes [offset, offset + length) of an initializer's data, or null. */
  readTensor(name: string, offset: number, length: number): Uint8Array | null;
  dims(name: string): number[];
  type(name: string): number;
  /** The graph's nodes in order, its outputs, and the element types its inputs declare. */
  nodes: NodeInfo[];
  outputs: Set<string>;
  inputTypes: Map<string, number>;
  /** A node's INT attribute, or null. */
  intAttribute(node: NodeInfo, name: string): number | null;
}

/**
 * A node's new inputs; with `opType`, an op of that type on them instead (attributes dropped); with `fields`, these node
 * fields in its place, reading `inputs` and writing `outputs` (default: the node's).
 */
interface NodeEdit {
  inputs: string[];
  opType?: string;
  fields?: Uint8Array;
  outputs?: string[];
}

/**
 * Edits a model's graph: `edit` may give a node new inputs, or replace it (see NodeEdit); then nodes no graph output
 * depends on are removed, and `extraInputs` (graph input or initializer fields) added. Null when `edit` changed nothing.
 */
function editGraph(
  model: Uint8Array,
  prepare: (context: EditContext) => {
    edit(node: NodeInfo): NodeEdit | null;
    extraInputs(): Uint8Array[];
  },
  read?: ExternalReader,
): Uint8Array<ArrayBuffer> | null {
  const reader = new BufferedReader(bytesSource(model as Uint8Array<ArrayBuffer>));
  const out = new Writer();
  let changed = false;
  forEachField(reader, 0, model.byteLength, (field, wire, fieldStart, graphStart, graphEnd) => {
    if (field !== MODEL_GRAPH || wire !== WIRE_LENGTH) {
      out.bytes(reader.slice(fieldStart, graphEnd));
      return;
    }
    const inits = new Map<string, { payloadStart: number; end: number }>();
    const graphNodes: { node: NodeInfo; payloadStart: number }[] = [];
    const outputs: string[] = [];
    const inputTypes = new Map<string, number>();
    forEachField(reader, graphStart, graphEnd, (f, w, s2, p, e) => {
      if (w !== WIRE_LENGTH) return;
      if (f === GRAPH_INITIALIZER) inits.set(tensorName(reader, p, e), { payloadStart: p, end: e });
      if (f === GRAPH_NODE) graphNodes.push({ node: readNode(reader, s2, p, e), payloadStart: p });
      if (f === GRAPH_OUTPUT) outputs.push(valueInfoName(reader, p, e));
      if (f === GRAPH_INPUT) inputTypes.set(valueInfoName(reader, p, e), valueInfoTensor(reader, p, e).type);
    });
    const payloadStarts = new Map(graphNodes.map(({ node, payloadStart }) => [node.start, payloadStart]));
    const context: EditContext = {
      inits,
      nodes: graphNodes.map(({ node }) => node),
      outputs: new Set(outputs),
      inputTypes,
      intAttribute(node, name) {
        let found: number | null = null;
        forEachField(reader, payloadStarts.get(node.start)!, node.end, (f, w, _s, p, e) => {
          if (f !== NODE_ATTRIBUTE || w !== WIRE_LENGTH || attributeName(reader, p, e) !== name) return;
          forEachField(reader, p, e, (af, aw, _as, ap) => {
            if (af === ATTRIBUTE_I && aw === WIRE_VARINT) found = reader.varint(ap)[0];
          });
        });
        return found;
      },
      dims(name) {
        const at = inits.get(name);
        const found: number[] = [];
        if (!at) return found;
        forEachField(reader, at.payloadStart, at.end, (f, w, _s, p, e) => {
          if (f === TENSOR_DIMS && w === WIRE_VARINT) found.push(reader.varint(p)[0]);
          else if (f === TENSOR_DIMS && w === WIRE_LENGTH)
            for (let pos = p; pos < e; ) {
              const [v, next] = reader.varint(pos);
              found.push(v);
              pos = next;
            }
        });
        return found;
      },
      type(name) {
        const at = inits.get(name);
        return at ? tensorType(reader, at.payloadStart, at.end) : 0;
      },
      readTensor(name, offset, length) {
        const at = inits.get(name);
        if (!at) return null;
        let bytes: Uint8Array | null = null;
        const external: Record<string, string> = {};
        forEachField(reader, at.payloadStart, at.end, (f, w, _s, p, e) => {
          if (f === TENSOR_RAW_DATA && w === WIRE_LENGTH && offset + length <= e - p) bytes = reader.slice(p + offset, p + offset + length);
          if (f === TENSOR_EXTERNAL_DATA && w === WIRE_LENGTH) {
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
        if (bytes) return bytes;
        if (!read || !external.location) return null;
        return read(external.location, Number(external.offset ?? 0) + offset, length);
      },
    };
    const { edit, extraInputs } = prepare(context);
    let edited = false;
    const nodes = graphNodes.map(({ node, payloadStart }) => {
      const change = edit(node);
      if (change) edited = true;
      return { node, payloadStart, inputs: change?.inputs ?? node.inputs, opType: change?.opType, fields: change?.fields, outputs: change?.outputs ?? node.outputs };
    });
    if (!edited) {
      out.bytes(reader.slice(fieldStart, graphEnd));
      return;
    }
    changed = true;
    const producer = new Map<string, number>();
    nodes.forEach(({ outputs: written }, i) => written.forEach((o) => producer.set(o, i)));
    const live = new Set<number>();
    const pending = [...outputs];
    while (pending.length) {
      const index = producer.get(pending.pop()!);
      if (index === undefined || live.has(index)) continue;
      live.add(index);
      pending.push(...nodes[index]!.inputs.filter(Boolean));
    }
    const byStart = new Map(nodes.map((entry, i) => [entry.node.start, i]));
    const graph = new Writer();
    forEachField(reader, graphStart, graphEnd, (f, w, s2, p, e) => {
      if (f === GRAPH_NODE && w === WIRE_LENGTH) {
        const index = byStart.get(s2)!;
        if (!live.has(index)) return;
        const entry = nodes[index]!;
        if (entry.fields) graph.bytes(entry.fields);
        else if (entry.opType) graph.bytes(nodeField(entry.opType, entry.node.name, entry.inputs, entry.node.outputs, {}, entry.opType === entry.node.opType ? entry.node.domain : ''));
        else graph.bytes(entry.inputs === entry.node.inputs ? reader.slice(s2, e) : withInputs(reader, p, e, entry.inputs));
        return;
      }
      graph.bytes(reader.slice(s2, e));
    });
    for (const input of extraInputs()) graph.bytes(input);
    out.tag(MODEL_GRAPH, WIRE_LENGTH);
    out.lengthDelimited(graph.finish());
  });
  return changed ? out.finish() : null;
}

/**
 * For a static cache (see onnx-adapter.ts): every GroupQueryAttention takes its sequence lengths from two new graph
 * inputs, SEQLENS_INPUT (int32 [1]: positions filled, minus one) and TOTAL_LENGTH_INPUT (int32 scalar: positions
 * filled), instead of from the attention mask, which some exports read by its length (Qwen3.5's: the mask is as long
 * as the buffer there, not the conversation). Its attention bias input, which only masks positions past the
 * conversation, goes too: GroupQueryAttention doesn't read past seqlens_k.
 *
 * RotaryEmbedding nodes whose cos/sin tables the graph builds from position_ids every step (Qwen3.5's multimodal
 * rotary: some 150 small ops per step in the transformers.js exports; table lookups and synthetic row positions in
 * ONNX Runtime GenAI's; both with shape arithmetic on the CPU) take the new tokens' rows of those tables as inputs
 * instead (ROTARY_COS_INPUT, ROTARY_SIN_INPUT, with ROTARY_ROWS_INPUT indexing them): for text, whose three position
 * rows are equal, they are cos and sin of position × each frequency. The frequencies come from the model's inv_freq,
 * or from its cos/sin tables (row 1: f = atan2(sin, cos)).
 *
 * Nodes no output depends on any more (the mask's and positions' shape arithmetic, which ONNX Runtime runs on the CPU
 * and graph capture refuses) are removed. Null if the graph has no GroupQueryAttention.
 */
export function staticAttentionInputs(model: Uint8Array, read?: ExternalReader): StaticGraph | null {
  let rotaryFrequencies: Float32Array | null = null;
  let attention = false;
  const graph = editGraph(
    model,
    (context) => {
      const frequencies = rotaryTableFrequencies(context);
      return {
        edit(node) {
          // Tables built from position_ids every step (not constant caches): the new tokens' rows come in as inputs.
          if (node.opType === 'RotaryEmbedding' && frequencies && node.inputs.length >= 4 && !context.inits.has(node.inputs[2]!) && !context.inits.has(node.inputs[3]!)) {
            rotaryFrequencies = frequencies;
            return { inputs: [node.inputs[0]!, ROTARY_ROWS_INPUT, ROTARY_COS_INPUT, ROTARY_SIN_INPUT, ...node.inputs.slice(4)] };
          }
          if (node.opType === 'GroupQueryAttention' && node.inputs.length >= 7) {
            attention = true;
            const inputs = [...node.inputs];
            inputs[5] = SEQLENS_INPUT;
            inputs[6] = TOTAL_LENGTH_INPUT;
            if (inputs.length > 10) inputs[10] = '';
            while (inputs.length > 7 && inputs[inputs.length - 1] === '') inputs.pop();
            return { inputs };
          }
          return null;
        },
        extraInputs() {
          const added = [tensorValueInfo(GRAPH_INPUT, SEQLENS_INPUT, TENSOR_TYPE_INT32, [1]), tensorValueInfo(GRAPH_INPUT, TOTAL_LENGTH_INPUT, TENSOR_TYPE_INT32, [])];
          if (rotaryFrequencies) {
            const half = (rotaryFrequencies as Float32Array).length;
            added.push(
              tensorValueInfo(GRAPH_INPUT, ROTARY_ROWS_INPUT, TENSOR_TYPE_INT64, [1, 'tokens']),
              tensorValueInfo(GRAPH_INPUT, ROTARY_COS_INPUT, TENSOR_TYPE_FLOAT, ['tokens', half]),
              tensorValueInfo(GRAPH_INPUT, ROTARY_SIN_INPUT, TENSOR_TYPE_FLOAT, ['tokens', half]),
            );
          }
          return added;
        },
      };
    },
    read,
  );
  return graph && attention ? { graph, rotaryFrequencies } : null;
}

/** The rotary frequencies: a float32 initializer named *inv_freq, or row 1 of float32 *cos_cache / *sin_cache tables. */
function rotaryTableFrequencies(context: EditContext): Float32Array | null {
  const names = [...context.inits.keys()];
  const invFreq = names.find((n) => n.endsWith('inv_freq') && context.type(n) === TENSOR_TYPE_FLOAT);
  if (invFreq) {
    const count = context.dims(invFreq).reduce((a, b) => a * b, 1);
    const bytes = context.readTensor(invFreq, 0, count * 4);
    return bytes ? new Float32Array(bytes.slice().buffer) : null;
  }
  const cos = names.find((n) => n.endsWith('cos_cache') && context.type(n) === TENSOR_TYPE_FLOAT && context.dims(n).length === 2);
  const sin = cos && names.find((n) => n === cos.replace(/cos_cache$/, 'sin_cache'));
  if (!cos || !sin) return null;
  const half = context.dims(cos)[1]!;
  const cosRow = context.readTensor(cos, half * 4, half * 4);
  const sinRow = context.readTensor(sin, half * 4, half * 4);
  if (!cosRow || !sinRow) return null;
  const c = new Float32Array(cosRow.slice().buffer);
  const s = new Float32Array(sinRow.slice().buffer);
  return Float32Array.from({ length: half }, (_, j) => Math.atan2(s[j]!, c[j]!));
}

/**
 * Embedding graphs of ONNX Runtime GenAI's multimodal builds take image features and scatter them over the image
 * tokens' rows (through NonZero, which runs on the CPU): for text there are none, so the output is the token lookup
 * itself. The ScatterND writing inputs_embeds becomes an Identity of its data input, and the rest goes unused. Null
 * if the graph has no such scatter.
 */
export function textOnlyEmbeddings(model: Uint8Array): Uint8Array<ArrayBuffer> | null {
  return editGraph(model, () => ({
    edit(node) {
      return node.opType === 'ScatterND' && node.outputs[0] === 'inputs_embeds' ? { inputs: [node.inputs[0]!], opType: 'Identity' } : null;
    },
    extraInputs: () => [],
  }));
}

/**
 * Fewer GPU kernels per token, for WebGPU: float16 graphs often leave float16 for float32 around a few ops (norms, rotary,
 * gates) and come straight back, and each of those Casts is a kernel of its own. See float16Islands and threeDRotary.
 * Null if neither changed anything.
 */
export function leanGraph(model: Uint8Array, read?: ExternalReader): Uint8Array<ArrayBuffer> | null {
  const islands = float16Islands(model, read);
  const rotary = threeDRotary(islands ?? model, read);
  return rotary ?? islands;
}

/**
 * Ops a float16 island may hold, and which of their inputs carry float data ('all', or input positions). Not Softplus:
 * linear attention's log decay (A · softplus(a + dt_bias)) stays in float32, as its error builds up in the recurrent
 * state (Qwen3.5 0.8B's logits drifted from the reference with it in float16).
 */
const ISLAND_OPS: Record<string, 'all' | number[]> = {
  SimplifiedLayerNormalization: 'all',
  SkipSimplifiedLayerNormalization: 'all',
  Add: 'all',
  Sub: 'all',
  Mul: 'all',
  Div: 'all',
  MatMul: 'all',
  Sigmoid: 'all',
  Neg: 'all',
  Reshape: [0],
  Transpose: [0],
  Split: [0],
  Squeeze: [0],
  Unsqueeze: [0],
  Shape: [0],
  RotaryEmbedding: [0, 2, 3],
};

/**
 * Float32 islands in a float16 graph, computed in float16 instead: an island is a group of ops (ISLAND_OPS) whose float
 * inputs come in through Casts to float32 from float16 tensors, or from float32 weights or other float32 tensors, and
 * whose float outputs leave only through Casts to float16. Its entry and exit Casts go, its float32 weights get float16
 * copies, and each other float32 tensor it reads gets one Cast (shared by every island reading it). Norms keep their
 * statistics in float32 inside ONNX Runtime's kernels; the rest computes in float16, as the graph around it does.
 */
function float16Islands(model: Uint8Array, read?: ExternalReader): Uint8Array<ArrayBuffer> | null {
  return editGraph(
    model,
    (context) => {
      const { nodes } = context;
      const producer = new Map<string, number>();
      const consumers = new Map<string, number[]>();
      nodes.forEach((node, i) => {
        node.outputs.forEach((o) => producer.set(o, i));
        for (const input of node.inputs) if (input) consumers.get(input)?.push(i) ?? consumers.set(input, [i]);
      });
      const castTo = (i: number | undefined) => (i !== undefined && nodes[i]!.opType === 'Cast' ? context.intAttribute(nodes[i]!, 'to') : null);
      // Element types, as far as the ops say (0: unknown).
      const types = new Map<string, number>(context.inputTypes);
      for (const name of context.inits.keys()) types.set(name, context.type(name));
      for (let pass = 0, changed = true; changed && pass < 4; pass++) {
        changed = false;
        nodes.forEach((node, i) => {
          if (types.has(node.outputs[0]!)) return;
          const from = (index: number) => types.get(node.inputs[index] ?? '');
          const type =
            node.opType === 'Cast' ? castTo(i)
            : node.opType === 'Shape' ? TENSOR_TYPE_INT64
            : node.opType === 'SimplifiedLayerNormalization' ? from(1)
            : node.opType === 'GatherBlockQuantized' ? from(2)
            : from(0);
          if (type) {
            for (const o of node.outputs) types.set(o, type);
            changed = true;
          }
        });
      }
      const floatInputs = (i: number) => {
        const node = nodes[i]!;
        const positions = ISLAND_OPS[node.opType];
        if (!positions) return null;
        return node.inputs.filter((input, k) => input && (positions === 'all' || positions.includes(k)));
      };
      // An entry: a Cast to float32 of a float16 tensor.
      const entry = (i: number | undefined) => castTo(i) === TENSOR_TYPE_FLOAT && types.get(nodes[i!]!.inputs[0]!) === TENSOR_TYPE_FLOAT16;
      const member = (i: number | undefined) =>
        i !== undefined && !!ISLAND_OPS[nodes[i]!.opType] && (!nodes[i]!.domain || nodes[i]!.domain === MICROSOFT_DOMAIN);
      const assigned = new Set<number>();
      const rename = new Map<number, Map<string, string>>();
      const alias = new Map<string, string>();
      const weights = new Set<string>();
      const casts = new Set<string>();
      nodes.forEach((_, start) => {
        if (!entry(start)) return;
        for (const seed of consumers.get(nodes[start]!.outputs[0]!) ?? []) {
          if (assigned.has(seed) || !member(seed)) continue;
          const members = new Set<number>();
          const exits = new Set<number>();
          const queue = [seed];
          let ok = true;
          while (queue.length && ok) {
            const i = queue.pop()!;
            if (members.has(i)) continue;
            if (!member(i) || assigned.has(i)) {
              ok = false;
              break;
            }
            members.add(i);
            const node = nodes[i]!;
            for (const input of floatInputs(i)!) {
              const from = producer.get(input);
              if (entry(from) || context.inits.has(input) || types.get(input) === TENSOR_TYPE_FLOAT16) continue;
              if (types.get(input) !== TENSOR_TYPE_FLOAT) ok = false;
              else if (member(from) && castTo(from) === null) queue.push(from!);
            }
            node.outputs.forEach((output, k) => {
              if (node.opType === 'Shape' || !output) return;
              const readers = consumers.get(output) ?? [];
              // A norm's mean and inverse deviation outputs: float32 whatever the input.
              if (node.opType.endsWith('SimplifiedLayerNormalization') && k > 0 && (node.opType.startsWith('Skip') ? k < 3 : true)) {
                if (readers.length) ok = false;
                return;
              }
              if (context.outputs.has(output)) ok = false;
              for (const r of readers) {
                if (castTo(r) === TENSOR_TYPE_FLOAT16) exits.add(r);
                else if (member(r)) queue.push(r);
                else ok = false;
              }
            });
          }
          members.forEach((i) => assigned.add(i));
          if (!ok || exits.size === 0) continue;
          for (const i of members) {
            const names = new Map<string, string>();
            for (const input of floatInputs(i)!) {
              const from = producer.get(input);
              if (entry(from)) names.set(input, nodes[from!]!.inputs[0]!);
              else if (context.inits.has(input) && context.type(input) === TENSOR_TYPE_FLOAT) {
                names.set(input, `${input}/qi_float16`);
                weights.add(input);
              } else if (!members.has(from ?? -1) && types.get(input) === TENSOR_TYPE_FLOAT) {
                names.set(input, `${input}/qi_float16`);
                casts.add(input);
              }
            }
            if (names.size) rename.set(i, names);
          }
          // An exit writing a graph output stays, a Cast from float16 to float16.
          for (const exit of exits) if (!context.outputs.has(nodes[exit]!.outputs[0]!)) alias.set(nodes[exit]!.outputs[0]!, nodes[exit]!.inputs[0]!);
        }
      });
      // A Cast for each float32 tensor islands read, placed with its first reader.
      const placed = new Set<string>();
      const reader = new BufferedReader(bytesSource(model as Uint8Array<ArrayBuffer>));
      const index = new Map(nodes.map((node, i) => [node, i]));
      return {
        edit(node) {
          const names = rename.get(index.get(node)!);
          // An island's entry may read another's exit: through to the tensor that exit read.
          const inputs = node.inputs.map((input) => {
            const renamed = names?.get(input) ?? input;
            return alias.get(renamed) ?? renamed;
          });
          if (!names && inputs.every((input, k) => input === node.inputs[k])) return null;
          const added = [...(names?.keys() ?? [])].filter((input) => casts.has(input) && !placed.has(input));
          if (!added.length) return { inputs };
          const fields = new Writer();
          for (const input of added) {
            placed.add(input);
            fields.bytes(nodeField('Cast', `${input}/qi_float16_cast`, [input], [`${input}/qi_float16`], { to: TENSOR_TYPE_FLOAT16 }));
          }
          fields.bytes(rebuiltNode(reader, node, { inputs }));
          return { inputs: [...inputs.filter((input) => !added.some((a) => `${a}/qi_float16` === input)), ...added], fields: fields.finish() };
        },
        extraInputs() {
          return [...weights].map((source) => {
            const dims = context.dims(source);
            const count = dims.reduce((a, b) => a * b, 1);
            const bytes = context.readTensor(source, 0, count * 4);
            if (!bytes) throw new Error(`float16Islands: can't read ${source}`);
            const values = new Float32Array(bytes.slice().buffer);
            return halfInitializer(`${source}/qi_float16`, dims, Uint16Array.from(values, floatToHalf));
          });
        },
      };
    },
    read,
  );
}

/** A node's field rebuilt with other inputs, outputs or INT attributes (set or replaced), the rest kept. */
function rebuiltNode(reader: BufferedReader, node: NodeInfo, change: { inputs?: string[]; outputs?: string[]; ints?: Record<string, number> }): Uint8Array {
  // The payload follows the field's tag and length.
  const payloadStart = reader.varint(reader.varint(node.start)[1])[1];
  const out = new Writer();
  const str = (field: number, value: string) => {
    out.tag(field, WIRE_LENGTH);
    out.lengthDelimited(utf8(value));
  };
  for (const input of change.inputs ?? node.inputs) str(NODE_INPUT, input);
  for (const output of change.outputs ?? node.outputs) str(NODE_OUTPUT, output);
  const ints = change.ints ?? {};
  forEachField(reader, payloadStart, node.end, (field, wire, fieldStart, p, e) => {
    if (field === NODE_INPUT || field === NODE_OUTPUT) return;
    if (field === NODE_ATTRIBUTE && wire === WIRE_LENGTH && attributeName(reader, p, e) in ints) return;
    out.bytes(reader.slice(fieldStart, e));
  });
  for (const [name, value] of Object.entries(ints)) {
    const attribute = new Writer();
    attribute.tag(ATTRIBUTE_NAME, WIRE_LENGTH);
    attribute.lengthDelimited(utf8(name));
    attribute.tag(ATTRIBUTE_I, WIRE_VARINT);
    attribute.varint(value);
    attribute.tag(ATTRIBUTE_TYPE, WIRE_VARINT);
    attribute.varint(ATTRIBUTE_TYPE_INT);
    out.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
    out.lengthDelimited(attribute.finish());
  }
  const field = new Writer();
  field.tag(GRAPH_NODE, WIRE_LENGTH);
  field.lengthDelimited(out.finish());
  return field.finish();
}

/**
 * RotaryEmbedding on [batch, heads, tokens, head size] between a Reshape to 4-D plus Transpose and their inverse: ONNX
 * Runtime's takes [batch, tokens, hidden] with num_heads directly, so the four ops around it go.
 */
function threeDRotary(model: Uint8Array, read?: ExternalReader): Uint8Array<ArrayBuffer> | null {
  return editGraph(
    model,
    (context) => {
      const { nodes } = context;
      const producer = new Map<string, NodeInfo>();
      const consumers = new Map<string, NodeInfo[]>();
      for (const node of nodes) {
        node.outputs.forEach((o) => producer.set(o, node));
        for (const input of node.inputs) if (input) consumers.get(input)?.push(node) ?? consumers.set(input, [node]);
      }
      const only = (name: string) => {
        const list = consumers.get(name) ?? [];
        return list.length === 1 && !context.outputs.has(name) ? list[0]! : null;
      };
      const shape = (name: string | undefined) => {
        if (!name || context.type(name) !== TENSOR_TYPE_INT64) return null;
        const count = context.dims(name).reduce((a, b) => a * b, 1);
        const bytes = context.readTensor(name, 0, count * 8);
        if (!bytes) return null;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return Array.from({ length: count }, (_, k) => Number(view.getBigInt64(k * 8, true)));
      };
      const swap = (node: NodeInfo | null | undefined) => node?.opType === 'Transpose' && node.perm?.join() === '0,2,1,3';
      const reader = new BufferedReader(bytesSource(model as Uint8Array<ArrayBuffer>));
      const plans = new Map<NodeInfo, { input: string; output: string; heads: number }>();
      const done = new Set<NodeInfo>();
      for (const node of nodes) {
        if (node.opType !== 'RotaryEmbedding' || node.domain !== MICROSOFT_DOMAIN) continue;
        const before = producer.get(node.inputs[0]!);
        const reshapeIn = before && producer.get(before.inputs[0]!);
        const after = only(node.outputs[0]!);
        const reshapeOut = after && only(after.outputs[0]!);
        if (!swap(before) || !swap(after) || reshapeIn?.opType !== 'Reshape' || reshapeOut?.opType !== 'Reshape') continue;
        if (only(reshapeIn.outputs[0]!) !== before || only(before!.outputs[0]!) !== node) continue;
        const four = shape(reshapeIn.inputs[1]);
        const three = shape(reshapeOut.inputs[1]);
        if (four?.length !== 4 || four[0] !== 0 || four[1] !== 0 || four[2]! <= 0 || four[3]! <= 0) continue;
        if (three?.length !== 3 || three[0] !== 0 || three[1] !== 0 || three[2] !== four[2]! * four[3]!) continue;
        const heads = context.intAttribute(node, 'num_heads');
        if (heads && heads !== four[2]) continue;
        plans.set(node, { input: reshapeIn.inputs[0]!, output: reshapeOut.outputs[0]!, heads: four[2]! });
        done.add(reshapeOut);
      }
      return {
        edit(node) {
          const plan = plans.get(node);
          if (plan) {
            const inputs = [plan.input, ...node.inputs.slice(1)];
            return { inputs, outputs: [plan.output], fields: rebuiltNode(reader, node, { inputs, outputs: [plan.output], ints: { num_heads: plan.heads } }) };
          }
          // The final Reshape: its output is now the rotary's.
          if (done.has(node)) return { inputs: [], outputs: [], fields: new Uint8Array(0) };
          return null;
        },
        extraInputs: () => [],
      };
    },
    read,
  );
}

/** What graphStates reports about a graph's outputs (see onnx-adapter.ts: static caches). */
export interface GraphStates {
  /** The op type of the node writing each graph output. */
  writers: Map<string, string>;
  /** The graph has control flow (If, Loop, Scan), which ONNX Runtime runs on the CPU: no graph capture. */
  controlFlow: boolean;
}

/** Which op writes each output of a (rewritten) model graph, and whether it has control flow. */
export function graphStates(model: Uint8Array): GraphStates {
  const reader = new BufferedReader(bytesSource(model as Uint8Array<ArrayBuffer>));
  const writers = new Map<string, string>();
  const outputs = new Set<string>();
  let controlFlow = false;
  forEachField(reader, 0, model.byteLength, (field, wire, _s, graphStart, graphEnd) => {
    if (field !== MODEL_GRAPH || wire !== WIRE_LENGTH) return;
    forEachField(reader, graphStart, graphEnd, (f, w, fieldStart, p, e) => {
      if (w !== WIRE_LENGTH) return;
      if (f === GRAPH_OUTPUT) outputs.add(valueInfoName(reader, p, e));
      if (f !== GRAPH_NODE) return;
      const node = readNode(reader, fieldStart, p, e);
      if (node.opType === 'If' || node.opType === 'Loop' || node.opType === 'Scan') controlFlow = true;
      for (const output of node.outputs) writers.set(output, node.opType);
    });
  });
  for (const name of writers.keys()) if (!outputs.has(name)) writers.delete(name);
  return { writers, controlFlow };
}

/** Packed-input MultiHeadAttention nodes with their parts separated (see the top of this file). */
function planPackedAttention(reader: BufferedReader, start: number, end: number): { replace: Map<number, Uint8Array>; extra: Uint8Array } {
  const replace = new Map<number, Uint8Array>();
  const extra = new Writer();
  const added = new Set<string>();
  const constant = (name: string, bytes: () => Uint8Array) => {
    if (!added.has(name)) extra.bytes(bytes());
    added.add(name);
    return name;
  };
  forEachField(reader, start, end, (field, wire, fieldStart, payloadStart, fieldEnd) => {
    if (field !== GRAPH_NODE || wire !== WIRE_LENGTH) return;
    const node = readNode(reader, fieldStart, payloadStart, fieldEnd);
    if (node.opType !== 'MultiHeadAttention' || node.domain !== MICROSOFT_DOMAIN) return;
    const [query, key, value] = node.inputs;
    // Packed QKV: only the query. Packed KV: query and key, no value.
    const packedQkv = !!query && !key && !value;
    const packedKv = !!query && !!key && !value;
    if (!packedQkv && !packedKv) return;
    const p = `${node.name || node.outputs[0]}/unpacked`;
    const out = new Writer();
    const flat = constant('/packed_attention/flat', () => int64Initializer('/packed_attention/flat', [0, 0, -1]));
    const part = (source: string, index: number, name: string) => {
      const at = constant(`/packed_attention/index_${index}`, () => scalarInt64Initializer(`/packed_attention/index_${index}`, index));
      out.bytes(nodeField('Gather', `${p}/${name}_gather`, [source, at], [`${p}/${name}_4d`], { axis: 3 }));
      out.bytes(nodeField('Reshape', `${p}/${name}`, [`${p}/${name}_4d`, flat], [`${p}/${name}`]));
      return `${p}/${name}`;
    };
    const inputs = packedQkv
      ? [part(query!, 0, 'q'), part(query!, 1, 'k'), part(query!, 2, 'v')]
      : [query!, part(key!, 0, 'k'), part(key!, 1, 'v')];
    out.bytes(withInputs(reader, payloadStart, fieldEnd, [...inputs, ...node.inputs.slice(3)]));
    replace.set(fieldStart, out.finish());
  });
  return { replace, extra: extra.finish() };
}

/** A GraphProto.node field: the node with its inputs replaced (attributes and the rest kept). */
function withInputs(reader: BufferedReader, payloadStart: number, end: number, inputs: string[]): Uint8Array {
  const node = new Writer();
  for (const input of inputs) {
    node.tag(NODE_INPUT, WIRE_LENGTH);
    node.lengthDelimited(utf8(input));
  }
  forEachField(reader, payloadStart, end, (field, _wire, fieldStart, _p, e) => {
    if (field !== NODE_INPUT) node.bytes(reader.slice(fieldStart, e));
  });
  const out = new Writer();
  out.tag(GRAPH_NODE, WIRE_LENGTH);
  out.lengthDelimited(node.finish());
  return out.finish();
}

/** A GraphProto.initializer field (tag included): an int64 scalar. */
function scalarInt64Initializer(name: string, value: number): Uint8Array {
  const raw = new DataView(new ArrayBuffer(8));
  raw.setBigInt64(0, BigInt(value), true);
  const tensor = new Writer();
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

/** An AttributeProto's name. */
function attributeName(reader: BufferedReader, start: number, end: number): string {
  let name = '';
  forEachField(reader, start, end, (f, w, _s, p, e) => {
    if (f === ATTRIBUTE_NAME && w === WIRE_LENGTH) name = text(reader, p, e);
  });
  return name;
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

/** A GraphProto.node field (tag included) for an op with INT, FLOAT, INTS and STRING attributes. */
function nodeField(
  opType: string,
  name: string,
  inputs: string[],
  outputs: string[],
  ints: Record<string, number> = {},
  domain = '',
  floats: Record<string, number> = {},
  intLists: Record<string, number[]> = {},
  strings: Record<string, string> = {},
): Uint8Array {
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
  for (const [key, value] of Object.entries(floats)) {
    const attribute = new Writer();
    attribute.tag(ATTRIBUTE_NAME, WIRE_LENGTH);
    attribute.lengthDelimited(utf8(key));
    attribute.tag(ATTRIBUTE_F, WIRE_32BIT);
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value, true);
    attribute.bytes(bytes);
    attribute.tag(ATTRIBUTE_TYPE, WIRE_VARINT);
    attribute.varint(ATTRIBUTE_TYPE_FLOAT);
    node.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
    node.lengthDelimited(attribute.finish());
  }
  for (const [key, value] of Object.entries(strings)) {
    const attribute = new Writer();
    attribute.tag(ATTRIBUTE_NAME, WIRE_LENGTH);
    attribute.lengthDelimited(utf8(key));
    attribute.tag(ATTRIBUTE_S, WIRE_LENGTH);
    attribute.lengthDelimited(utf8(value));
    attribute.tag(ATTRIBUTE_TYPE, WIRE_VARINT);
    attribute.varint(ATTRIBUTE_TYPE_STRING);
    node.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
    node.lengthDelimited(attribute.finish());
  }
  for (const [key, values] of Object.entries(intLists)) {
    const attribute = new Writer();
    attribute.tag(ATTRIBUTE_NAME, WIRE_LENGTH);
    attribute.lengthDelimited(utf8(key));
    for (const v of values) {
      attribute.tag(ATTRIBUTE_INTS, WIRE_VARINT);
      attribute.varint(v);
    }
    attribute.tag(ATTRIBUTE_TYPE, WIRE_VARINT);
    attribute.varint(ATTRIBUTE_TYPE_INTS);
    node.tag(NODE_ATTRIBUTE, WIRE_LENGTH);
    node.lengthDelimited(attribute.finish());
  }
  const field = new Writer();
  field.tag(GRAPH_NODE, WIRE_LENGTH);
  field.lengthDelimited(node.finish());
  return field.finish();
}

/** A GraphProto.initializer field (tag included): a float16 tensor from its bits. */
function halfInitializer(name: string, dims: number[], bits: Uint16Array): Uint8Array {
  const tensor = new Writer();
  for (const dim of dims) {
    tensor.tag(TENSOR_DIMS, WIRE_VARINT);
    tensor.varint(dim);
  }
  tensor.tag(TENSOR_DATA_TYPE, WIRE_VARINT);
  tensor.varint(TENSOR_TYPE_FLOAT16);
  tensor.tag(TENSOR_NAME, WIRE_LENGTH);
  tensor.lengthDelimited(utf8(name));
  tensor.tag(TENSOR_RAW_DATA, WIRE_LENGTH);
  tensor.lengthDelimited(new Uint8Array(bits.buffer, bits.byteOffset, bits.byteLength));
  const field = new Writer();
  field.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
  field.lengthDelimited(tensor.finish());
  return field.finish();
}

/** A GraphProto.initializer field (tag included): a 1-D float32 tensor. */
function floatInitializer(name: string, values: number[]): Uint8Array {
  const raw = new DataView(new ArrayBuffer(values.length * 4));
  values.forEach((v, i) => raw.setFloat32(i * 4, v, true));
  const tensor = new Writer();
  tensor.tag(TENSOR_DIMS, WIRE_VARINT);
  tensor.varint(values.length);
  tensor.tag(TENSOR_DATA_TYPE, WIRE_VARINT);
  tensor.varint(TENSOR_TYPE_FLOAT);
  tensor.tag(TENSOR_NAME, WIRE_LENGTH);
  tensor.lengthDelimited(utf8(name));
  tensor.tag(TENSOR_RAW_DATA, WIRE_LENGTH);
  tensor.lengthDelimited(new Uint8Array(raw.buffer));
  const field = new Writer();
  field.tag(GRAPH_INITIALIZER, WIRE_LENGTH);
  field.lengthDelimited(tensor.finish());
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

function rewriteTensor(
  reader: BufferedReader,
  start: number,
  end: number,
  location: string,
  stats: Stats,
  context: RewriteContext,
): Uint8Array {
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
    if (field !== TENSOR_RAW_DATA && field !== TENSOR_DATA_LOCATION && field !== TENSOR_EXTERNAL_DATA)
      out.bytes(reader.slice(fieldStart, fieldEnd));
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
    if (value < 0) {
      // Negative int64s (attributes such as axis = -1): ten bytes, two's complement.
      let big = BigInt.asUintN(64, BigInt(value));
      for (let i = 0; i < 9; i++) {
        this.buf[this.len++] = Number(big & 0x7fn) | 0x80;
        big >>= 7n;
      }
      this.buf[this.len++] = Number(big);
      return;
    }
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
