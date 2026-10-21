"""Int4 weight-only quantization of an ONNX model, without calibration (round to nearest, symmetric), for small
speech models whose weights are mostly convolutions, where ONNX Runtime's MatMulNBits alone reaches little:

- MatMul with a constant matrix (and Gemm with one, rewritten as MatMul + Add first) -> MatMulNBits, 4-bit,
  block 32 along the inner dimension (ONNX Runtime's own quantizer); embedding lookups (Gather of a constant table)
  -> GatherBlockQuantized.
- Conv, ConvTranspose and LSTM weights -> a blocked int4 DequantizeLinear feeding the op (opset 21): blocks of 32
  along the input channels (the kernel's input axis for LSTM), one scale per block. ONNX Runtime's WebGPU backend
  decodes these on the GPU.
- Everything else (biases, norms, small tensors under 4 KB) stays as it is, except float64 sections, which become
  float32 (WebGPU has none).
- Float16 graphs work too: their convolution weights keep float16 scales (Audio8's codec).

The model is first converted to opset 21 if it is older, the first with int4 and blocked DequantizeLinear.

`--conv-bits 8` stores the Conv / ConvTranspose / LSTM weights as int8 instead (the MatMuls stay 4-bit): vocoders,
which turn features into the waveform, buzz audibly at 4 bits (Supertonic 3, Paradee). `--matmul-bits 8` does the
same for the MatMuls (and leaves embedding lookups float). `--gather-bits 0|4` sets the embedding lookups apart from
the MatMuls: 0 keeps them float (a codec's codebooks), 4 quantizes them to 4 bits. `--exclude TEXT` (repeatable) keeps
every node whose name contains TEXT float: a small sensitive block of a larger graph.

    tools/export/.venv/bin/python tools/export/quantize_int4.py IN.onnx OUT.onnx [--block 32] [--conv-bits 4|8] [--matmul-bits 4|8] [--gather-bits 0|4] [--exclude TEXT ...]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter

MIN_BYTES = 4096


def upgrade(model: onnx.ModelProto) -> onnx.ModelProto:
    version = next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)
    if version >= 21:
        return model
    return version_converter.convert_version(model, 21)


def gemm_to_matmul(model: onnx.ModelProto) -> int:
    """Gemm(A, constant B[, C]) with alpha = beta = 1 and no transA -> MatMul (+ Add): MatMulNBits then covers it."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes, done = [], 0
    for n in g.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type != 'Gemm' or n.input[1] not in inits or attrs.get('transA', 0) or attrs.get('alpha', 1.0) != 1.0 or attrs.get('beta', 1.0) != 1.0:
            nodes.append(n)
            continue
        b = numpy_helper.to_array(inits[n.input[1]])
        if attrs.get('transB', 0):
            b = b.T
        name = n.input[1] + '/matmul'
        g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(b), name))
        if len(n.input) > 2 and n.input[2]:
            product = n.output[0] + '/matmul'
            nodes.append(helper.make_node('MatMul', [n.input[0], name], [product], name=(n.name or n.output[0]) + '/MatMul'))
            nodes.append(helper.make_node('Add', [product, n.input[2]], list(n.output), name=(n.name or n.output[0]) + '/Add'))
        else:
            nodes.append(helper.make_node('MatMul', [n.input[0], name], list(n.output), name=(n.name or n.output[0]) + '/MatMul'))
        done += 1
    del g.node[:]
    g.node.extend(nodes)
    return done


def float64_to_float32(model: onnx.ModelProto) -> int:
    """Runs float64 sections in float32: WebGPU has no doubles, and ONNX Runtime Web has no CPU kernels for some
    (Paradee's pitch-phase accumulation: Cast to double, CumSum, Floor, ConstantOfShape). Casts to double become
    casts to float, double constants become float ones."""
    g = model.graph
    done = 0
    for n in g.node:
        for a in n.attribute:
            if n.op_type == 'Cast' and a.name == 'to' and a.i == TensorProto.DOUBLE:
                a.i = TensorProto.FLOAT
                done += 1
            elif a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.DOUBLE:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
                done += 1
    for i in g.initializer:
        if i.data_type == TensorProto.DOUBLE:
            i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.float32), i.name))
            done += 1
    for v in list(g.value_info):
        if v.type.tensor_type.elem_type == TensorProto.DOUBLE:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    return done


def int4_tensor(name: str, q: np.ndarray) -> TensorProto:
    """A TensorProto of type INT4 from values in [-8, 7] (two per byte, the first in the low nibble)."""
    flat = q.astype(np.int8).reshape(-1)
    if flat.size % 2:
        flat = np.append(flat, np.int8(0))
    packed = ((flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)).astype(np.uint8)
    return helper.make_tensor(name, TensorProto.INT4, q.shape, packed.tobytes(), raw=True)


def int8_tensor(name: str, q: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(q.astype(np.int8), name)


def blocked_int(w: np.ndarray, axis: int, block: int, bits: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Symmetric `bits`-bit codes and float32 scales of w, in blocks of `block` along `axis` (the last block may be
    short)."""
    top = (1 << (bits - 1)) - 1
    d = w.shape[axis]
    block = min(block, d)
    blocks = -(-d // block)
    pad = blocks * block - d
    moved = np.moveaxis(w, axis, -1)
    if pad:
        moved = np.concatenate([moved, np.zeros(moved.shape[:-1] + (pad,), w.dtype)], axis=-1)
    grouped = moved.reshape(moved.shape[:-1] + (blocks, block))
    scale = np.abs(grouped).max(-1) / top
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(grouped / scale[..., None]), -top - 1, top)
    q = q.reshape(moved.shape)[..., :d]
    return np.moveaxis(q, -1, axis), np.moveaxis(scale, -1, axis).astype(np.float32), block


# Weight inputs and the axis their blocks run along: the input channels (for LSTM, W's and R's inner axis).
DQ_WEIGHTS = {'Conv': ((1, 1),), 'ConvTranspose': ((1, 0),), 'LSTM': ((1, 2), (2, 2))}


def excluded(node: onnx.NodeProto, exclude: tuple[str, ...]) -> bool:
    return any(e in node.name for e in exclude)


def dequantize_weights(model: onnx.ModelProto, block: int, bits: int, exclude: tuple[str, ...] = ()) -> int:
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    readers = {}
    for n in g.node:
        for i in n.input:
            readers[i] = readers.get(i, 0) + 1
    nodes, replaced, done = [], set(), 0
    for n in g.node:
        for index, axis in () if excluded(n, exclude) else DQ_WEIGHTS.get(n.op_type, ()):
            name = n.input[index]
            init = inits.get(name)
            if init is not None and init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16) and readers.get(name) == 1:
                # A float16 weight is quantized in float32 and keeps float16 scales, so it dequantizes to float16.
                w = numpy_helper.to_array(init)
                half = w.dtype == np.float16
                w = w.astype(np.float32)
                if w.nbytes >= MIN_BYTES:
                    # A depthwise kernel has one input channel: its blocks run along the output channels instead.
                    use = axis if w.shape[axis] > 1 else 0
                    q, scale, size = blocked_int(w, use, block, bits)
                    codes = f'{name}/int{bits}'
                    g.initializer.append(int4_tensor(codes, q) if bits == 4 else int8_tensor(codes, q))
                    g.initializer.append(numpy_helper.from_array(scale.astype(np.float16) if half else scale, name + '/scale'))
                    nodes.append(helper.make_node('DequantizeLinear', [codes, name + '/scale'], [name + '/dequantized'],
                                                  name=name + '/DequantizeLinear', axis=use, block_size=size))
                    n.input[index] = name + '/dequantized'
                    replaced.add(name)
                    done += 1
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    kept = [i for i in g.initializer if i.name not in replaced]
    del g.initializer[:]
    g.initializer.extend(kept)
    return done


def untie_matmul_weights(model: onnx.ModelProto) -> int:
    """Gives every MatMul its own copy of a weight that several MatMuls read (an unrolled loop, e.g. a diffusion
    sampler), since ONNX Runtime's quantizer leaves shared weights float; dedupe_initializers() merges the
    quantized copies again."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    # The exporter writes the repeated reads of a weight as Identity nodes of it: read the weight directly.
    alias = {n.output[0]: n.input[0] for n in g.node if n.op_type == 'Identity' and n.input[0] in inits}
    for n in g.node:
        if n.op_type == 'MatMul' and len(n.input) > 1 and n.input[1] in alias:
            n.input[1] = alias[n.input[1]]
    readers = {}
    for n in g.node:
        if n.op_type == 'MatMul' and len(n.input) > 1 and n.input[1] in inits:
            readers.setdefault(n.input[1], []).append(n)
    done = 0
    for name, nodes in readers.items():
        for k, n in enumerate(nodes[1:], 1):
            copy = TensorProto()
            copy.CopyFrom(inits[name])
            copy.name = f'{name}/copy{k}'
            g.initializer.append(copy)
            n.input[1] = copy.name
            done += 1
    return done


def dedupe_initializers(model: onnx.ModelProto) -> int:
    """Points readers of byte-identical initializers at one copy and drops the rest."""
    import hashlib
    g = model.graph
    first, rename = {}, {}
    for i in g.initializer:
        key = (i.data_type, tuple(i.dims), hashlib.sha1(i.raw_data or i.SerializeToString()).hexdigest())
        if key in first:
            rename[i.name] = first[key]
        else:
            first[key] = i.name
    for n in g.node:
        for j, x in enumerate(n.input):
            if x in rename:
                n.input[j] = rename[x]
    # Drop Identity nodes nobody reads any more, then initializers nobody reads.
    outputs = {o.name for o in g.output}
    while True:
        used = {x for n in g.node for x in n.input} | outputs
        dead = [n for n in g.node if n.op_type == 'Identity' and n.output[0] not in used]
        if not dead:
            break
        for n in dead:
            g.node.remove(n)
    used = {x for n in g.node for x in n.input} | outputs
    kept = [i for i in g.initializer if i.name not in rename and i.name in used]
    del g.initializer[:]
    g.initializer.extend(kept)
    return len(rename)


def quantize(src: Path, dst: Path, block: int, conv_bits: int = 4, matmul_bits: int = 4, gather_bits: int | None = None,
             exclude: tuple[str, ...] = (), untie: bool = False):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    # Upgrade the graph before reading external weights in: the converter serializes the model, which protobuf
    # cannot do past 2 GB.
    from onnx.external_data_helper import load_external_data_for_model
    model = upgrade(onnx.load(str(src), load_external_data=False))
    load_external_data_for_model(model, str(src.parent))
    doubles = float64_to_float32(model)
    gemms = gemm_to_matmul(model)
    if untie:
        print(f'{untie_matmul_weights(model)} shared MatMul weights untied')
    # Embedding lookups follow the MatMuls (ONNX Runtime's quantizer has no 8-bit Gather, so --matmul-bits 8 leaves
    # them float) unless --gather-bits says otherwise: 0 keeps them float, 4 quantizes them to 4 bits.
    if gather_bits is None:
        gather_bits = 4 if matmul_bits == 4 else 0
    passes = [(matmul_bits, ('MatMul', 'Gather') if gather_bits == matmul_bits else ('MatMul',))]
    if gather_bits and gather_bits != matmul_bits:
        passes.append((gather_bits, ('Gather',)))
    for bits, ops in passes:
        skip = [n.name for n in model.graph.node if excluded(n, exclude)]
        q = MatMulNBitsQuantizer(model, bits=bits, block_size=block, is_symmetric=True, op_types_to_quantize=ops,
                                 quant_axes=(('MatMul', 0), ('Gather', 1)), nodes_to_exclude=skip)
        q.process()
        model = q.model.model
    convs = dequantize_weights(model, block, conv_bits, exclude)
    if untie:
        print(f'{dedupe_initializers(model)} duplicate initializers merged')
    matmuls = sum(1 for n in model.graph.node if n.op_type == 'MatMulNBits')
    gathers = sum(1 for n in model.graph.node if n.op_type == 'GatherBlockQuantized')
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst), save_as_external_data=model.ByteSize() > 2_000_000_000)
    print(f'{src.name}: {doubles} float64 -> float32, {gemms} Gemm -> MatMul, {matmuls} MatMulNBits, {gathers} GatherBlockQuantized, {convs} int{conv_bits} conv/LSTM weights; '
          f'{src.stat().st_size / 1e6:.1f} -> {dst.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--conv-bits', type=int, choices=(4, 8), default=4)
    parser.add_argument('--matmul-bits', type=int, choices=(4, 8), default=4)
    parser.add_argument('--gather-bits', type=int, choices=(0, 4), default=None)
    parser.add_argument('--exclude', action='append', default=[], help='keep nodes whose name contains this float (repeatable)')
    parser.add_argument('--untie', action='store_true', help='quantize MatMul weights shared by several MatMuls (one copy kept)')
    args = parser.parse_args()
    quantize(args.src, args.dst, args.block, args.conv_bits, args.matmul_bits, args.gather_bits, tuple(args.exclude), args.untie)
