"""Undoes ONNX Runtime's dynamic int8 quantization (quantize_dynamic): every
DynamicQuantizeLinear -> MatMulInteger / ConvInteger -> Cast -> Mul(x_scale * w_scale) chain becomes a float MatMul /
Conv on the original activation, with the weight dequantized to float32 ((q - zero_point) * scale). The result can
then go through quantize_int4.py. Used for models published only in that form (Kitten TTS micro and mini 0.8).

    tools/export/.venv/bin/python -I tools/export/dequantize_dynamic.py IN.onnx OUT.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INTEGER_OPS = {'MatMulInteger': 'MatMul', 'ConvInteger': 'Conv'}


def dequantize(model: onnx.ModelProto) -> int:
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    drop, replace, done = set(), {}, 0
    for n in g.node:
        float_op = INTEGER_OPS.get(n.op_type)
        if not float_op:
            continue
        quantize = producer[n.input[0]]
        cast = consumers[n.output[0]][0]
        mul = consumers[cast.output[0]][0]
        scales = producer[next(i for i in mul.input if i != cast.output[0])]
        assert quantize.op_type == 'DynamicQuantizeLinear' and cast.op_type == 'Cast' and mul.op_type == 'Mul' and scales.op_type == 'Mul'
        w_scale = numpy_helper.to_array(inits[next(i for i in scales.input if i in inits)]).astype(np.float32)
        q = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        zero = numpy_helper.to_array(inits[n.input[3]]).astype(np.float32) if len(n.input) > 3 and n.input[3] else np.float32(0)
        if float_op == 'Conv':
            # Per-output-channel parameters apply along axis 0 of the [out, in, k...] kernel.
            shape = (-1,) + (1,) * (q.ndim - 1)
            w = (q - zero.reshape(shape) if zero.ndim else q - zero) * (w_scale.reshape(shape) if w_scale.ndim else w_scale)
        else:
            w = (q - zero) * w_scale  # per-column parameters broadcast along the last axis of [in, out]
        name = n.input[1].removesuffix('_quantized')
        if name not in inits:  # weights shared by several ops are dequantized once
            inits[name] = numpy_helper.from_array(w.astype(np.float32), name)
            g.initializer.append(inits[name])
        replace[id(n)] = helper.make_node(float_op, [quantize.input[0], name], list(mul.output), name=n.name.replace('Integer', '') or None,
                                          **{a.name: helper.get_attribute_value(a) for a in n.attribute})
        drop.update({id(cast), id(mul)})
        done += 1
    for n in g.node:
        if n.op_type == 'DynamicQuantizeLSTM':
            replace[id(n)] = float_lstm(n, inits, g)
            done += 1
        elif n.op_type == 'Cast' and n.input[0] in inits and inits[n.input[0]].data_type == onnx.TensorProto.FLOAT16:
            # A float16 constant cast up to float32 (Kitten's shared ALBERT weights): stored as float32 instead, so
            # quantize_int4.py sees a constant MatMul weight.
            name = n.input[0] + '/float32'
            if name not in inits:
                inits[name] = numpy_helper.from_array(numpy_helper.to_array(inits[n.input[0]]).astype(np.float32), name)
                g.initializer.append(inits[name])
            replace[id(n)] = helper.make_node('Identity', [name], list(n.output), name=n.name)
    nodes = [replace.get(id(n), n) for n in g.node if id(n) not in drop]
    del g.node[:]
    g.node.extend(nodes)
    prune(model)
    fold_identities(model)
    return done


def float_lstm(n: onnx.NodeProto, inits: dict, g: onnx.GraphProto) -> onnx.NodeProto:
    """com.microsoft DynamicQuantizeLSTM -> ONNX LSTM. Its int8 W and R are [directions, input, 4 * hidden] (ONNX's
    are [directions, 4 * hidden, input]), with a scale and zero point per direction (or per direction and column)."""
    x, w, r, b, lengths, h0, c0, p, w_scale, w_zero, r_scale, r_zero = (list(n.input) + [''] * 12)[:12]
    floats = []
    for q, scale, zero in ((w, w_scale, w_zero), (r, r_scale, r_zero)):
        codes = numpy_helper.to_array(inits[q]).astype(np.float32)
        s = numpy_helper.to_array(inits[scale]).astype(np.float32)
        z = numpy_helper.to_array(inits[zero]).astype(np.float32)
        shape = s.shape + (1,) * (codes.ndim - s.ndim) if s.ndim == 1 else (s.shape[0], 1, s.shape[1])
        weights = ((codes - z.reshape(shape)) * s.reshape(shape)).transpose(0, 2, 1)
        name = q.removesuffix('_quantized')
        g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(weights), name))
        floats.append(name)
    inputs = [x, *floats, b, lengths, h0, c0, p]
    while inputs and not inputs[-1]:
        inputs.pop()
    return helper.make_node('LSTM', inputs, list(n.output), name=n.name.removesuffix('_quant'),
                            **{a.name: helper.get_attribute_value(a) for a in n.attribute})


def subgraph_reads(n: onnx.NodeProto) -> set:
    """Every value read inside n's subgraphs (Loop / If / Scan bodies, at any depth): outer-scope values among them
    must survive pruning."""
    reads = set()
    for a in n.attribute:
        for sub in [a.g] if a.type == onnx.AttributeProto.GRAPH else list(a.graphs):
            for inner in sub.node:
                reads.update(inner.input)
                reads.update(subgraph_reads(inner))
            reads.update(o.name for o in sub.output)
    return reads


def fold_identities(model: onnx.ModelProto):
    """Points readers of Identity(constant) straight at the constant, so the weight has a single reader."""
    g = model.graph
    inits = {i.name for i in g.initializer}
    outputs = {o.name for o in g.output}
    alias = {n.output[0]: n.input[0] for n in g.node if n.op_type == 'Identity' and n.input[0] in inits and n.output[0] not in outputs}
    nodes = [n for n in g.node if not (n.op_type == 'Identity' and n.output[0] in alias)]
    for n in nodes:
        for k, i in enumerate(n.input):
            if i in alias:
                n.input[k] = alias[i]
    del g.node[:]
    g.node.extend(nodes)


def prune(model: onnx.ModelProto):
    """Removes nodes and initializers nothing reads any more (the activation quantizers, scale products, int8 weights)."""
    g = model.graph
    needed = {o.name for o in g.output}
    # Repeated to a fixed point: Kitten's exporter places a Loop before the node producing a value its body reads.
    keep, grown = set(), True
    while grown:
        grown = False
        for k, n in enumerate(g.node):
            if k not in keep and any(o in needed for o in n.output):
                keep.add(k)
                needed.update(n.input)
                needed.update(subgraph_reads(n))
                grown = True
    kept = [n for k, n in enumerate(g.node) if k in keep]
    del g.node[:]
    g.node.extend(kept)
    inits = [i for i in g.initializer if i.name in needed]
    del g.initializer[:]
    g.initializer.extend(inits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.src))
    count = dequantize(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    print(f'{args.src.name}: {count} integer ops -> float; {args.src.stat().st_size / 1e6:.1f} -> {args.dst.stat().st_size / 1e6:.1f} MB')
