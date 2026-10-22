"""Rewrites each attention block of Zonos's exported backbone as one com.microsoft GroupQueryAttention node.

The export (tools/export/export_zonos.py) spells attention out as plain ops: per layer, the rotary tables recomputed
from the positions (Range, Pow, Cos, Sin), Q and K rotated pair by pair (Gather, Mul, Sub, Add, Concat), the KV cache
concatenated, keys and values tiled for the 4 KV heads, two MatMuls and a Softmax: about 230 nodes a layer, many of
them small shape ops that ONNX Runtime Web runs on the CPU (96 ms a decode step on WebGPU). Zonos's attention
(zonos/backbone/_torch.py) is exactly GroupQueryAttention's: 16 query heads, 4 KV heads of 128, rotary over the whole
head with base 10000 on interleaved pairs, causal, scale 1/sqrt(128), cache [batch, 4, positions, 128]. So per layer:

    q, k, v = Split(in_proj)                     (unchanged)
    out, present.i.key, present.i.value = GroupQueryAttention(q, k, v, past.i.key, past.i.value, seqlens_k,
                                                               total_sequence_length, cos_cache, sin_cache)
    out_proj(out)                                (unchanged)

with seqlens_k = total - 1 for every batch row and total = past length + new rows, computed once; cos_cache and
sin_cache are [MAX_POSITIONS, 64] initializers. The graph's inputs and outputs keep their names and shapes.

    python tools/export/zonos_gqa.py IN.onnx OUT.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HEADS, KV_HEADS, HEAD_DIM, BASE = 16, 4, 128, 10000.0
# The prefix (phonemes and settings, a few hundred rows) plus at most 30 s of frames (2580) and the delay's 9.
MAX_POSITIONS = 4096


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    matmuls = [n for n in g.node if n.op_type == 'MatMulNBits']
    layers = sum(1 for i in g.input if i.name.startswith('past_key_values.') and i.name.endswith('.key'))
    assert len(matmuls) >= 4 * layers, len(matmuls)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    inv_freq = 1.0 / (BASE ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    angles = np.outer(np.arange(MAX_POSITIONS, dtype=np.float32), inv_freq)
    g.initializer.extend([
        numpy_helper.from_array(np.cos(angles).astype(np.float32), 'gqa_cos_cache'),
        numpy_helper.from_array(np.sin(angles).astype(np.float32), 'gqa_sin_cache'),
        numpy_helper.from_array(np.array([2], np.int64), 'gqa_axis2'),
        numpy_helper.from_array(np.array([1], np.int64), 'gqa_axis1'),
        numpy_helper.from_array(np.array([0], np.int64), 'gqa_axis0'),
        numpy_helper.from_array(np.array(1, np.int32), 'gqa_one'),
    ])
    # total = past length + new rows (int32 scalar); seqlens_k = total - 1 for each batch row ([batch] int32).
    shared = [
        helper.make_node('Shape', ['past_key_values.0.key'], ['gqa_past_shape']),
        helper.make_node('Gather', ['gqa_past_shape', 'gqa_axis2'], ['gqa_past_len']),
        helper.make_node('Shape', ['inputs_embeds'], ['gqa_in_shape']),
        helper.make_node('Gather', ['gqa_in_shape', 'gqa_axis1'], ['gqa_new_len']),
        helper.make_node('Gather', ['gqa_in_shape', 'gqa_axis0'], ['gqa_batch']),
        helper.make_node('Add', ['gqa_past_len', 'gqa_new_len'], ['gqa_total64']),
        helper.make_node('Cast', ['gqa_total64'], ['gqa_total1'], to=TensorProto.INT32),
        helper.make_node('Squeeze', ['gqa_total1', 'gqa_axis0'], ['gqa_total']),
        helper.make_node('Sub', ['gqa_total1', 'gqa_one'], ['gqa_last']),
        helper.make_node('Expand', ['gqa_last', 'gqa_batch'], ['gqa_seqlens_k']),
    ]

    new_nodes = []
    for i in range(layers):
        in_proj, out_proj = matmuls[4 * i], matmuls[4 * i + 1]
        split = [n for n in consumers[in_proj.output[0]] if n.op_type == 'Split']
        assert len(split) == 1 and len(split[0].output) == 3, f'layer {i}: no 3-way Split after {in_proj.name}'
        q, k, v = split[0].output
        attn_out = f'gqa_{i}_out'
        new_nodes.append(helper.make_node(
            'GroupQueryAttention',
            [q, k, v, f'past_key_values.{i}.key', f'past_key_values.{i}.value', 'gqa_seqlens_k', 'gqa_total',
             'gqa_cos_cache', 'gqa_sin_cache'],
            [attn_out, f'present.{i}.key', f'present.{i}.value'],
            name=f'/layers.{i}/GroupQueryAttention', domain='com.microsoft',
            num_heads=HEADS, kv_num_heads=KV_HEADS, do_rotary=1, rotary_interleaved=1))
        out_proj.input[0] = attn_out

    # Drop the old producers of the present outputs, then everything no longer used.
    outputs = {o.name for o in g.output}
    present = {f'present.{i}.{kv}' for i in range(layers) for kv in ('key', 'value')}
    kept = [n for n in g.node if not (set(n.output) & present)]
    nodes = shared + kept + new_nodes
    while True:
        used = outputs | {i for n in nodes for i in n.input}
        alive = [n for n in nodes if any(o in used for o in n.output)]
        if len(alive) == len(nodes):
            break
        nodes = alive
    # Topological order: GQA nodes after their Splits, out_proj after its GQA.
    produced = {i.name for i in g.input} | {t.name for t in g.initializer}
    ordered, pending = [], nodes
    while pending:
        ready = [n for n in pending if all(not x or x in produced for x in n.input)]
        assert ready, 'cycle or missing input'
        for n in ready:
            ordered.append(n)
            produced.update(n.output)
        ready_ids = {id(n) for n in ready}
        pending = [n for n in pending if id(n) not in ready_ids]
    del g.node[:]
    g.node.extend(ordered)
    used = {i for n in g.node for i in n.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.value_info[:]
    if not any(o.domain == 'com.microsoft' for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    return model


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    m = rewrite(onnx.load(src))
    print({op: sum(n.op_type == op for n in m.graph.node) for op in sorted({n.op_type for n in m.graph.node})})
    onnx.save(m, dst)
