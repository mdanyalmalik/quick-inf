"""Replaces GatherBlockQuantized lookups (ONNX Runtime contrib op) with a plain Gather on a float16 table and a Cast
to float32, for graphs that run on ONNX Runtime Web's CPU (WebAssembly) engine, which has no kernel for them. The
table holds the same values (the int4 codes times their block scales), so outputs don't change beyond float16
rounding of the dequantized values. All weights are written into the one output file.

    python tools/export/dequantize_gathers.py IN.onnx OUT.onnx
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def dequantize(codes: np.ndarray, scales: np.ndarray, quantize_axis: int, block: int, zero_points: np.ndarray | None) -> np.ndarray:
    codes = codes.astype(np.float32)
    if zero_points is not None:
        codes = codes - np.repeat(zero_points.astype(np.float32), block, axis=quantize_axis)[tuple(slice(0, n) for n in codes.shape)]
    expanded = np.repeat(scales.astype(np.float32), block, axis=quantize_axis)[tuple(slice(0, n) for n in codes.shape)]
    return codes * expanded


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = onnx.load(src)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    tables: dict[tuple, str] = {}
    nodes, done = [], 0
    for n in g.node:
        if n.op_type != 'GatherBlockQuantized' or n.domain != 'com.microsoft':
            nodes.append(n)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if attrs.get('bits', 4) != 4:
            raise SystemExit(f'{n.name}: only 4-bit lookups are handled')
        key = tuple(n.input[i] if i != 1 else '' for i in range(len(n.input)))
        if key not in tables:
            codes = numpy_helper.to_array(inits[n.input[0]])
            scales = numpy_helper.to_array(inits[n.input[2]])
            zero_points = numpy_helper.to_array(inits[n.input[3]]) if len(n.input) > 3 and n.input[3] else None
            table = dequantize(codes, scales, attrs.get('quantize_axis', 1), attrs.get('block_size', 128), zero_points)
            name = n.input[0] + '/float16'
            g.initializer.append(numpy_helper.from_array(table.astype(np.float16), name))
            tables[key] = name
        half = n.output[0] + '/float16'
        nodes.append(helper.make_node('Gather', [tables[key], n.input[1]], [half], name=n.name + '/Gather', axis=attrs.get('gather_axis', 0)))
        nodes.append(helper.make_node('Cast', [half], list(n.output), name=n.name + '/Cast', to=TensorProto.FLOAT))
        done += 1
    del g.node[:]
    g.node.extend(nodes)
    used = {i for node in g.node for i in node.input}
    kept = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(kept)
    onnx.save(model, dst)
    print(f'{src}: {done} GatherBlockQuantized -> Gather on {len(tables)} float16 tables')


if __name__ == '__main__':
    main()
