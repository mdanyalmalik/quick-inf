"""Rewrites the inverse STFT's ConvTranspose in NeuCodec's ONNX decoder as MatMul plus overlap-add.

ONNX Runtime Web's WebGPU ConvTranspose gives wrong results for this one (1922 input channels, one output channel,
kernel 1920, stride 480; cosine 0.5 against the CPU), while every node before it matches. The same sum is

    frames = x^T @ W                         [1, T, 1920]   (W: the basis as [1922, 1920])
    out    = sum_j pad(frames[:, :, j], j)   [1, T + 3, 480] (j = 0..3: block j of each frame, shifted j hops)

reshaped to [1, 1, 480 (T + 3)], the ConvTranspose's output.

    python tools/export/neucodec_overlap_add.py IN.onnx OUT.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    idx, node = next((i, n) for i, n in enumerate(g.node) if n.op_type == 'ConvTranspose')
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    hop = attrs['strides'][0]
    assert attrs.get('group', 1) == 1 and list(attrs.get('pads', [0, 0])) == [0, 0], attrs
    x, w = node.input[0], node.input[1]
    out = node.output[0]
    shape_of = {i.name: list(i.dims) for i in g.initializer}
    # The basis may be an initializer or a DequantizeLinear of one: its shape is [channels, 1, kernel].
    producer = next((n for n in g.node if w in n.output), None)
    wshape = shape_of.get(w) or shape_of[producer.input[0]]
    channels, _, kernel = wshape
    blocks = kernel // hop
    assert blocks * hop == kernel, (kernel, hop)

    def const(name, value):
        g.initializer.append(numpy_helper.from_array(np.asarray(value, np.int64), name))
        return name

    p = 'istft_ola/'
    nodes = [
        helper.make_node('Reshape', [w, const(p + 'w_shape', [channels, kernel])], [p + 'w2']),
        helper.make_node('Transpose', [x], [p + 'xt'], perm=[0, 2, 1]),
        helper.make_node('MatMul', [p + 'xt', p + 'w2'], [p + 'frames']),
        helper.make_node('Reshape', [p + 'frames', const(p + 'f_shape', [0, 0, blocks, hop])], [p + 'blocks']),
    ]
    parts = []
    for j in range(blocks):
        nodes.append(helper.make_node('Gather', [p + 'blocks', const(p + f'j{j}', j)], [p + f'b{j}'], axis=2))
        nodes.append(helper.make_node('Pad', [p + f'b{j}', const(p + f'pad{j}', [0, j, 0, 0, blocks - 1 - j, 0])], [p + f'p{j}']))
        parts.append(p + f'p{j}')
    nodes.append(helper.make_node('Sum', parts, [p + 'sum']))
    nodes.append(helper.make_node('Reshape', [p + 'sum', const(p + 'o_shape', [0, 1, -1])], [out]))
    del g.node[idx]
    for k, n in enumerate(nodes):
        g.node.insert(idx + k, n)
    return model


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = rewrite(onnx.load(src))
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print('rewritten', dst)


if __name__ == '__main__':
    main()
