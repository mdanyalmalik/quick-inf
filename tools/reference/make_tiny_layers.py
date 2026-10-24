"""Writes src/runtime/onnx/__fixtures__/tiny-layers.onnx: a three-layer toy decoder for moe-split.test.ts's splitLayers
tests, named like ONNX Runtime GenAI's builder names nodes (/model/embed_tokens/…, /model/layers.N/…, /lm_head/…).

Each layer adds a projection of its normed input to the residual stream (a fused skip-add with the next layer's norm,
as the builder fuses them, so two values cross every cut), scaled by a value computed from the attention mask (used
by every layer: recomputed per segment, not passed), and appends its input to its own cache (past_key_values.N.key ->
present.N.key). The weights are random, float32, stored in the graph.

    tools/export/.venv/bin/python tools/reference/make_tiny_layers.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

OUT = Path(__file__).resolve().parents[2] / 'src/runtime/onnx/__fixtures__/tiny-layers.onnx'
H, V, LAYERS = 16, 32, 3


def main():
    rng = np.random.default_rng(0)
    inits = [numpy_helper.from_array(rng.standard_normal((V, H)).astype(np.float32), 'embed.weight'),
             numpy_helper.from_array(rng.standard_normal((H, V)).astype(np.float32), 'lm_head.weight')]
    nodes = [
        helper.make_node('Gather', ['embed.weight', 'input_ids'], ['h0'], name='/model/embed_tokens/Gather'),
        helper.make_node('Cast', ['attention_mask'], ['mask_f'], name='/model/attn_mask/Cast', to=T.FLOAT),
        helper.make_node('ReduceMean', ['mask_f'], ['mask_scale'], name='/model/attn_mask/ReduceMean', keepdims=0),
    ]
    inputs = [helper.make_tensor_value_info('input_ids', T.INT64, [1, 'seq']), helper.make_tensor_value_info('attention_mask', T.INT64, [1, 'total'])]
    outputs = [helper.make_tensor_value_info('logits', T.FLOAT, [1, 'seq', V])]
    residual, delta = 'h0', None
    for i in range(LAYERS):
        p = f'/model/layers.{i}'
        inits += [numpy_helper.from_array(rng.standard_normal((H, H)).astype(np.float32) * 0.3, f'w{i}'),
                  numpy_helper.from_array(np.ones(H, np.float32), f'norm{i}')]
        if delta is None:
            nodes.append(helper.make_node('SimplifiedLayerNormalization', [residual, f'norm{i}'], [f'n{i}'], name=f'{p}/input_layernorm/LayerNorm', axis=-1, epsilon=1e-5))
        else:
            nodes.append(helper.make_node('SkipSimplifiedLayerNormalization', [delta, residual, f'norm{i}'], [f'n{i}', '', '', f'r{i}'],
                                          name=f'{p}/input_layernorm/SkipLayerNorm', domain='com.microsoft', epsilon=1e-5))
            residual = f'r{i}'
        nodes += [
            helper.make_node('MatMul', [f'n{i}', f'w{i}'], [f'm{i}'], name=f'{p}/mlp/MatMul'),
            helper.make_node('Mul', [f'm{i}', 'mask_scale'], [f'd{i}'], name=f'{p}/mlp/Mul'),
            helper.make_node('Concat', [f'past_key_values.{i}.key', f'n{i}'], [f'present.{i}.key'], name=f'{p}/attn/Concat', axis=1),
        ]
        delta = f'd{i}'
        inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.key', T.FLOAT, [1, 'past', H]))
        outputs.append(helper.make_tensor_value_info(f'present.{i}.key', T.FLOAT, [1, 'total', H]))
    inits.append(numpy_helper.from_array(np.ones(H, np.float32), 'final_norm'))
    nodes += [
        helper.make_node('SkipSimplifiedLayerNormalization', [delta, residual, 'final_norm'], ['final'], name=f'/model/layers.{LAYERS}/final_norm_layernorm/SkipLayerNorm',
                         domain='com.microsoft', epsilon=1e-5),
        helper.make_node('MatMul', ['final', 'lm_head.weight'], ['logits'], name='/lm_head/MatMul'),
    ]
    graph = helper.make_graph(nodes, 'tiny-layers', inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('com.microsoft', 1)], ir_version=8)
    onnx.save(model, str(OUT))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
