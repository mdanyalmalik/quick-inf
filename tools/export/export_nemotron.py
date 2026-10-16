"""Exports NVIDIA Nemotron 3 Nano 30B-A3B (NemotronHForCausalLM: Mamba-2, attention and mixture-of-experts blocks) to an
ONNX decoder in ONNX Runtime GenAI's layout, which ONNX Runtime GenAI's own builder doesn't support. Weights stream
from the checkpoint one shard at a time (see onnx_export.py).

Each of the 52 blocks is RMSNorm, one mixer and a residual add; the mixers follow config.hybrid_override_pattern:

- M, Mamba-2: in_proj → [z | xBC | dt]; xBC through a causal depthwise convolution with SiLU (CausalConvWithState,
  state past_key_values.{i}.conv_state); then the state-space scan as LinearAttention's "gated" rule, in float32 as
  Nemotron keeps its SSM state: query = C, key = B (each of the 8 groups shared by 8 of the 64 heads), value = dt·x,
  decay = dt·A (log space), state past_key_values.{i}.recurrent_state [64, 128, 64]. Then + D·x, gated RMSNorm in
  groups of 512 (× SiLU(z)) and out_proj.
- *, attention: GroupQueryAttention, 32 query heads and 2 KV heads of 128, no positional encoding.
- E, mixture of experts: a DeepSeek-V3-style router (sigmoid + correction bias, top 6 of 128, normalised, × 2.5) and
  non-gated experts with squared ReLU, plus a shared expert. ONNX Runtime's WebGPU QMoE only offers a fused SwiGLU
  (g·σ(αg)·(u + β)) for gated experts, so each expert's up projection is stored twice, as both halves: with g = u, a
  large α and β = 0 that is ReLU(u)·u = ReLU(u)², at the cost of storing the up projections twice.

Linear layers: 4-bit MatMulNBits (block 32); experts: 4-bit QMoE (block 64); embeddings, activations and KV cache
float16.

    tools/export/.venv/bin/python tools/export/export_nemotron.py [--layers N] [--bits 8] [--out DIR]
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper

from onnx_export import F16, F32, I64, MS, U8, Graph, Settings, Weights, attention_lengths, matmul_nbits, qmoe_weights, rms_norm, sigmoid_router, skip_rms_norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'Nemotron-3-Nano-30B-A3B'
REPO = 'NVIDIA-Nemotron-3-Nano-30B-A3B-ONNX-WebGPU'
SOURCE = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
REVISION = 'bf77c3174f68ad409e1c2aa60daeb46e32d1c606'
EXPERT_BLOCK = 64


def mamba(g: Graph, W, cfg: dict, i: int, x: str) -> str:
    p = f'backbone.layers.{i}.mixer'
    name = f'/model/layers.{i}/mamba'
    heads, head_dim, groups, state = cfg['mamba_num_heads'], cfg['mamba_head_dim'], cfg['n_groups'], cfg['ssm_state_size']
    inner = heads * head_dim
    conv_dim = inner + 2 * groups * state

    proj = matmul_nbits(g, x, W(f'{p}.in_proj.weight'), f'{name}/in_proj/MatMul')
    z, xbc, dt = g.node('Split', [proj, g.const(f'{name}/split', [inner, conv_dim, heads])], f'{name}/Split', outputs=3, axis=-1)

    # Causal depthwise convolution over [batch, channels, time], with its carried state.
    xbc_t = g.node('Transpose', [xbc], f'{name}/conv/Transpose', perm=[0, 2, 1])
    conv_w = g.init(f'{p}.conv1d.weight', W(f'{p}.conv1d.weight').numpy().astype(np.float16), F16)
    conv_b = g.init(f'{p}.conv1d.bias', W(f'{p}.conv1d.bias').numpy().astype(np.float16), F16)
    conv_out = f'{name}/conv/output_0'
    g.node(
        'CausalConvWithState',
        [xbc_t, conv_w, conv_b, f'past_key_values.{i}.conv_state'],
        f'{name}/conv',
        outputs=[conv_out, f'present.{i}.conv_state'],
        domain=MS,
        activation='silu',
    )
    xbc = g.node('Transpose', [conv_out], f'{name}/conv/Transpose_back', perm=[0, 2, 1])
    xbc = g.node('Cast', [xbc], f'{name}/Cast_xbc', to=F32)
    xs, b, c = g.node('Split', [xbc, g.const(f'{name}/split_xbc', [inner, groups * state, groups * state])], f'{name}/Split_xbc', outputs=3, axis=-1)

    # dt = softplus(dt + dt_bias), at least time_step_min; decay = dt·A with A = -exp(A_log).
    dt = g.node('Cast', [dt], f'{name}/Cast_dt', to=F32)
    dt = g.node('Add', [dt, g.init(f'{p}.dt_bias', W(f'{p}.dt_bias').numpy(), F32)], f'{name}/dt/Add')
    dt = g.node('Softplus', [dt], f'{name}/dt/Softplus')
    dt = g.node('Max', [dt, g.const(f'{name}/dt/min', cfg['time_step_min'], F32)], f'{name}/dt/Max')
    a = -torch.exp(W(f'{p}.A_log'))
    decay = g.node('Mul', [dt, g.init(f'{p}.A', a.numpy(), F32)], f'{name}/decay')

    # value = dt·x per head; query = C and key = B, each group repeated for its heads.
    xh = g.node('Reshape', [xs, g.const(f'{name}/heads_shape', [0, 0, heads, head_dim])], f'{name}/x_heads')
    v = g.node('Mul', [xh, g.node('Unsqueeze', [dt, g.const(f'{name}/last', [-1])], f'{name}/dt/Unsqueeze')], f'{name}/v/Mul')
    v = g.node('Reshape', [v, g.const(f'{name}/flat_v', [0, 0, inner])], f'{name}/v')

    def per_head(t: str, label: str) -> str:
        t = g.node('Reshape', [t, g.const(f'{name}/{label}/groups', [0, 0, groups, 1, state])], f'{name}/{label}/Reshape')
        shape = g.node('Concat', [g.node('Slice', [g.node('Shape', [t], f'{name}/{label}/Shape'), g.const(f'{name}/{label}/s0', [0]), g.const(f'{name}/{label}/s1', [2])], f'{name}/{label}/Slice'), g.const(f'{name}/{label}/rest', [groups, heads // groups, state])], f'{name}/{label}/Concat', axis=0)
        t = g.node('Expand', [t, shape], f'{name}/{label}/Expand')
        return g.node('Reshape', [t, g.const(f'{name}/{label}/flat', [0, 0, heads * state])], f'{name}/{label}/Flat')

    q, k = per_head(c, 'C'), per_head(b, 'B')
    y = f'{name}/scan/output_0'
    g.node(
        'LinearAttention',
        [q, k, v, f'past_key_values.{i}.recurrent_state', decay],
        f'{name}/scan',
        outputs=[y, f'present.{i}.recurrent_state'],
        domain=MS,
        q_num_heads=heads,
        kv_num_heads=heads,
        update_rule='gated',
        scale=1.0,
    )
    d = W(f'{p}.D').repeat_interleave(head_dim)
    y = g.node('Add', [y, g.node('Mul', [xs, g.init(f'{p}.D_expanded', d.numpy(), F32)], f'{name}/D/Mul')], f'{name}/D/Add')

    # Gated RMSNorm in groups: (y · SiLU(z)), normalised per group of inner / n_groups, × weight.
    z = g.node('Cast', [z], f'{name}/Cast_z', to=F32)
    y = g.node('Mul', [y, g.node('Mul', [z, g.node('Sigmoid', [z], f'{name}/gate/Sigmoid')], f'{name}/gate/Silu')], f'{name}/gate/Mul')
    size = inner // groups
    y = g.node('Reshape', [y, g.const(f'{name}/norm/groups', [0, 0, groups, size])], f'{name}/norm/Reshape')
    y = g.node('SimplifiedLayerNormalization', [y, g.init(f'{name}/norm/ones', np.ones(size, np.float32), F32)], f'{name}/norm', axis=-1, epsilon=cfg['layer_norm_epsilon'], stash_type=1)
    y = g.node('Reshape', [y, g.const(f'{name}/norm/flat', [0, 0, inner])], f'{name}/norm/Flat')
    y = g.node('Mul', [y, g.init(f'{p}.norm.weight', W(f'{p}.norm.weight').numpy(), F32)], f'{name}/norm/Mul')
    y = g.node('Cast', [y], f'{name}/Cast_out', to=F16)
    return matmul_nbits(g, y, W(f'{p}.out_proj.weight'), f'{name}/out_proj/MatMul')


def attention(g: Graph, W, cfg: dict, i: int, x: str, mask: dict) -> str:
    p = f'backbone.layers.{i}.mixer'
    name = f'/model/layers.{i}/attn'
    q = matmul_nbits(g, x, W(f'{p}.q_proj.weight'), f'{name}/q_proj/MatMul')
    k = matmul_nbits(g, x, W(f'{p}.k_proj.weight'), f'{name}/k_proj/MatMul')
    v = matmul_nbits(g, x, W(f'{p}.v_proj.weight'), f'{name}/v_proj/MatMul')
    out = f'{name}/GroupQueryAttention/output_0'
    g.node(
        'GroupQueryAttention',
        [q, k, v, f'past_key_values.{i}.key', f'past_key_values.{i}.value', mask['seqlens'], mask['total']],
        f'{name}/GroupQueryAttention',
        outputs=[out, f'present.{i}.key', f'present.{i}.value'],
        domain=MS,
        num_heads=cfg['num_attention_heads'],
        kv_num_heads=cfg['num_key_value_heads'],
        scale=1 / math.sqrt(cfg['head_dim']),
        local_window_size=-1,
    )
    return matmul_nbits(g, out, W(f'{p}.o_proj.weight'), f'{name}/o_proj/MatMul')


def relu2_mlp(g: Graph, x: str, up: torch.Tensor, down: torch.Tensor, name: str) -> str:
    u = g.node('Relu', [matmul_nbits(g, x, up, f'{name}/up_proj/MatMul')], f'{name}/Relu')
    return matmul_nbits(g, g.node('Mul', [u, u], f'{name}/Square'), down, f'{name}/down_proj/MatMul')


def moe(g: Graph, W, cfg: dict, i: int, x: str) -> str:
    p = f'backbone.layers.{i}.mixer'
    name = f'/model/layers.{i}/moe'
    experts, k = cfg['n_routed_experts'], cfg['num_experts_per_tok']
    probs = sigmoid_router(g, x, W(f'{p}.gate.weight'), W(f'{p}.gate.e_score_correction_bias'), k, f'{name}/router')

    bits = Settings.bits
    fc1_q, fc1_s, fc2_q, fc2_s = [], [], [], []
    for e in range(experts):
        q, s = qmoe_weights(W(f'{p}.experts.{e}.up_proj.weight'), bits, EXPERT_BLOCK)
        fc1_q.append(q)
        fc1_s.append(s)
        q, s = qmoe_weights(W(f'{p}.experts.{e}.down_proj.weight'), bits, EXPERT_BLOCK)
        fc2_q.append(q)
        fc2_s.append(s)
    inter, hidden = fc1_q[0].shape[0], fc2_q[0].shape[0]
    names = [
        g.init(f'{p}.experts.gate_up_proj.qweight', np.stack(fc1_q), U8),
        g.init(f'{p}.experts.gate_up_proj.scales', np.stack(fc1_s), F16),
        g.init(f'{p}.experts.gate_up_proj.bias', np.zeros((experts, inter), np.float16), F16),
        g.init(f'{p}.experts.down_proj.qweight', np.stack(fc2_q), U8),
        g.init(f'{p}.experts.down_proj.scales', np.stack(fc2_s), F16),
        g.init(f'{p}.experts.down_proj.bias', np.zeros((experts, hidden), np.float16), F16),
    ]
    routed = g.node(
        'QMoE',
        [x, probs, *names],
        f'{name}/QMoE',
        domain=MS,
        # Squared ReLU: in Quick Inference's ONNX Runtime build (github.com/mdanyalmalik/onnxruntime-web-kernels), as
        # ONNX Runtime's CUDA MoE names it; stock ONNX Runtime Web's WebGPU QMoE only has SwiGLU.
        activation_type='relu2',
        k=k,
        normalize_routing_weights=1,
        expert_weight_bits=bits,
        block_size=EXPERT_BLOCK,
        use_sparse_mixer=0,
    )
    routed = g.node('Mul', [routed, g.const(f'{name}/scale', cfg['routed_scaling_factor'], F16)], f'{name}/scale')
    sp = f'{p}.shared_experts'
    shared = relu2_mlp(g, x, W(f'{sp}.up_proj.weight'), W(f'{sp}.down_proj.weight'), f'/model/layers.{i}/shared_expert')
    return g.node('Add', [routed, shared], f'{name}/Add')


def build(layers: int, out: Path):
    cfg = json.loads((SRC / 'config.json').read_text())
    pattern = cfg['hybrid_override_pattern']
    full = layers == cfg['num_hidden_layers']
    W = Weights(SRC, SOURCE, REVISION, stream=full)
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    eps = cfg['layer_norm_epsilon']
    conv_dim = cfg['mamba_num_heads'] * cfg['mamba_head_dim'] + 2 * cfg['n_groups'] * cfg['ssm_state_size']

    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', cfg['vocab_size']])]
    for i in range(layers):
        if pattern[i] == '*':
            kv = cfg['num_key_value_heads'], cfg['head_dim']
            for kind in ('key', 'value'):
                g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', kv[0], 'past_sequence_length', kv[1]]))
                g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', kv[0], 'total_sequence_length', kv[1]]))
        elif pattern[i] == 'M':
            conv = ['batch_size', conv_dim, cfg['conv_kernel'] - 1]
            rec = ['batch_size', cfg['mamba_num_heads'], cfg['ssm_state_size'], cfg['mamba_head_dim']]
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.conv_state', F16, conv))
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.recurrent_state', F32, rec))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.conv_state', F16, conv))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.recurrent_state', F32, rec))
    mask = attention_lengths(g)

    emb = g.init('backbone.embeddings.weight', W('backbone.embeddings.weight').numpy().astype(np.float16), F16)
    residual = g.node('Gather', [emb, 'input_ids'], '/model/embed_tokens/Gather')
    h = rms_norm(g, residual, W('backbone.layers.0.norm.weight'), '/model/layers.0/norm', eps)
    for i in range(layers):
        kind = pattern[i]
        print(f'layer {i} {kind}', flush=True)
        if kind == 'M':
            m = mamba(g, W, cfg, i, h)
        elif kind == '*':
            m = attention(g, W, cfg, i, h, mask)
        elif kind == 'E':
            m = moe(g, W, cfg, i, h)
        else:
            p = f'backbone.layers.{i}.mixer'
            m = relu2_mlp(g, h, W(f'{p}.up_proj.weight'), W(f'{p}.down_proj.weight'), f'/model/layers.{i}/mlp')
        if i + 1 < layers:
            h, residual = skip_rms_norm(g, m, residual, W(f'backbone.layers.{i + 1}.norm.weight'), f'/model/layers.{i + 1}/norm', eps)
        else:
            h, _ = skip_rms_norm(g, m, residual, W('backbone.norm_f.weight'), '/model/norm_f', eps)
    logits = matmul_nbits(g, h, W('lm_head.weight'), '/lm_head/MatMul')
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()

    graph = helper.make_graph(g.nodes, 'Nemotron-3-Nano', g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)


def finish(out: Path):
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'generation_config.json', 'config.json'):
        shutil.copy(SRC / name, out / name)
    shutil.copy(HERE / 'licences' / 'NVIDIA_NEMOTRON_OPEN_MODEL_LICENSE.md', out / 'LICENSE.md')
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = f"""---
license: other
license_name: nvidia-nemotron-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
base_model:
- {SOURCE}
tags:
- onnx
- webgpu
- quick-inference
---

# {REPO}

[{SOURCE}](https://huggingface.co/{SOURCE}) converted to ONNX for running in a web browser with ONNX Runtime Web's
WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to an ONNX decoder in ONNX Runtime GenAI's layout by Quick Inference's own exporter: Mamba-2 blocks as
  `CausalConvWithState` plus `LinearAttention` (gated rule, float32 state), attention blocks as `GroupQueryAttention`,
  expert blocks as `QMoE`.
- The experts use QMoE's `relu2` activation (squared ReLU, as ONNX Runtime's CUDA MoE names it). ONNX Runtime Web's
  WebGPU QMoE gains it in Quick Inference's build (https://github.com/mdanyalmalik/onnxruntime-web-kernels); stock
  ONNX Runtime Web only runs SwiGLU there, so these files need that build.
- Linear layers are quantised to 4 bits (round to nearest, block size 32), the routed experts to 4 bits (block size
  64). Embeddings, activations and the attention KV cache are float16.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

Licensed by NVIDIA Corporation under the NVIDIA Nemotron Open Model License (LICENSE.md), as the original.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=None, help='only the first N blocks (for testing)')
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8])
    args = parser.parse_args()
    Settings.bits = args.bits
    cfg = json.loads((SRC / 'config.json').read_text())
    build(args.layers or cfg['num_hidden_layers'], args.out)
    finish(args.out)
