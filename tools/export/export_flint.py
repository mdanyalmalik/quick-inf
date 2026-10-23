"""Exports Recursive's Flint (FlintForCausalLM, Apache 2.0) to an ONNX decoder in ONNX Runtime GenAI's layout, which
ONNX Runtime GenAI's own builder doesn't support. Weights stream from the checkpoint one tensor at a time into
model.onnx.data, and with the full model each shard is downloaded when needed and deleted when done, so the export needs
a few GB of memory and disk, not the checkpoint's 80 GB.

Flint (its SGLang model code, flint_sglang/model.py) is Qwen3-MoE's decoder with:
- attention: 32 query heads and 4 KV heads of 128, per-head Q and K RMSNorm, scale 1/sqrt(128), and learned sinks (a
  per-head logit added to the softmax denominator only): GroupQueryAttention's head_sink. Sliding-window layers (513
  tokens including the current one: local_window_size 513, which counts the current token) use RoPE (theta 10000,
  halves rotated, not interleaved); full-attention layers have no positional encoding (do_rotary 0).
- dense SwiGLU MLPs on layers 0 and 1; elsewhere 256 routed experts, 6 per token, chosen by sigmoid score and weighted
  by those scores renormalised to sum to one (QMoE gets log(score) for the chosen experts: onnx_export.sigmoid_router,
  with a zero bias), plus an ungated shared expert added to their output.

Linear layers: 4-bit MatMulNBits (block 32); experts: 4-bit QMoE; the output layer 8-bit (Flint was validated with a
float32 one); embeddings, activations and the KV cache float16, except:
- the residual stream and the RMSNorms run in float32 (their normalised outputs go to float16): Flint has massive
  activations (the residual reaches about 9,000 in one channel from layer 46 on), and ONNX Runtime WebGPU's float16
  SkipSimplifiedLayerNormalization overflowed there on some decode steps (one infinite output from finite inputs);
- the experts of layers 44-48 see their input divided by EXPERT_SCALE, with QMoE's SwiGLU activation_alpha set to it
  (QMoE computes g * sigmoid(alpha * g) * u, so the intermediate is silu(g) * u / EXPERT_SCALE^2) and their output
  multiplied by EXPERT_SCALE^2: the same values, but layer 46's expert intermediates (outputs about 7,000) overflowed
  float16 inside QMoE on WebGPU otherwise.

    tools/export/.venv/bin/python tools/export/export_flint.py [--layers N] [--bits 4|8] [--out DIR]
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from onnx_export import F16, F32, I64, MS, U8, Graph, Settings, Weights, attention_lengths, matmul_nbits, qmoe_weights, rms_norm, sigmoid_router

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'flint'
REPO = 'Flint-ONNX-WebGPU'
SOURCE = 'RecursiveSI/flint'
REVISION = '2a247be0c0aec895ccb0ac7ca92229ec6df37288'
# Rotary tables cover this many positions (the app's longest context), not the model's 262,144.
ROTARY_POSITIONS = 65536
HEAD_BITS = 8
# Expert inputs scaled down by this (and their output up by its square) in these layers (see the top of this file).
EXPERT_SCALE = 8.0
SCALED_EXPERT_LAYERS = range(44, 49)


def norm32(g: Graph, x: str, w: torch.Tensor, name: str, eps: float) -> str:
    """RMSNorm of a float32 x, as float16."""
    wn = g.init(f'{name}.weight', w.numpy().astype(np.float32), F32)
    out = g.node('SimplifiedLayerNormalization', [x, wn], name, axis=-1, epsilon=eps, stash_type=1)
    return g.node('Cast', [out], f'{name}/Cast', to=F16)


def skip_norm32(g: Graph, x: str, residual: str, w: torch.Tensor, name: str, eps: float):
    """(norm(x + residual) as float16, x + residual) with the float16 block output x added to the float32 residual."""
    wn = g.init(f'{name}.weight', w.numpy().astype(np.float32), F32)
    x32 = g.node('Cast', [x], f'{name}/Cast_x', to=F32)
    outs = [f'{name}/output_0', '', '', f'{name}/output_3']
    g.node('SkipSimplifiedLayerNormalization', [x32, residual, wn], name, outputs=outs, domain=MS, epsilon=eps)
    return g.node('Cast', [outs[0]], f'{name}/Cast', to=F16), outs[3]


def silu_mlp(g: Graph, x: str, W: Weights, p: str, name: str) -> str:
    gt = matmul_nbits(g, x, W(f'{p}.gate_proj.weight'), f'{name}/gate_proj/MatMul')
    u = matmul_nbits(g, x, W(f'{p}.up_proj.weight'), f'{name}/up_proj/MatMul')
    act = g.node('Mul', [gt, g.node('Sigmoid', [gt], f'{name}/act/Sigmoid')], f'{name}/act/Mul')
    return matmul_nbits(g, g.node('Mul', [act, u], f'{name}/Mul'), W(f'{p}.down_proj.weight'), f'{name}/down_proj/MatMul')


def head_norm(g: Graph, x: str, w: torch.Tensor, heads: int, head_dim: int, name: str, eps: float) -> str:
    """RMSNorm of each head's slice of x [B, S, heads * head_dim]."""
    per_head = g.node('Reshape', [x, g.const(f'{name}/shape', [0, 0, heads, head_dim])], f'{name}/Reshape')
    normed = rms_norm(g, per_head, w, name, eps)
    return g.node('Reshape', [normed, g.const(f'{name}/flat', [0, 0, heads * head_dim])], f'{name}/Flatten')


def attention(g: Graph, W: Weights, cfg: dict, i: int, x: str, mask: dict) -> str:
    p = f'model.layers.{i}.self_attn'
    name = f'/model/layers.{i}/attn'
    heads, kv_heads, head_dim, eps = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['rms_norm_eps']
    sliding = cfg['layer_types'][i] == 'sliding_attention'
    rope = i not in set(cfg['no_rope_layer_indices'])

    q = matmul_nbits(g, x, W(f'{p}.q_proj.weight'), f'{name}/q_proj/MatMul')
    k = matmul_nbits(g, x, W(f'{p}.k_proj.weight'), f'{name}/k_proj/MatMul')
    v = matmul_nbits(g, x, W(f'{p}.v_proj.weight'), f'{name}/v_proj/MatMul')
    q = head_norm(g, q, W(f'{p}.q_norm.weight'), heads, head_dim, f'{name}/q_norm', eps)
    k = head_norm(g, k, W(f'{p}.k_norm.weight'), kv_heads, head_dim, f'{name}/k_norm', eps)
    sinks = g.init(f'{p}.sinks', W(f'{p}.softmax_offset').numpy().astype(np.float16), F16)

    past = [f'past_key_values.{i}.key', f'past_key_values.{i}.value']
    present = [f'present.{i}.key', f'present.{i}.value']
    out = f'{name}/GroupQueryAttention/output_0'
    rotary = ['cos_cache', 'sin_cache'] if rope else ['', '']
    g.node(
        'GroupQueryAttention',
        [q, k, v, *past, mask['seqlens'], mask['total'], *rotary, '', '', sinks],
        f'{name}/GroupQueryAttention',
        outputs=[out, *present],
        domain=MS,
        num_heads=heads,
        kv_num_heads=kv_heads,
        scale=1 / math.sqrt(head_dim),
        do_rotary=1 if rope else 0,
        rotary_interleaved=0,
        local_window_size=cfg['sliding_window'] if sliding else -1,
    )
    return matmul_nbits(g, out, W(f'{p}.o_proj.weight'), f'{name}/o_proj/MatMul')


def moe(g: Graph, W: Weights, cfg: dict, i: int, x: str) -> str:
    p = f'model.layers.{i}.mlp'
    name = f'/model/layers.{i}/moe'
    experts, k = cfg['num_experts'], cfg['num_experts_per_tok']
    probs = sigmoid_router(g, x, W(f'{p}.gate.weight'), torch.zeros(experts), k, f'{name}/router')

    # Experts: gate and up rows interleaved (swiglu_fusion=1), each expert quantised like MatMulNBits.
    bits, block = Settings.bits, Settings.block
    fc1_q, fc1_s, fc2_q, fc2_s = [], [], [], []
    for e in range(experts):
        ep = f'{p}.experts.{e}'
        gate_w, up_w = W(f'{ep}.gate_proj.weight'), W(f'{ep}.up_proj.weight')
        q, s = qmoe_weights(torch.stack([gate_w, up_w], dim=1).reshape(-1, gate_w.shape[1]), bits, block)
        fc1_q.append(q)
        fc1_s.append(s)
        q, s = qmoe_weights(W(f'{ep}.down_proj.weight'), bits, block)
        fc2_q.append(q)
        fc2_s.append(s)
    inter2, hidden = fc1_q[0].shape[0], fc2_q[0].shape[0]
    names = [
        g.init(f'{p}.experts.gate_up_proj.qweight', np.stack(fc1_q), U8),
        g.init(f'{p}.experts.gate_up_proj.scales', np.stack(fc1_s), F16),
        g.init(f'{p}.experts.gate_up_proj.bias', np.zeros((experts, inter2), np.float16), F16),
        g.init(f'{p}.experts.down_proj.qweight', np.stack(fc2_q), U8),
        g.init(f'{p}.experts.down_proj.scales', np.stack(fc2_s), F16),
        g.init(f'{p}.experts.down_proj.bias', np.zeros((experts, hidden), np.float16), F16),
    ]
    scale = EXPERT_SCALE if i in SCALED_EXPERT_LAYERS else 1.0
    qx = g.node('Mul', [x, g.const(f'{name}/in_scale', 1 / scale, F16)], f'{name}/in_scale') if scale != 1 else x
    routed = g.node(
        'QMoE',
        [qx, probs, *names],
        f'{name}/QMoE',
        domain=MS,
        activation_type='swiglu',
        swiglu_fusion=1,
        activation_alpha=scale,
        activation_beta=0.0,
        k=k,
        normalize_routing_weights=1,
        expert_weight_bits=bits,
        block_size=block,
        use_sparse_mixer=0,
    )
    if scale != 1:
        routed = g.node('Mul', [routed, g.const(f'{name}/out_scale', scale * scale, F16)], f'{name}/out_scale')
    shared = silu_mlp(g, x, W, f'{p}.shared_expert', f'/model/layers.{i}/shared_expert')
    return g.node('Add', [routed, shared], f'{name}/Add')


def rotary_caches(g: Graph, cfg: dict):
    dim = cfg['head_dim']
    inv = 1.0 / (cfg['rope_theta'] ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = np.outer(np.arange(ROTARY_POSITIONS, dtype=np.float64), inv)
    g.init('cos_cache', np.cos(freqs).astype(np.float16), F16)
    g.init('sin_cache', np.sin(freqs).astype(np.float16), F16)


def build(layers: int, out: Path):
    cfg = json.loads((SRC / 'config.json').read_text())
    full = layers == cfg['num_hidden_layers']
    W = Weights(SRC, SOURCE, REVISION, stream=full)
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    kv_heads, head_dim, eps = cfg['num_key_value_heads'], cfg['head_dim'], cfg['rms_norm_eps']

    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', cfg['vocab_size']])]
    for i in range(layers):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', kv_heads, 'past_sequence_length', head_dim]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', kv_heads, 'total_sequence_length', head_dim]))

    mask = attention_lengths(g)
    rotary_caches(g, cfg)

    emb = g.init('model.embed_tokens.weight', W('model.embed_tokens.weight').numpy().astype(np.float16), F16)
    embedded = g.node('Gather', [emb, 'input_ids'], '/model/embed_tokens/Gather')
    residual = g.node('Cast', [embedded], '/model/embed_tokens/Cast', to=F32)
    h = norm32(g, residual, W('model.layers.0.input_layernorm.weight'), '/model/layers.0/input_layernorm', eps)
    dense = set(cfg['mlp_only_layers'])
    for i in range(layers):
        print(f'layer {i}', flush=True)
        a = attention(g, W, cfg, i, h, mask)
        h, residual = skip_norm32(g, a, residual, W(f'model.layers.{i}.post_attention_layernorm.weight'), f'/model/layers.{i}/post_attention_layernorm', eps)
        f = silu_mlp(g, h, W, f'model.layers.{i}.mlp', f'/model/layers.{i}/mlp') if i in dense else moe(g, W, cfg, i, h)
        if i + 1 < layers:
            h, residual = skip_norm32(g, f, residual, W(f'model.layers.{i + 1}.input_layernorm.weight'), f'/model/layers.{i + 1}/input_layernorm', eps)
        else:
            h, _ = skip_norm32(g, f, residual, W('model.norm.weight'), '/model/norm', eps)
    bits = Settings.bits
    Settings.bits = max(bits, HEAD_BITS)
    logits = matmul_nbits(g, h, W('lm_head.weight'), '/lm_head/MatMul')
    Settings.bits = bits
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()

    graph = helper.make_graph(g.nodes, 'Flint', g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)
    return cfg


def genai_config(cfg: dict, layers: int, out: Path):
    gen = json.loads((SRC / 'generation_config.json').read_text())
    config = {
        'model': {
            'bos_token_id': gen.get('bos_token_id'),
            'context_length': ROTARY_POSITIONS,
            'decoder': {
                'session_options': {'log_id': 'onnxruntime-genai', 'provider_options': [{'webgpu': {}}]},
                'filename': 'model.onnx',
                'head_size': cfg['head_dim'],
                'hidden_size': cfg['hidden_size'],
                'inputs': {
                    'input_ids': 'input_ids',
                    'attention_mask': 'attention_mask',
                    'past_key_names': 'past_key_values.%d.key',
                    'past_value_names': 'past_key_values.%d.value',
                },
                'outputs': {'logits': 'logits', 'present_key_names': 'present.%d.key', 'present_value_names': 'present.%d.value'},
                'num_attention_heads': cfg['num_attention_heads'],
                'num_hidden_layers': layers,
                'num_key_value_heads': cfg['num_key_value_heads'],
            },
            'eos_token_id': cfg['eos_token_id'],
            'pad_token_id': cfg['pad_token_id'],
            'type': 'flint',
            'vocab_size': cfg['vocab_size'],
        },
        'search': {
            'do_sample': True,
            'max_length': ROTARY_POSITIONS,
            'temperature': gen.get('temperature', 1.0),
            'top_p': gen.get('top_p', 0.95),
            'top_k': gen.get('top_k', 20),
        },
    }
    (out / 'genai_config.json').write_text(json.dumps(config, indent=4))


def finish(out: Path):
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'config.json', 'LICENSE'):
        shutil.copy(SRC / name, out / name)
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = """---
license: apache-2.0
base_model:
- RecursiveSI/flint
tags:
- onnx
- webgpu
- quick-inference
---

# Flint-ONNX-WebGPU

Recursive's [Flint](https://huggingface.co/RecursiveSI/flint) (39.8B parameters, about 2.5B active per token)
converted to ONNX for running in a web browser with ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to an ONNX decoder in ONNX Runtime GenAI's layout (`input_ids`, `attention_mask`, `past_key_values.*` →
  `logits`, `present.*`) by Quick Inference's own exporter (`tools/export/export_flint.py` in
  [Quick Inference](https://github.com/mdanyalmalik/quick-inf)), following Flint's SGLang model code.
- Attention is `GroupQueryAttention` with Flint's learned sinks (`head_sink`), its 513-token sliding window on the
  sliding layers, and rotary embeddings only where Flint uses them. The routed experts run as one `QMoE` per layer:
  sigmoid scores, the top 6 renormalised; the shared expert is added to their output.
- Linear layers and experts are quantised to 4 bits (round to nearest, symmetric, block size 32), the output layer to
  8 bits. Embeddings, activations and the KV cache are float16; the residual stream and the RMSNorms are float32, and
  the last five layers' experts run on an input scaled down by 8 (with the activation and output compensated
  exactly), because Flint's massive activations overflow float16 otherwise. Rotary tables cover 65,536 positions.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

Apache 2.0, as the original (LICENSE).
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=None, help='only the first N layers (for testing)')
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8])
    args = parser.parse_args()
    Settings.bits = args.bits
    cfg = json.loads((SRC / 'config.json').read_text())
    layers = args.layers or cfg['num_hidden_layers']
    build(layers, args.out)
    genai_config(cfg, layers, args.out)
    finish(args.out)
