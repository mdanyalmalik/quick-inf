"""Exports GLM-4.7 Flash (Glm4MoeLiteForCausalLM: multi-head latent attention plus a mixture of experts) to an ONNX
decoder in ONNX Runtime GenAI's layout, which ONNX Runtime GenAI's own builder doesn't support. Weights stream from the
checkpoint one tensor at a time into model.onnx.data, so the export needs a few GB of memory, not the model's 60 GB.

Attention is the "absorbed" form of multi-head latent attention (as DeepSeek serves it): each layer caches the
normalised 512-wide latent and the 64-wide rotary key, one 576-wide key/value head shared by the 20 query heads, about
108 KB per token in all. The key projection of kv_b_proj is folded into the query projection and its value projection
into o_proj, so attention runs as GroupQueryAttention with one KV head:

    q (per head) = [q_rope (64) | W_k^T q_nope (512)]       key = value = [k_rope (64) | latent (512)]
    output (per head) = attention over the latents (the value's first 64 columns are dropped), then o_proj ∘ W_v

The router picks experts with sigmoid scores plus a correction bias but weights them by the plain scores. QMoE applies
a softmax to its router input, so it gets log(score) for the chosen experts and -10000 elsewhere: the softmax over that
is each chosen score divided by their sum, as GLM's normalised top-k weights; the routed scaling factor multiplies
QMoE's output.

Linear layers: 4-bit MatMulNBits (round to nearest, block size 32, ONNX Runtime's signed-scale encoding); experts: 4-bit
QMoE; embeddings float16; activations and KV cache float16. The multi-token-prediction layer (47) is left out.

    tools/export/.venv/bin/python tools/export/export_glm.py [--layers N] [--out DIR]
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

from onnx_export import F16, F32, I64, MS, U8, Graph, Settings, Weights, attention_lengths, matmul_nbits, qmoe_weights, rms_norm, sigmoid_router, skip_rms_norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'GLM-4.7-Flash'
REPO = 'GLM-4.7-Flash-ONNX-WebGPU'
SOURCE = 'zai-org/GLM-4.7-Flash'
REVISION = '7dd20894a642a0aa287e9827cb1a1f7f91386b67'


def silu_mlp(g: Graph, x: str, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, name: str) -> str:
    gt = matmul_nbits(g, x, gate, f'{name}/gate_proj/MatMul')
    u = matmul_nbits(g, x, up, f'{name}/up_proj/MatMul')
    act = g.node('Mul', [gt, g.node('Sigmoid', [gt], f'{name}/act/Sigmoid')], f'{name}/act/Mul')
    return matmul_nbits(g, g.node('Mul', [act, u], f'{name}/Mul'), down, f'{name}/down_proj/MatMul')


def attention(g: Graph, W: Weights, cfg: dict, i: int, x: str, mask: dict) -> str:
    p = f'model.layers.{i}.self_attn'
    name = f'/model/layers.{i}/attn'
    heads, nope, rope, vdim, rank = (cfg[k] for k in ('num_attention_heads', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'kv_lora_rank'))
    eps = cfg['rms_norm_eps']
    width = rope + rank

    # Queries: [rope part | nope part folded through kv_b_proj's key projection], per head.
    q_a = matmul_nbits(g, x, W(f'{p}.q_a_proj.weight'), f'{name}/q_a_proj/MatMul')
    q_a = rms_norm(g, q_a, W(f'{p}.q_a_layernorm.weight'), f'{name}/q_a_layernorm', eps)
    q_b = W(f'{p}.q_b_proj.weight').view(heads, nope + rope, -1)  # [H, 256, 768]
    kv_b = W(f'{p}.kv_b_proj.weight').view(heads, nope + vdim, rank)  # [H, 448, 512]
    w_k, w_v = kv_b[:, :nope], kv_b[:, nope:]  # [H, 192, 512], [H, 256, 512]
    q_lat = torch.einsum('hdr,hdc->hrc', w_k, q_b[:, :nope])  # [H, 512, 768]
    q_w = torch.cat([q_b[:, nope:], q_lat], dim=1).reshape(heads * width, -1)  # [H*576, 768]
    q = matmul_nbits(g, q_a, q_w, f'{name}/q_proj/MatMul')

    # Key = value = [rotary key | normalised latent], one head.
    kv_a = matmul_nbits(g, x, W(f'{p}.kv_a_proj_with_mqa.weight'), f'{name}/kv_a_proj/MatMul')
    latent, k_rope = g.node('Split', [kv_a, g.const(f'{name}/split', [rank, rope])], f'{name}/kv_split', outputs=2, axis=-1)
    latent = rms_norm(g, latent, W(f'{p}.kv_a_layernorm.weight'), f'{name}/kv_a_layernorm', eps)
    kv = g.node('Concat', [k_rope, latent], f'{name}/kv_concat', axis=-1)

    past = [f'past_key_values.{i}.key', f'past_key_values.{i}.value']
    present = [f'present.{i}.key', f'present.{i}.value']
    out = f'{name}/GroupQueryAttention/output_0'
    g.node(
        'GroupQueryAttention',
        [q, kv, kv, *past, mask['seqlens'], mask['total'], 'cos_cache', 'sin_cache'],
        f'{name}/GroupQueryAttention',
        outputs=[out, *present],
        domain=MS,
        num_heads=heads,
        kv_num_heads=1,
        scale=1 / math.sqrt(nope + rope),
        do_rotary=1,
        rotary_interleaved=1,
        local_window_size=-1,
    )
    # Drop each head's first 64 (rotary) columns, then o_proj folded with kv_b_proj's value projection.
    per_head = g.node('Reshape', [out, g.const(f'{name}/heads_shape', [0, 0, heads, width])], f'{name}/heads')
    lat = g.node('Slice', [per_head, g.const(f'{name}/starts', [rope]), g.const(f'{name}/ends', [width]), g.const(f'{name}/axes', [3])], f'{name}/latents')
    lat = g.node('Reshape', [lat, g.const(f'{name}/flat_shape', [0, 0, heads * rank])], f'{name}/flat')
    o = W(f'{p}.o_proj.weight').view(-1, heads, vdim)  # [2048, H, 256]
    o_w = torch.einsum('ohv,hvr->ohr', o, w_v).reshape(o.shape[0], heads * rank)  # [2048, H*512]
    return matmul_nbits(g, lat, o_w, f'{name}/o_proj/MatMul')


def moe(g: Graph, W: Weights, cfg: dict, i: int, x: str) -> str:
    p = f'model.layers.{i}.mlp'
    name = f'/model/layers.{i}/moe'
    experts, k = cfg['n_routed_experts'], cfg['num_experts_per_tok']

    probs = sigmoid_router(g, x, W(f'{p}.gate.weight'), W(f'{p}.gate.e_score_correction_bias'), k, f'{name}/router')

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
    routed = g.node(
        'QMoE',
        [x, probs, *names],
        f'{name}/QMoE',
        domain=MS,
        activation_type='swiglu',
        swiglu_fusion=1,
        activation_alpha=1.0,
        activation_beta=0.0,
        k=k,
        normalize_routing_weights=1,
        expert_weight_bits=bits,
        block_size=block,
        use_sparse_mixer=0,
    )
    routed = g.node('Mul', [routed, g.const(f'{name}/scale', cfg['routed_scaling_factor'], F16)], f'{name}/scale')
    sp = f'{p}.shared_experts'
    shared = silu_mlp(g, x, W(f'{sp}.gate_proj.weight'), W(f'{sp}.up_proj.weight'), W(f'{sp}.down_proj.weight'), f'/model/layers.{i}/shared_expert')
    return g.node('Add', [routed, shared], f'{name}/Add')


def rotary_caches(g: Graph, cfg: dict):
    dim = cfg['qk_rope_head_dim']
    theta = cfg.get('rope_theta') or cfg['rope_parameters']['rope_theta']
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = np.outer(np.arange(cfg['max_position_embeddings'], dtype=np.float64), inv)
    g.init('cos_cache', np.cos(freqs).astype(np.float16), F16)
    g.init('sin_cache', np.sin(freqs).astype(np.float16), F16)


def build(layers: int, out: Path):
    cfg = json.loads((SRC / 'config.json').read_text())
    full = layers == cfg['num_hidden_layers']
    W = Weights(SRC, SOURCE, REVISION, stream=full, skip_prefix=f"model.layers.{cfg['num_hidden_layers']}.")
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    width = cfg['qk_rope_head_dim'] + cfg['kv_lora_rank']
    eps = cfg['rms_norm_eps']

    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', cfg['vocab_size']])]
    for i in range(layers):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', 1, 'past_sequence_length', width]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', 1, 'total_sequence_length', width]))

    mask = attention_lengths(g)
    rotary_caches(g, cfg)

    emb = g.init('model.embed_tokens.weight', W('model.embed_tokens.weight').numpy().astype(np.float16), F16)
    residual = g.node('Gather', [emb, 'input_ids'], '/model/embed_tokens/Gather')
    h = rms_norm(g, residual, W('model.layers.0.input_layernorm.weight'), '/model/layers.0/input_layernorm', eps)
    for i in range(layers):
        print(f'layer {i}', flush=True)
        a = attention(g, W, cfg, i, h, mask)
        h, residual = skip_rms_norm(g, a, residual, W(f'model.layers.{i}.post_attention_layernorm.weight'), f'/model/layers.{i}/post_attention_layernorm', eps)
        if i < cfg['first_k_dense_replace']:
            p = f'model.layers.{i}.mlp'
            f = silu_mlp(g, h, W(f'{p}.gate_proj.weight'), W(f'{p}.up_proj.weight'), W(f'{p}.down_proj.weight'), f'/model/layers.{i}/mlp')
        else:
            f = moe(g, W, cfg, i, h)
        if i + 1 < layers:
            h, residual = skip_rms_norm(g, f, residual, W(f'model.layers.{i + 1}.input_layernorm.weight'), f'/model/layers.{i + 1}/input_layernorm', eps)
        else:
            h, _ = skip_rms_norm(g, f, residual, W('model.norm.weight'), '/model/norm', eps)
    logits = matmul_nbits(g, h, W('lm_head.weight'), '/lm_head/MatMul')
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()

    graph = helper.make_graph(g.nodes, 'GLM-4.7-Flash', g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)
    return cfg


def genai_config(cfg: dict, layers: int, out: Path):
    width = cfg['qk_rope_head_dim'] + cfg['kv_lora_rank']
    gen = json.loads((SRC / 'generation_config.json').read_text())
    config = {
        'model': {
            'bos_token_id': gen.get('bos_token_id'),
            'context_length': cfg['max_position_embeddings'],
            'decoder': {
                'session_options': {'log_id': 'onnxruntime-genai', 'provider_options': [{'webgpu': {}}]},
                'filename': 'model.onnx',
                'head_size': width,
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
                'num_key_value_heads': 1,
            },
            'eos_token_id': cfg['eos_token_id'],
            'pad_token_id': cfg['pad_token_id'],
            'type': 'glm4_moe_lite',
            'vocab_size': cfg['vocab_size'],
        },
        'search': {'do_sample': True, 'max_length': cfg['max_position_embeddings'], 'temperature': gen.get('temperature', 1.0), 'top_p': gen.get('top_p', 0.95)},
    }
    (out / 'genai_config.json').write_text(json.dumps(config, indent=4))


def finish(out: Path):
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'config.json'):
        shutil.copy(SRC / name, out / name)
    # The original repository declares MIT in its model card but has no licence file.
    (out / 'LICENSE').write_text(MIT_NOTE + (HERE / 'licences' / 'MIT.txt').read_text())
    (out / 'README.md').write_text(MODEL_CARD)


MIT_NOTE = (
    'GLM-4.7-Flash is released by Z.ai (zai-org) under the MIT License, as stated in its model card at '
    'https://huggingface.co/zai-org/GLM-4.7-Flash. The license text follows.\n\n'
)

MODEL_CARD = """---
license: mit
base_model:
- zai-org/GLM-4.7-Flash
tags:
- onnx
- webgpu
- quick-inference
---

# GLM-4.7-Flash-ONNX-WebGPU

[zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) converted to ONNX for running in a web browser
with ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to an ONNX decoder in ONNX Runtime GenAI's layout (`input_ids`, `attention_mask`, `past_key_values.*` →
  `logits`, `present.*`) by Quick Inference's own exporter.
- Attention is the absorbed form of multi-head latent attention: the KV cache holds the 512-wide normalised latent and
  the 64-wide rotary key (one shared head, 576 wide); the key and value projections of `kv_b_proj` are folded into the
  query projection and `o_proj`.
- Linear layers are quantised to 4 bits (round to nearest, block size 32); the routed experts to 4 bits for the QMoE
  operator. Embeddings, activations and the KV cache are float16.
- The multi-token-prediction layer is left out.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

MIT, as the original (LICENSE).
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
