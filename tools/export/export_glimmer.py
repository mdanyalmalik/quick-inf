"""Exports the text model of Meta's Muse Glimmer 30B (MuseGlimmerForConditionalGeneration, Apache 2.0) to an ONNX decoder
in ONNX Runtime GenAI's layout, from Red Hat's GPTQ 4-bit checkpoint (RedHatAI/Muse-Glimmer-30B-INT4: weights
symmetric 4-bit in blocks of 128, calibrated on 512 chat samples), whose packed weights are MatMulNBits' own layout:
each int32 holds 8 codes (code + 8, lowest nibble first), so its bytes are MatMulNBits' bytes and the codes are copied
as they are, with the default zero point 8. The vision tower is left out.

Muse Glimmer's text model (transformers' modeling_muse_glimmer.py), per layer:
- attention: 32 query heads and 2 KV heads of 128. Q and K are RMS-normalised per head (no weights), Q then scaled by
  qk_scale_factor (folded into GroupQueryAttention's scale with 1/sqrt(128)). Sliding layers (3 of every 4) use RoPE
  (theta 500,000, halves rotated) and a 2,048-token window including the current token; the others have no positional
  encoding. The attention output is gated by sigmoid(gate_proj(x)) before o_proj.
- sandwich norms: input_layernorm before attention, post_attention_layernorm on its output, pre_feedforward_layernorm
  before the SwiGLU MLP and post_feedforward_layernorm on its output; all centred RMSNorms (scale 1 + w), the post ones
  with eps 1e-8.
- the embeddings are RMS-normalised (no weights) after the lookup; the final norm has a plain weight; logits are
  T * tanh(logits * output_multiplier / T) with T = final_logit_softcapping. The graph's `logits` are before that cap:
  the app applies it to the row it samples from (in the graph it failed on WebGPU and kept the output layer from
  computing only the last position).

The residual stream and norms run in float32 (their outputs go to float16), as Gemma 3's do. The token embeddings
(a 4-bit GatherBlockQuantized, block 32) and the output layer (4-bit MatMulNBits, block 32) are rounded to nearest
here: Red Hat left them in bfloat16. Activations and the KV cache are float16.

    tools/export/.venv/bin/python tools/export/export_glimmer.py [--layers N] [--out DIR]
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

from onnx_export import F16, F32, I64, MS, U8, Graph, Settings, Weights, attention_lengths, matmul_nbits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'Muse-Glimmer-30B-INT4'
# Meta's own tokenizer files, chat template, licence and usage policy (Red Hat's template is an older one).
META = HERE / '.cache' / 'hf' / 'Muse-Glimmer-30B-meta'
REPO = 'Muse-Glimmer-30B-ONNX-WebGPU'
SOURCE = 'RedHatAI/Muse-Glimmer-30B-INT4'
REVISION = '397b1ee1eb83f09b9675e66e49d5ec3a522c509a'
ORIGINAL = 'meta-models/Muse-Glimmer-30B'
PREFIX = 'model.language_model.'
# Rotary tables cover this many positions (the app's longest context), not the model's 131,072.
ROTARY_POSITIONS = 65536
GPTQ_BLOCK = 128
# MatMulNBits' block size. 32 would split GPTQ's blocks of 128 in four sharing a scale (the same values), at the cost of
# 1.2 GB more scales (15.7 GB). 128 keeps 14.5 GB: our ONNX Runtime build's fast prompt kernel takes any multiple of 32
# (onnxruntime-web-kernels v1.30.0-qi.3; stock 1.30 takes only 32).
MATMUL_BLOCK = GPTQ_BLOCK


def gptq_matmul(g: Graph, x: str, W: Weights, p: str, name: str) -> str:
    """x @ w.T for Red Hat's packed 4-bit weight `p` (weight_packed int32 [N, K/8], weight_scale [N, K/128])."""
    packed = W(f'{p}.weight_packed', native=True)
    scale = W(f'{p}.weight_scale', native=True)
    n, k = (int(v) for v in W(f'{p}.weight_shape', native=True).tolist())
    assert packed.dtype == torch.int32 and tuple(packed.shape) == (n, k // 8) and tuple(scale.shape) == (n, k // GPTQ_BLOCK)
    # Blocks of MATMUL_BLOCK along K hold the same bytes in the same order, each with its GPTQ block's scale.
    codes = packed.numpy().view(np.uint8).reshape(n, k // MATMUL_BLOCK, MATMUL_BLOCK // 2)
    scales = np.repeat(scale.float().numpy().astype(np.float16), GPTQ_BLOCK // MATMUL_BLOCK, axis=1)
    qn = g.init(f'{name}.weight_Q4G{MATMUL_BLOCK}', codes, U8)
    sn = g.init(f'{name}.weight_scale', scales.reshape(-1), F16)
    return g.node('MatMulNBits', [x, qn, sn], name, domain=MS, K=k, N=n, bits=4, block_size=MATMUL_BLOCK, accuracy_level=0)


def norm32(g: Graph, x: str, scale: np.ndarray, name: str, eps: float) -> str:
    """RMSNorm of a float32 x with this scale, in float32."""
    wn = g.init(f'{name}.weight', scale.astype(np.float32), F32)
    return g.node('SimplifiedLayerNormalization', [x, wn], name, axis=-1, epsilon=eps, stash_type=1)


def centred(W: Weights, name: str) -> np.ndarray:
    """A centred RMSNorm's scale: 1 + w."""
    return 1 + W(name).numpy()


def half(g: Graph, x: str, name: str) -> str:
    return g.node('Cast', [x], name, to=F16)


def head_rms(g: Graph, x: str, heads: int, head_dim: int, name: str, eps: float) -> str:
    """RMS normalisation (no weights) of each head's slice of x [B, S, heads * head_dim], float16."""
    per_head = g.node('Reshape', [x, g.const(f'{name}/shape', [0, 0, heads, head_dim])], f'{name}/Reshape')
    ones = g.init(f'{name}.weight', np.ones(head_dim, np.float16), F16)
    normed = g.node('SimplifiedLayerNormalization', [per_head, ones], name, axis=-1, epsilon=eps, stash_type=1)
    return g.node('Reshape', [normed, g.const(f'{name}/flat', [0, 0, heads * head_dim])], f'{name}/Flatten')


def attention(g: Graph, W: Weights, cfg: dict, i: int, x: str, mask: dict) -> str:
    p = f'{PREFIX}layers.{i}.self_attn'
    name = f'/model/layers.{i}/attn'
    heads, kv_heads, head_dim, eps = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['rms_norm_eps']
    sliding = cfg['layer_types'][i] == 'sliding_attention'
    rope = bool(cfg['layer_rope_theta'][i])
    assert not rope or cfg['layer_rope_theta'][i] == cfg['rope_parameters']['rope_theta']

    q = head_rms(g, gptq_matmul(g, x, W, f'{p}.q_proj', f'{name}/q_proj/MatMul'), heads, head_dim, f'{name}/q_norm', eps)
    k = head_rms(g, gptq_matmul(g, x, W, f'{p}.k_proj', f'{name}/k_proj/MatMul'), kv_heads, head_dim, f'{name}/k_norm', eps)
    v = gptq_matmul(g, x, W, f'{p}.v_proj', f'{name}/v_proj/MatMul')
    out = f'{name}/GroupQueryAttention/output_0'
    g.node(
        'GroupQueryAttention',
        [q, k, v, f'past_key_values.{i}.key', f'past_key_values.{i}.value', mask['seqlens'], mask['total'],
         *(['cos_cache', 'sin_cache'] if rope else ['', ''])],
        f'{name}/GroupQueryAttention',
        outputs=[out, f'present.{i}.key', f'present.{i}.value'],
        domain=MS,
        num_heads=heads,
        kv_num_heads=kv_heads,
        scale=cfg['qk_scale_factor'] / math.sqrt(head_dim),
        do_rotary=1 if rope else 0,
        rotary_interleaved=0,
        local_window_size=cfg['sliding_window'] if sliding else -1,
    )
    gate = g.node('Sigmoid', [gptq_matmul(g, x, W, f'{p}.gate_proj', f'{name}/gate_proj/MatMul')], f'{name}/gate/Sigmoid')
    gated = g.node('Mul', [out, gate], f'{name}/gate/Mul')
    return gptq_matmul(g, gated, W, f'{p}.o_proj', f'{name}/o_proj/MatMul')


def mlp(g: Graph, W: Weights, i: int, x: str) -> str:
    p, name = f'{PREFIX}layers.{i}.mlp', f'/model/layers.{i}/mlp'
    gt = gptq_matmul(g, x, W, f'{p}.gate_proj', f'{name}/gate_proj/MatMul')
    u = gptq_matmul(g, x, W, f'{p}.up_proj', f'{name}/up_proj/MatMul')
    act = g.node('Mul', [gt, g.node('Sigmoid', [gt], f'{name}/act/Sigmoid')], f'{name}/act/Mul')
    return gptq_matmul(g, g.node('Mul', [act, u], f'{name}/Mul'), W, f'{p}.down_proj', f'{name}/down_proj/MatMul')


def embedding(g: Graph, W: Weights, vocab: int, hidden: int) -> str:
    """The token embeddings as a 4-bit GatherBlockQuantized (symmetric, block 32, codes + 8 with the default zero point 8)."""
    table = W(f'{PREFIX}embed_tokens.weight', native=True)
    block = 32
    codes = np.empty((vocab, hidden // 2), np.uint8)
    scales = np.empty((vocab, hidden // block), np.float16)
    rows = 8192
    for start in range(0, vocab, rows):
        w = table[start:start + rows].float().numpy().reshape(-1, hidden // block, block)
        s = np.abs(w).max(-1, keepdims=True) / 7
        q = np.clip(np.rint(w / np.where(s == 0, 1, s)), -8, 7).astype(np.int16) + 8
        q = q.reshape(q.shape[0], hidden)
        codes[start:start + rows] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        scales[start:start + rows] = s[..., 0].astype(np.float16)
    data = g.init('model.embed_tokens.weight_Q4', codes, U8)
    sc = g.init('model.embed_tokens.weight_scale', scales, F16)
    return g.node('GatherBlockQuantized', [data, 'input_ids', sc], '/model/embed_tokens/GatherBlockQuantized', domain=MS,
                  bits=4, block_size=block, gather_axis=0, quantize_axis=1)


def rotary_caches(g: Graph, cfg: dict):
    dim = cfg['head_dim']
    inv = 1.0 / (cfg['rope_parameters']['rope_theta'] ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = np.outer(np.arange(ROTARY_POSITIONS, dtype=np.float64), inv)
    g.init('cos_cache', np.cos(freqs).astype(np.float16), F16)
    g.init('sin_cache', np.sin(freqs).astype(np.float16), F16)


def build(layers: int, out: Path):
    cfg = json.loads((SRC / 'config.json').read_text())['text_config']
    W = Weights(SRC, SOURCE, REVISION)
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    kv_heads, head_dim, eps, post_eps = cfg['num_key_value_heads'], cfg['head_dim'], cfg['rms_norm_eps'], cfg['post_norm_eps']
    hidden, vocab = cfg['hidden_size'], cfg['vocab_size']

    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', vocab])]
    for i in range(layers):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', kv_heads, 'past_sequence_length', head_dim]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', kv_heads, 'total_sequence_length', head_dim]))
    mask = attention_lengths(g)
    rotary_caches(g, cfg)

    # The embedding first, at the start of the weight file (see Graph.init); normalised (no weights) in float32.
    embedded = g.node('Cast', [embedding(g, W, vocab, hidden)], '/model/embed_tokens/Cast', to=F32)
    residual = norm32(g, embedded, np.ones(hidden), '/model/embed_tokens/embed_norm', eps)
    for i in range(layers):
        print(f'layer {i}', flush=True)
        p, name = f'{PREFIX}layers.{i}', f'/model/layers.{i}'
        h = half(g, norm32(g, residual, centred(W, f'{p}.input_layernorm.weight'), f'{name}/input_layernorm', eps), f'{name}/input_layernorm/Cast')
        a = norm32(g, g.node('Cast', [attention(g, W, cfg, i, h, mask)], f'{name}/attn/Cast', to=F32),
                   centred(W, f'{p}.post_attention_layernorm.weight'), f'{name}/post_attention_layernorm', post_eps)
        # residual + a, and its pre-feedforward norm, in one node.
        wn = g.init(f'{name}/pre_feedforward_layernorm.weight', centred(W, f'{p}.pre_feedforward_layernorm.weight').astype(np.float32), F32)
        outs = [f'{name}/pre_feedforward_layernorm/output_0', '', '', f'{name}/pre_feedforward_layernorm/output_3']
        g.node('SkipSimplifiedLayerNormalization', [a, residual, wn], f'{name}/pre_feedforward_layernorm', outputs=outs, domain=MS, epsilon=eps)
        residual = outs[3]
        m = norm32(g, g.node('Cast', [mlp(g, W, i, half(g, outs[0], f'{name}/pre_feedforward_layernorm/Cast'))], f'{name}/mlp/Cast', to=F32),
                   centred(W, f'{p}.post_feedforward_layernorm.weight'), f'{name}/post_feedforward_layernorm', post_eps)
        residual = g.node('Add', [residual, m], f'{name}/mlp/residual')
    h = half(g, norm32(g, residual, W(f'{PREFIX}norm.weight').numpy(), '/model/norm', eps), '/model/norm/Cast')
    # The logits before the final soft cap, which the app applies to the row it samples from (the catalogue's
    # logitSoftcap): after the output layer it would keep the app's last-position rewrite from applying.
    logits = matmul_nbits(g, h, W('lm_head.weight', True), '/lm_head/MatMul')
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()

    graph = helper.make_graph(g.nodes, 'Muse-Glimmer-30B', g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)


def finish(out: Path):
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'config.json', 'LICENSE', 'USAGE_POLICY.md'):
        shutil.copy(META / name, out / name)
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = f"""---
license: apache-2.0
base_model:
- {ORIGINAL}
- {SOURCE}
tags:
- onnx
- webgpu
- quick-inference
---

# {REPO}

The text model of Meta's [Muse Glimmer 30B](https://huggingface.co/{ORIGINAL}) converted to ONNX for running in a web
browser with ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Made from Red Hat's GPTQ 4-bit checkpoint, [{SOURCE}](https://huggingface.co/{SOURCE}) (symmetric 4-bit weights in
  blocks of 128, calibrated on 512 chat samples): its weight codes and scales are copied unchanged into `MatMulNBits`.
- Text only: the vision tower and projector are left out.
- Converted to an ONNX decoder in ONNX Runtime GenAI's layout (`input_ids`, `attention_mask`, `past_key_values.*` →
  `logits`, `present.*`) by Quick Inference's own exporter (`tools/export/export_glimmer.py` in
  [Quick Inference](https://github.com/mdanyalmalik/quick-inf)), following transformers' Muse Glimmer model code:
  `GroupQueryAttention` with the 2,048-token sliding window and rotary embeddings on the sliding layers only, the
  output gate and the sandwich norms.
- `logits` are **before** the final soft cap: apply `20 * tanh(logits * 0.19611613513818404 / 20)` (the config's
  `output_multiplier` and `final_logit_softcapping`) before sampling, as Quick Inference does.
- The token embeddings and the output layer, which Red Hat kept in bfloat16, are quantised to 4 bits here (round to
  nearest, symmetric, block size 32). Activations and the KV cache are float16; the residual stream and the norms are
  float32. Rotary tables cover 65,536 positions.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

Apache 2.0, as the original (LICENSE). Meta's usage policy for the model is in USAGE_POLICY.md.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=None, help='only the first N layers (for testing)')
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    args = parser.parse_args()
    cfg = json.loads((SRC / 'config.json').read_text())['text_config']
    build(args.layers or cfg['num_hidden_layers'], args.out)
    finish(args.out)
