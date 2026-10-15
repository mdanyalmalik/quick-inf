"""Exports dense decoders to ONNX in ONNX Runtime GenAI's layout with little memory: weights stream from the checkpoint
one tensor at a time (see onnx_export.py), unlike ONNX Runtime GenAI's builder, which holds the model's float16
weights in memory while it builds (several GB of swap on a 24 GB machine for an 8B model).

Supported: Qwen3ForCausalLM (DeepSeek-R1-0528-Qwen3-8B) and Gemma 3's text model (Gemma3ForConditionalGeneration,
without its vision tower). Linear layers and the LM head: 4-bit MatMulNBits (round to nearest, block size 32);
embeddings, activations and KV cache float16; rotary embedding inside GroupQueryAttention. Gemma 3 keeps its residual
stream and norms in float32 (its activations outgrow float16), with sliding-window attention on 5 of every 6 layers.

    tools/export/.venv/bin/python tools/export/export_dense.py deepseek-r1-0528-qwen3-8b [--layers N] [--out DIR]
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
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from onnx_export import F16, F32, I64, MS, to_float16, Graph, Settings, Weights, attention_lengths, matmul_nbits, rms_norm, skip_rms_norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

MODELS = {
    'deepseek-r1-0528-qwen3-8b': {
        'source': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'revision': '6e8885a6ff5c1dc5201574c8fd700323f23c25fa',
        'folder': 'DeepSeek-R1-0528-Qwen3-8B',
        'repo': 'DeepSeek-R1-0528-Qwen3-8B-ONNX-WebGPU',
        'license': 'mit',
        # The config sets YaRN's attn_factor 0.878, which cancels YaRN's default magnitude scaling (1 + 0.1 ln 4) to
        # exactly 1, as vLLM applies it; transformers ignores attn_factor.
        'attention_factor': 1.0,
        'changes': ["YaRN's magnitude scaling is 1.0: the config's attn_factor (0.878) times YaRN's default (1 + 0.1 ln 4), as vLLM applies it."],
        'files': ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'LICENSE'],
    },
    'janus-pro-7b': {
        # The language model of Janus Pro 7B's image generation (see export_janus.py for the rest): input embeddings
        # in, final hidden states out (Janus has its own heads for text and image tokens).
        'arch': 'llama',
        'embeds': True,
        'source': 'deepseek-community/Janus-Pro-7B',
        'revision': '8f23139ed8d37346501091b3690291cb7b16f9fc',
        'folder': 'Janus-Pro-7B',
        'prefix': 'model.language_model.',
        'repo': 'Janus-Pro-7B-ONNX-WebGPU',
        'graph': 'language_model.onnx',
        'files': [],
    },
    'gemma-3-12b-it': {
        'arch': 'gemma3',
        # Google's repo is gated; unsloth's copy has the same weights without the gate.
        'source': 'unsloth/gemma-3-12b-it',
        'original': 'google/gemma-3-12b-it',
        'revision': '9478e665381f42974aa06177b019352fb6291876',
        'folder': 'gemma-3-12b-it',
        'repo': 'gemma-3-12b-it-ONNX-WebGPU',
        'license': 'gemma',
        'files': ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json', 'config.json', 'generation_config.json'],
        'licences': ['GEMMA_TERMS_OF_USE.md', 'GEMMA_PROHIBITED_USE_POLICY.md'],
        'notice': 'Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n',
        'changes': ['Only the language model is converted (the vision tower is left out): text in, text out.'],
        'restrictions': (
            '\n## Use restrictions\n\n'
            'Gemma is provided under and subject to the Gemma Terms of Use (GEMMA_TERMS_OF_USE.md). By using these files you '
            'agree to those terms, including the use restrictions in their section 3.2: you may not use these files or '
            'anything made from them for the restricted uses in the Gemma Prohibited Use Policy '
            '(GEMMA_PROHIBITED_USE_POLICY.md). If you distribute them or anything made from them, you must pass these '
            'restrictions on in the same way.\n'
        ),
    },
}


def rotary_caches(g: Graph, config, attention_factor: float | None):
    """cos/sin caches [max positions, head_dim / 2] from transformers' own rotary parameters."""
    params = config.rope_parameters
    if attention_factor is not None:
        params['attention_factor'] = attention_factor
    rope_type = params.get('rope_type', 'default')
    if rope_type == 'default':
        dim = config.head_dim
        inv_freq = 1.0 / (params['rope_theta'] ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        scale = 1.0
    else:
        inv_freq, scale = ROPE_INIT_FUNCTIONS[rope_type](config, 'cpu')
    freqs = torch.outer(torch.arange(config.max_position_embeddings, dtype=torch.float64), inv_freq.double())
    g.init('cos_cache', (freqs.cos() * scale).numpy().astype(np.float16), F16)
    g.init('sin_cache', (freqs.sin() * scale).numpy().astype(np.float16), F16)


def qwen3_layer(g: Graph, W, cfg, i: int, x: str, mask: dict) -> tuple[str, str]:
    """Attention block of layer i; returns its output."""
    p = f'layers.{i}.self_attn'
    name = f'/model/layers.{i}/attn'
    heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps = cfg.rms_norm_eps

    def projected(proj: str, n_heads: int, norm: str | None) -> str:
        y = matmul_nbits(g, x, W(f'{p}.{proj}.weight'), f'{name}/{proj}/MatMul')
        if norm is None:
            return y
        y = g.node('Reshape', [y, g.const(f'{name}/{proj}/heads', [0, 0, n_heads, dim])], f'{name}/{proj}/Reshape')
        y = rms_norm(g, y, W(f'{p}.{norm}.weight'), f'{name}/{norm}', eps)
        return g.node('Reshape', [y, g.const(f'{name}/{proj}/flat', [0, 0, n_heads * dim])], f'{name}/{proj}/Flat')

    norms = getattr(cfg, 'model_type', '') == 'qwen3'
    q = projected('q_proj', heads, 'q_norm' if norms else None)
    k = projected('k_proj', kv_heads, 'k_norm' if norms else None)
    v = projected('v_proj', kv_heads, None)
    out = f'{name}/GroupQueryAttention/output_0'
    g.node(
        'GroupQueryAttention',
        [q, k, v, f'past_key_values.{i}.key', f'past_key_values.{i}.value', mask['seqlens'], mask['total'], 'cos_cache', 'sin_cache'],
        f'{name}/GroupQueryAttention',
        outputs=[out, f'present.{i}.key', f'present.{i}.value'],
        domain=MS,
        num_heads=heads,
        kv_num_heads=kv_heads,
        scale=1 / math.sqrt(dim),
        do_rotary=1,
        rotary_interleaved=0,
        local_window_size=-1,
    )
    return matmul_nbits(g, out, W(f'{p}.o_proj.weight'), f'{name}/o_proj/MatMul')


def silu_mlp(g: Graph, W, i: int, x: str) -> str:
    p = f'layers.{i}.mlp'
    name = f'/model/layers.{i}/mlp'
    gate = matmul_nbits(g, x, W(f'{p}.gate_proj.weight'), f'{name}/gate_proj/MatMul')
    up = matmul_nbits(g, x, W(f'{p}.up_proj.weight'), f'{name}/up_proj/MatMul')
    act = g.node('Mul', [gate, g.node('Sigmoid', [gate], f'{name}/act/Sigmoid')], f'{name}/act/Mul')
    return matmul_nbits(g, g.node('Mul', [act, up], f'{name}/Mul'), W(f'{p}.down_proj.weight'), f'{name}/down_proj/MatMul')


def gemma_norm(g: Graph, x: str, w: torch.Tensor, name: str, eps: float) -> str:
    """Gemma's RMSNorm (scale 1 + w), in float32."""
    wn = g.init(f'{name}.weight', (1 + w).numpy().astype(np.float32), F32)
    return g.node('SimplifiedLayerNormalization', [x, wn], name, axis=-1, epsilon=eps, stash_type=1)


def build_gemma3(spec: dict, layers: int | None, out: Path):
    src = HERE / '.cache' / 'hf' / spec['folder']
    cfg = AutoConfig.from_pretrained(src).text_config
    layers = layers or cfg.num_hidden_layers
    W0 = Weights(src, spec['source'], spec['revision'])
    W = lambda name: W0(f'language_model.model.{name}')
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    heads, kv_heads, dim, eps = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps

    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', cfg.vocab_size])]
    for i in range(layers):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', kv_heads, 'past_sequence_length', dim]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', kv_heads, 'total_sequence_length', dim]))
    mask = attention_lengths(g)

    # The 2 GB embedding first, at the start of the weight file (see Graph.init).
    embed = W0('language_model.model.embed_tokens.weight', native=True)
    emb = g.init('model.embed_tokens.weight', to_float16(embed), F16)
    # Global layers: rope_theta with linear scaling; local (sliding-window) layers: rope_local_base_freq.
    positions = torch.arange(cfg.max_position_embeddings, dtype=torch.float64)
    rope = cfg.rope_parameters
    for label, params in (('global', rope['full_attention']), ('local', rope['sliding_attention'])):
        theta, factor = params['rope_theta'], params.get('factor', 1.0) if params['rope_type'] == 'linear' else 1.0
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)) / factor
        freqs = torch.outer(positions, inv)
        g.init(f'cos_cache_{label}', freqs.cos().numpy().astype(np.float16), F16)
        g.init(f'sin_cache_{label}', freqs.sin().numpy().astype(np.float16), F16)

    x = g.node('Gather', [emb, 'input_ids'], '/model/embed_tokens/Gather')
    x = g.node('Cast', [x], '/model/embed_tokens/Cast', to=F32)
    # Gemma scales the embeddings by sqrt(hidden size), rounded to the model's bfloat16.
    scale = float(torch.tensor(cfg.hidden_size**0.5, dtype=torch.bfloat16))
    residual = g.node('Mul', [x, g.const('/model/embed_tokens/scale', scale, F32)], '/model/embed_tokens/Mul')
    half = lambda t, n: g.node('Cast', [t], n, to=F16)
    full = lambda t, n: g.node('Cast', [t], n, to=F32)

    for i in range(layers):
        print(f'layer {i}', flush=True)
        p, name = f'layers.{i}', f'/model/layers.{i}'
        local = cfg.layer_types[i] == 'sliding_attention'
        h = half(gemma_norm(g, residual, W(f'{p}.input_layernorm.weight'), f'{name}/input_layernorm', eps), f'{name}/input_layernorm/Cast')

        def projected(proj: str, n_heads: int, norm: str | None) -> str:
            y = matmul_nbits(g, h, W(f'{p}.self_attn.{proj}.weight'), f'{name}/attn/{proj}/MatMul')
            if norm is None:
                return y
            y = g.node('Reshape', [y, g.const(f'{name}/attn/{proj}/heads', [0, 0, n_heads, dim])], f'{name}/attn/{proj}/Reshape')
            y = half(gemma_norm(g, full(y, f'{name}/attn/{norm}/Cast32'), W(f'{p}.self_attn.{norm}.weight'), f'{name}/attn/{norm}', eps), f'{name}/attn/{norm}/Cast16')
            return g.node('Reshape', [y, g.const(f'{name}/attn/{proj}/flat', [0, 0, n_heads * dim])], f'{name}/attn/{proj}/Flat')

        q, k, v = projected('q_proj', heads, 'q_norm'), projected('k_proj', kv_heads, 'k_norm'), projected('v_proj', kv_heads, None)
        label = 'local' if local else 'global'
        attn = f'{name}/attn/GroupQueryAttention/output_0'
        g.node(
            'GroupQueryAttention',
            [q, k, v, f'past_key_values.{i}.key', f'past_key_values.{i}.value', mask['seqlens'], mask['total'], f'cos_cache_{label}', f'sin_cache_{label}'],
            f'{name}/attn/GroupQueryAttention',
            outputs=[attn, f'present.{i}.key', f'present.{i}.value'],
            domain=MS,
            num_heads=heads,
            kv_num_heads=kv_heads,
            scale=cfg.query_pre_attn_scalar**-0.5,
            do_rotary=1,
            rotary_interleaved=0,
            local_window_size=cfg.sliding_window if local else -1,
        )
        a = full(matmul_nbits(g, attn, W(f'{p}.self_attn.o_proj.weight'), f'{name}/attn/o_proj/MatMul'), f'{name}/attn/o_proj/Cast')
        a = gemma_norm(g, a, W(f'{p}.post_attention_layernorm.weight'), f'{name}/post_attention_layernorm', eps)
        residual = g.node('Add', [residual, a], f'{name}/attn/residual')

        h = half(gemma_norm(g, residual, W(f'{p}.pre_feedforward_layernorm.weight'), f'{name}/pre_feedforward_layernorm', eps), f'{name}/pre_feedforward_layernorm/Cast')
        gate = matmul_nbits(g, h, W(f'{p}.mlp.gate_proj.weight'), f'{name}/mlp/gate_proj/MatMul')
        up = matmul_nbits(g, h, W(f'{p}.mlp.up_proj.weight'), f'{name}/mlp/up_proj/MatMul')
        act = g.node('FastGelu', [gate], f'{name}/mlp/FastGelu', domain=MS)
        m = matmul_nbits(g, g.node('Mul', [act, up], f'{name}/mlp/Mul'), W(f'{p}.mlp.down_proj.weight'), f'{name}/mlp/down_proj/MatMul')
        m = gemma_norm(g, full(m, f'{name}/mlp/Cast'), W(f'{p}.post_feedforward_layernorm.weight'), f'{name}/post_feedforward_layernorm', eps)
        residual = g.node('Add', [residual, m], f'{name}/mlp/residual')

    h = half(gemma_norm(g, residual, W('norm.weight'), '/model/norm', eps), '/model/norm/Cast')
    # The LM head is the embedding matrix (tied).
    logits = matmul_nbits(g, h, embed, '/lm_head/MatMul')
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()
    graph = helper.make_graph(g.nodes, spec['repo'], g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)


def build(spec: dict, layers: int | None, out: Path):
    if spec.get('arch') == 'gemma3':
        return build_gemma3(spec, layers, out)
    src = HERE / '.cache' / 'hf' / spec['folder']
    cfg = AutoConfig.from_pretrained(src)
    cfg = getattr(cfg, 'text_config', cfg)
    layers = layers or cfg.num_hidden_layers
    W0 = Weights(src, spec['source'], spec['revision'])
    prefix = spec.get('prefix', 'model.')
    W = lambda name, native=False: W0(prefix + name, native)
    embeds = spec.get('embeds', False)
    graph_name = spec.get('graph', 'model.onnx')
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / (graph_name + '.data'))
    kv_heads, dim, eps = cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps

    g.inputs = [
        helper.make_tensor_value_info('inputs_embeds', F16, ['batch_size', 'sequence_length', cfg.hidden_size])
        if embeds
        else helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [
        helper.make_tensor_value_info('hidden_states', F16, ['batch_size', 'sequence_length', cfg.hidden_size])
        if embeds
        else helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', cfg.vocab_size])
    ]
    for i in range(layers):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', kv_heads, 'past_sequence_length', dim]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', kv_heads, 'total_sequence_length', dim]))
    mask = attention_lengths(g)
    rotary_caches(g, cfg, spec.get('attention_factor'))

    if embeds:
        residual = 'inputs_embeds'
    else:
        emb = g.init('model.embed_tokens.weight', to_float16(W('embed_tokens.weight', native=True)), F16)
        residual = g.node('Gather', [emb, 'input_ids'], '/model/embed_tokens/Gather')
    h = rms_norm(g, residual, W('layers.0.input_layernorm.weight'), '/model/layers.0/input_layernorm', eps)
    for i in range(layers):
        print(f'layer {i}', flush=True)
        a = qwen3_layer(g, W, cfg, i, h, mask)
        h, residual = skip_rms_norm(g, a, residual, W(f'layers.{i}.post_attention_layernorm.weight'), f'/model/layers.{i}/post_attention_layernorm', eps)
        f = silu_mlp(g, W, i, h)
        if i + 1 < layers:
            h, residual = skip_rms_norm(g, f, residual, W(f'layers.{i + 1}.input_layernorm.weight'), f'/model/layers.{i + 1}/input_layernorm', eps)
        else:
            h, _ = skip_rms_norm(g, f, residual, W('norm.weight'), '/model/norm', eps)
    if embeds:
        g.nodes.append(helper.make_node('Identity', [h], ['hidden_states'], name='/model/Identity'))
    else:
        logits = matmul_nbits(g, h, W0('lm_head.weight', True), '/lm_head/MatMul')
        g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()
    graph = helper.make_graph(g.nodes, spec['repo'], g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / graph_name))
    print('saved', out, g.offset, flush=True)


def finish(spec: dict, out: Path):
    if 'license' not in spec:
        return  # Part of a bigger export with its own files and card (Janus: export_janus.py).
    src = HERE / '.cache' / 'hf' / spec['folder']
    for name in spec['files']:
        shutil.copy(src / name, out / name)
    for name in spec.get('licences', []):
        shutil.copy(HERE / 'licences' / name, out / name)
    if 'notice' in spec:
        (out / 'NOTICE').write_text(spec['notice'])
    source = spec.get('original', spec['source'])
    licence_files = ', '.join(f for f in sorted(p.name for p in out.iterdir()) if f.startswith(('LICENSE', 'NOTICE', 'GEMMA')))
    (out / 'README.md').write_text(f"""---
license: {spec['license']}
base_model:
- {source}
tags:
- onnx
- webgpu
- quick-inference
---

# {spec['repo']}

[{source}](https://huggingface.co/{source}) converted to ONNX for running in a web browser with ONNX Runtime Web's
WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to an ONNX decoder in ONNX Runtime GenAI's layout (`input_ids`, `attention_mask`, `past_key_values.*` →
  `logits`, `present.*`) by Quick Inference's own exporter, with the rotary embedding inside `GroupQueryAttention`.
- Linear layers and the LM head are quantised to 4 bits (round to nearest, block size 32). Embeddings, activations and
  the KV cache are float16.
{''.join(f'- {c}' + chr(10) for c in spec.get('changes', []))}
These are modified files: their outputs are close to, but not identical to, the original model's.
{spec.get('restrictions', '')}
## Licence

Licensed under the same terms as the original model: see {licence_files}.
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=sorted(MODELS))
    parser.add_argument('--layers', type=int, default=None, help='only the first N layers (for testing)')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8])
    args = parser.parse_args()
    Settings.bits = args.bits
    spec = MODELS[args.model]
    out = args.out or ROOT / 'exports' / spec['repo']
    build(spec, args.layers, out)
    finish(spec, out)
