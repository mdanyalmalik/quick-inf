"""Converts a llama.cpp GGUF of a Llama-architecture model to ONNX for Quick Inference, keeping its calibrated
quantization: every quantized matrix goes into com.microsoft MatMulGgml (the token embedding into GatherGgml), ops of
Quick Inference's ONNX Runtime build that read llama.cpp's types directly (ggml_pack.py has the layout). Not one weight
is re-quantized: each matrix is checked to decode to exactly gguf-py's dequantization of the original.

The graph is export_dense.py's (ONNX Runtime GenAI's layout, rotary embedding inside GroupQueryAttention). llama.cpp's
converter interleaves the two halves of each query and key head (for its rotary embedding on adjacent pairs); rows are
put back in the original order, so the graph is the original model's.

    tools/export/.venv/bin/python tools/export/export_gguf.py nanbeige4.3-33b-iq2_m [--layers N] [--out DIR]
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import gguf
import numpy as np
import torch
import onnx
from onnx import TensorProto, helper

import ggml_pack
from onnx_export import F16, F32, I32, I64, MS, Graph, attention_lengths, rms_norm, skip_rms_norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
U8 = TensorProto.UINT8

NANBEIGE = {
    'source': 'Nanbeige/Nanbeige4.3-33B',
    'revision': 'c1093c501cc6460427c0ac634db7223d0450c0ee',
    'folder': 'nanbeige-src',
    'license': 'apache-2.0',
    # The rotary tables cover this many positions (the model's own 524,288 would take 128 MB).
    'max_positions': 131072,
    'files': ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json'],
}
MODELS = {
    # bartowski's IQ2_M (importance matrix from his calibration-v6 text): closest to the Q6_K model of the 2-bit
    # GGUFs measured (mean KL divergence 0.41 against abenzerps' GSQ-RCO IQ2_M's 0.47; same top token 74% vs 71%).
    'nanbeige4.3-33b-iq2_m': {
        **NANBEIGE,
        'gguf': 'bartowski/Nanbeige_Nanbeige4.3-33B-GGUF',
        'gguf_revision': '9beb68421fdf7be095093a6888325fa2c83ddd38',
        'gguf_file': 'Nanbeige_Nanbeige4.3-33B-IQ2_M.gguf',
        'repo': 'Nanbeige4.3-33B-IQ2_M-ONNX-WebGPU',
        'quality': 'mean KL divergence 0.41, same top token 74% (bartowski Q4_K_M: 0.051, 91.5%)',
    },
    # bartowski's Q4_K_M: the closest to the Q6_K model of the 4-bit GGUFs measured (mean KL divergence 0.051; his
    # IQ4_XS 0.068, abenzerps' IQ4_XS 0.071).
    'nanbeige4.3-33b-q4_k_m': {
        **NANBEIGE,
        'gguf': 'bartowski/Nanbeige_Nanbeige4.3-33B-GGUF',
        'gguf_revision': '9beb68421fdf7be095093a6888325fa2c83ddd38',
        'gguf_file': 'Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf',
        'repo': 'Nanbeige4.3-33B-Q4_K_M-ONNX-WebGPU',
        'quality': 'mean KL divergence 0.051, same top token 91.5% (bartowski IQ2_M: 0.41, 74%)',
    },
}


class Weights:
    """The GGUF's tensors by name (memory-mapped)."""

    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.grids = {}

    def float(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            raise ValueError(f'{name}: {t.tensor_type.name} is not a float type')
        return torch.from_numpy(np.asarray(t.data, np.float32).reshape(tuple(reversed([int(d) for d in t.shape]))))

    def packed(self, name: str, heads: int | None = None) -> tuple[np.ndarray, int, int, int]:
        """(layout bytes [N, row bytes], ggml type, K, N), checked against gguf-py; with `heads`, rows go back to
        the original order of each head's two halves."""
        t = self.tensors[name]
        kind, k, n = int(t.tensor_type), int(t.shape[0]), int(t.shape[1])
        if kind not in ggml_pack.LAYOUTS:
            raise ValueError(f'{name}: {t.tensor_type.name} is not supported')
        data = np.asarray(t.data).reshape(n, -1)
        packed = np.empty((n, ggml_pack.row_words(kind, k)), np.uint32)
        for start in range(0, n, 2048):
            rows = data[start:start + 2048]
            packed[start:start + len(rows)] = ggml_pack.pack(kind, rows, len(rows), k)
            ggml_pack.check(kind, rows, len(rows), k, packed[start:start + len(rows)])
        if heads is not None:
            dim = n // heads
            # llama.cpp's row h * dim + 2i + t is the original's h * dim + (dim / 2) t + i.
            h, t_, i = np.meshgrid(np.arange(heads), np.arange(2), np.arange(dim // 2), indexing='ij')
            packed = packed[(h * dim + 2 * i + t_).reshape(-1)]
        return packed.view(np.uint8), kind, k, n

    def grid(self, g: Graph, kind: int) -> list[str]:
        """The type's grid input (shared by every node of the type), or none."""
        table = ggml_pack.grid(kind)
        if table is None:
            return []
        values = table.view(np.uint8)
        if kind not in self.grids:
            self.grids[kind] = g.init(f'ggml_grid.{gguf.GGMLQuantizationType(kind).name}', values, U8)
        return [self.grids[kind]]


def linear(g: Graph, W: Weights, x: str, tensor: str, name: str, heads: int | None = None) -> str:
    if W.tensors[tensor].tensor_type == gguf.GGMLQuantizationType.Q8_0:
        return q8_0_linear(g, W, x, tensor, name, heads)
    data, kind, k, n = W.packed(tensor, heads)
    b = g.init(f'{name}.weight', data, U8)
    return g.node('MatMulGgml', [x, b, *W.grid(g, kind)], name, domain=MS, K=k, N=n, ggml_type=kind)


def q8_0_linear(g: Graph, W: Weights, x: str, tensor: str, name: str, heads: int | None) -> str:
    """Q8_0 (blocks of 32: a float16 scale d, 32 signed bytes q; weight d * q) is MatMulNBits' 8-bit form exactly:
    codes q + 128 with the default zero point 128, scales d, blocks of 32. accuracy_level 0 keeps A in float16."""
    t = W.tensors[tensor]
    k, n = int(t.shape[0]), int(t.shape[1])
    blocks = np.asarray(t.data, np.uint8).reshape(n, k // 32, 34)
    scales = blocks[:, :, :2].copy().view('<f2').reshape(n, k // 32)
    codes = (blocks[:, :, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
    want = ggml_pack.quants.dequantize(blocks.reshape(n, -1), gguf.GGMLQuantizationType.Q8_0).reshape(n, k)
    got = (codes.astype(np.float32) - 128) * scales.astype(np.float32)[..., None]
    if not np.array_equal(got.reshape(n, k), want):
        raise ValueError(f'{tensor}: Q8_0 as MatMulNBits differs from gguf-py')
    if heads is not None:
        dim = n // heads
        h, t_, i = np.meshgrid(np.arange(heads), np.arange(2), np.arange(dim // 2), indexing='ij')
        order = (h * dim + 2 * i + t_).reshape(-1)
        codes, scales = codes[order], scales[order]
    b = g.init(f'{name}.weight', np.ascontiguousarray(codes), U8)
    sc = g.init(f'{name}.scales', np.ascontiguousarray(scales), F16)
    return g.node('MatMulNBits', [x, b, sc], name, domain=MS, K=k, N=n, bits=8, block_size=32, accuracy_level=0)


def rotary_caches(g: Graph, theta: float, dim: int, positions: int):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float64), inv_freq)
    g.init('cos_cache', freqs.cos().numpy().astype(np.float16), F16)
    g.init('sin_cache', freqs.sin().numpy().astype(np.float16), F16)


def field(W: Weights, key: str):
    f = W.reader.fields[key]
    return f.parts[f.data[0]][0]


def build(spec: dict, layers: int | None, out: Path):
    W = Weights(HERE / '.cache' / 'hf' / 'nanbeige-gguf' / spec['gguf_file'])
    arch = bytes(W.reader.fields['general.architecture'].parts[-1]).decode()
    if arch != 'llama':
        raise ValueError(f'{arch}: only llama GGUFs are supported')
    hidden = int(field(W, 'llama.embedding_length'))
    heads, kv_heads = int(field(W, 'llama.attention.head_count')), int(field(W, 'llama.attention.head_count_kv'))
    dim = int(field(W, 'llama.attention.key_length'))
    eps = float(field(W, 'llama.attention.layer_norm_rms_epsilon'))
    vocab = int(W.tensors['output.weight'].shape[1])
    layers = layers or int(field(W, 'llama.block_count'))
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', vocab])]
    for i in range(layers):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{i}.{kind}', F16, ['batch_size', kv_heads, 'past_sequence_length', dim]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{i}.{kind}', F16, ['batch_size', kv_heads, 'total_sequence_length', dim]))
    mask = attention_lengths(g)
    rotary_caches(g, float(field(W, 'llama.rope.freq_base')), dim, spec['max_positions'])

    emb, kind, _, _ = W.packed('token_embd.weight')
    ids = g.node('Cast', ['input_ids'], '/model/embed_tokens/Cast', to=I32)
    residual = g.node('GatherGgml', [g.init('model.embed_tokens.weight', emb, U8), ids, *W.grid(g, kind)],
                      '/model/embed_tokens/GatherGgml', domain=MS, K=hidden, ggml_type=kind)
    del emb
    h = rms_norm(g, residual, W.float('blk.0.attn_norm.weight'), '/model/layers.0/input_layernorm', eps)
    for i in range(layers):
        print(f'layer {i}', flush=True)
        blk, name = f'blk.{i}', f'/model/layers.{i}'
        q = linear(g, W, h, f'{blk}.attn_q.weight', f'{name}/attn/q_proj/MatMul', heads)
        k = linear(g, W, h, f'{blk}.attn_k.weight', f'{name}/attn/k_proj/MatMul', kv_heads)
        v = linear(g, W, h, f'{blk}.attn_v.weight', f'{name}/attn/v_proj/MatMul')
        attn = f'{name}/attn/GroupQueryAttention/output_0'
        g.node('GroupQueryAttention',
               [q, k, v, f'past_key_values.{i}.key', f'past_key_values.{i}.value', mask['seqlens'], mask['total'], 'cos_cache', 'sin_cache'],
               f'{name}/attn/GroupQueryAttention', outputs=[attn, f'present.{i}.key', f'present.{i}.value'], domain=MS,
               num_heads=heads, kv_num_heads=kv_heads, scale=1 / math.sqrt(dim), do_rotary=1, rotary_interleaved=0, local_window_size=-1)
        a = linear(g, W, attn, f'{blk}.attn_output.weight', f'{name}/attn/o_proj/MatMul')
        h, residual = skip_rms_norm(g, a, residual, W.float(f'{blk}.ffn_norm.weight'), f'{name}/post_attention_layernorm', eps)
        gate = linear(g, W, h, f'{blk}.ffn_gate.weight', f'{name}/mlp/gate_proj/MatMul')
        up = linear(g, W, h, f'{blk}.ffn_up.weight', f'{name}/mlp/up_proj/MatMul')
        act = g.node('Mul', [gate, g.node('Sigmoid', [gate], f'{name}/mlp/act/Sigmoid')], f'{name}/mlp/act/Mul')
        f = linear(g, W, g.node('Mul', [act, up], f'{name}/mlp/Mul'), f'{blk}.ffn_down.weight', f'{name}/mlp/down_proj/MatMul')
        if i + 1 < layers:
            h, residual = skip_rms_norm(g, f, residual, W.float(f'blk.{i + 1}.attn_norm.weight'), f'/model/layers.{i + 1}/input_layernorm', eps)
        else:
            h, _ = skip_rms_norm(g, f, residual, W.float('output_norm.weight'), '/model/norm', eps)
    logits = linear(g, W, h, 'output.weight', '/lm_head/MatMul')
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()
    graph = helper.make_graph(g.nodes, spec['repo'], g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)


def finish(spec: dict, out: Path):
    src = HERE / '.cache' / 'hf' / spec['folder']
    for name in spec['files']:
        shutil.copy(src / name, out / name)
    source, gguf_repo = spec['source'], spec['gguf']
    (out / 'README.md').write_text(f"""---
license: {spec['license']}
base_model:
- {source}
tags:
- onnx
- webgpu
- quick-inference
- gguf
---

# {spec['repo']}

[{source}](https://huggingface.co/{source}) for running in a web browser with ONNX Runtime Web's WebGPU backend, as
used by Quick Inference: [{gguf_repo}](https://huggingface.co/{gguf_repo})'s `{spec['gguf_file']}` (llama.cpp's
calibrated quantization, made with an importance matrix) converted to ONNX without re-quantizing anything.

## Runtime

The quantized matrices use two operators that only Quick Inference's ONNX Runtime Web build has
([onnxruntime-web-kernels](https://github.com/mdanyalmalik/onnxruntime-web-kernels), v1.30.0-qi.5 or later, WebGPU):
`com.microsoft.MatMulGgml` and `com.microsoft.GatherGgml`, which read llama.cpp's quantization types (IQ2_S, IQ3_S,
IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K) directly. Stock ONNX Runtime can't load this model.

## Changes from the original

- Converted to an ONNX decoder in ONNX Runtime GenAI's layout (`input_ids`, `attention_mask`, `past_key_values.*` →
  `logits`, `present.*`), with the rotary embedding inside `GroupQueryAttention`. Activations and the KV cache are
  float16; norms are float16 weights.
- Every quantized matrix keeps the GGUF's weights exactly: its blocks are only re-laid out so that each 32-weight
  sub-block's codes sit in words of their own (the file is the same size), and each was checked to decode to exactly
  what gguf-py decodes from the GGUF. Q8_0 matrices are ONNX Runtime's own 8-bit `MatMulNBits` (the same codes, plus
  128, and scales). llama.cpp's converter interleaves the two halves of each query and key head;
  their rows are put back in the original order.
- The rotary tables cover {spec['max_positions']:,} positions (the model's own maximum is 524,288).

These are modified files: their outputs are close to, but not identical to, the original model's.

## Quality

Measured with llama.cpp on the source GGUF against bartowski's Q6_K (WikiText-2 test, 20 chunks of 512 tokens):
{spec['quality']}.

## Licence

Licensed under the same terms as the original model: {spec['license']}.
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=sorted(MODELS))
    parser.add_argument('--layers', type=int, default=None, help='only the first N layers (for testing)')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    spec = MODELS[args.model]
    out = args.out or ROOT / 'exports' / spec['repo']
    build(spec, args.layers, out)
    finish(spec, out)
