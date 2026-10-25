"""Exports an EAGLE-3 draft head (one decoder layer that guesses the next tokens from the target model's hidden states)
to ONNX for speculative decoding in the app.

The head reads, per position t, the target's residual stream entering three of its layers at t (`taps`, concatenated,
as the target graph outputs them), mapped to the hidden size by its `fc`, together with the embedding of the token at
t + 1; later guesses feed the head's own output (`hidden`) instead of taps. Positions are the target's: the head keeps
its own KV cache alongside the target's (static: past and present are the same buffer, GroupQueryAttention reads
`seqlens_k` and `total_sequence_length` directly). The graph returns, for one row (`last`), its guess as a token id of
the target's vocabulary (argmax over the head's smaller vocabulary, mapped by `d2t`) and its output hidden state, so a
chain of guesses can stay on the GPU.

Weights: 4-bit MatMulNBits (round to nearest, block 32); the token embeddings (the target's, which the head shares)
a 4-bit GatherBlockQuantized; norms, activations and the cache float16.

    tools/export/.venv/bin/python tools/export/export_eagle3.py qwen3-1.7b [--out DIR]
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from onnx import TensorProto, helper

from onnx_export import F16, I32, I64, MS, Graph, Settings, Weights, matmul_nbits, rms_norm, skip_rms_norm

U8 = TensorProto.UINT8
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

HEADS = {
    'qwen3-1.7b': {
        'head': 'AngelSlim/Qwen3-1.7B_eagle3',
        'head_revision': '94441b48acc5804677ae12259617c83323b543a9',
        'target': 'Qwen/Qwen3-1.7B',
        'target_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        'folder': 'Qwen3-1.7B',
        'repo': 'Qwen3-1.7B-EAGLE3-ONNX-WebGPU',
        'license': 'apache-2.0',
        'licence_file': 'License_AngelSlim_model_and_dataset.txt',
    },
    'qwen3-4b': {
        'head': 'AngelSlim/Qwen3-4B_eagle3',
        'head_revision': 'fd331e59626c8e95c392381a16ee59d518727fbb',
        'target': 'Qwen/Qwen3-4B',
        'target_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
        'folder': 'Qwen3-4B',
        'repo': 'Qwen3-4B-EAGLE3-ONNX-WebGPU',
        'license': 'apache-2.0',
        'licence_file': 'License_AngelSlim_model_and_dataset.txt',
    },
    'llama-3.2-3b-instruct': {
        'head': 'thoughtworks/Llama-3.2-3B-Instruct-Eagle3',
        'head_revision': '02d343789b502a3edfe351bdd4537a44affb98cd',
        'target': 'unsloth/Llama-3.2-3B-Instruct',
        'target_revision': '006f5dcd1393c3add266de40994ba96225e9689d',
        'folder': 'Llama-3.2-3B-Instruct',
        'repo': 'Llama-3.2-3B-Instruct-EAGLE3-ONNX-WebGPU',
        'license': 'llama3.2',
        'licence_file': None,
    },
    'llama-3.1-8b-instruct': {
        'head': 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
        'head_revision': 'ada412b672e293d682423de84a095447bf38a637',
        'target': 'unsloth/Llama-3.1-8B-Instruct',
        'target_revision': '4699cc75b550f9c6f3173fb80f4703b62d946aa5',
        'folder': 'Llama-3.1-8B-Instruct',
        'repo': 'Llama-3.1-8B-Instruct-EAGLE3-ONNX-WebGPU',
        'license': 'apache-2.0',
        'licence_file': 'LICENSE',
    },
    'qwen3-8b': {
        'head': 'AngelSlim/Qwen3-8B_eagle3',
        'head_revision': '9629dfce7a4a10564dd48d3e5485c3976095653c',
        'target': 'Qwen/Qwen3-8B',
        'target_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
        'folder': 'Qwen3-8B',
        'repo': 'Qwen3-8B-EAGLE3-ONNX-WebGPU',
        'license': 'apache-2.0',
        'licence_file': 'License_AngelSlim_model_and_dataset.txt',
    },
}

# The target layers whose input (residual stream) the head reads: EAGLE-3's choice.
def tap_layers(layers: int) -> list[int]:
    return [2, layers // 2, layers - 3]


def embedding(g: Graph, table: torch.Tensor) -> str:
    """4-bit GatherBlockQuantized (symmetric, block 32, codes + 8 with the default zero point 8)."""
    vocab, hidden = table.shape
    block = 32
    codes = np.empty((vocab, hidden // 2), np.uint8)
    scales = np.empty((vocab, hidden // block), np.float16)
    for start in range(0, vocab, 8192):
        w = table[start:start + 8192].float().numpy().reshape(-1, hidden // block, block)
        s = np.abs(w).max(-1, keepdims=True) / 7
        q = np.clip(np.rint(w / np.where(s == 0, 1, s)), -8, 7).astype(np.int16) + 8
        q = q.reshape(q.shape[0], hidden)
        codes[start:start + 8192] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        scales[start:start + 8192] = s[..., 0].astype(np.float16)
    data = g.init('embed_tokens.weight_Q4', codes, U8)
    sc = g.init('embed_tokens.weight_scale', scales, F16)
    return g.node('GatherBlockQuantized', [data, 'input_ids', sc], '/embed_tokens/GatherBlockQuantized', domain=MS,
                  bits=4, block_size=block, gather_axis=0, quantize_axis=1)


def build(spec: dict, out: Path):
    head_dir = Path(snapshot_download(spec['head'], revision=spec['head_revision']))
    cfg = json.loads((head_dir / 'config.json').read_text())
    files = list(head_dir.glob('*.bin')) + list(head_dir.glob('*.safetensors'))
    if files[0].suffix == '.bin':
        sd = torch.load(files[0], map_location='cpu', weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(str(files[0]))
    src = HERE / '.cache' / 'hf' / spec['folder']
    target_cfg = json.loads((src / 'config.json').read_text())
    W = Weights(src, spec['target'], spec['target_revision'])

    H = cfg['hidden_size']
    heads, kv_heads = cfg['num_attention_heads'], cfg['num_key_value_heads']
    dim = cfg.get('head_dim', H // heads)
    eps = cfg['rms_norm_eps']
    taps = tap_layers(target_cfg['num_hidden_layers'])
    w = lambda name: sd[name].float()

    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, [1, 'sequence_length']),
        helper.make_tensor_value_info('taps', F16, [1, 'tap_length', 3 * H]),
        helper.make_tensor_value_info('hidden', F16, [1, 'hidden_length', H]),
        helper.make_tensor_value_info('last', I64, [1]),
        helper.make_tensor_value_info('seqlens_k', I32, [1]),
        helper.make_tensor_value_info('total_sequence_length', I32, []),
        helper.make_tensor_value_info('past_key_values.0.key', F16, [1, kv_heads, 'max_length', dim]),
        helper.make_tensor_value_info('past_key_values.0.value', F16, [1, kv_heads, 'max_length', dim]),
    ]
    g.outputs = [
        helper.make_tensor_value_info('token', I64, [1, 1]),
        helper.make_tensor_value_info('hidden_out', F16, [1, 1, H]),
        helper.make_tensor_value_info('present.0.key', F16, [1, kv_heads, 'max_length', dim]),
        helper.make_tensor_value_info('present.0.value', F16, [1, kv_heads, 'max_length', dim]),
    ]

    # Rotary caches (the head's theta) for GroupQueryAttention.
    positions = target_cfg.get('max_position_embeddings', 40960)
    # EAGLE's own code falls back to Llama's default theta when the config has none.
    inv = 1.0 / (cfg.get('rope_theta', 10000.0) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float64), inv)
    g.init('cos_cache', freqs.cos().numpy().astype(np.float16), F16)
    g.init('sin_cache', freqs.sin().numpy().astype(np.float16), F16)

    emb = embedding(g, W('model.embed_tokens.weight', native=True))
    mapped = matmul_nbits(g, 'taps', w('fc.weight'), '/fc/MatMul')
    hid = g.node('Concat', [mapped, 'hidden'], '/hidden/Concat', axis=1)
    x = g.node('Concat', [rms_norm(g, emb, w('midlayer.input_layernorm.weight'), '/input_layernorm', eps),
                          rms_norm(g, hid, w('midlayer.hidden_norm.weight'), '/hidden_norm', eps)], '/attn/Concat', axis=-1)
    p = 'midlayer.self_attn'
    q = matmul_nbits(g, x, w(f'{p}.q_proj.weight'), '/attn/q_proj/MatMul')
    k = matmul_nbits(g, x, w(f'{p}.k_proj.weight'), '/attn/k_proj/MatMul')
    v = matmul_nbits(g, x, w(f'{p}.v_proj.weight'), '/attn/v_proj/MatMul')
    attn = '/attn/GroupQueryAttention/output_0'
    g.node('GroupQueryAttention', [q, k, v, 'past_key_values.0.key', 'past_key_values.0.value', 'seqlens_k', 'total_sequence_length',
                                   'cos_cache', 'sin_cache'], '/attn/GroupQueryAttention',
           outputs=[attn, 'present.0.key', 'present.0.value'], domain=MS, num_heads=heads, kv_num_heads=kv_heads,
           scale=1 / math.sqrt(dim), do_rotary=1, rotary_interleaved=0, local_window_size=-1)
    o = matmul_nbits(g, attn, w(f'{p}.o_proj.weight'), '/attn/o_proj/MatMul')
    h, residual = skip_rms_norm(g, o, hid, w('midlayer.post_attention_layernorm.weight'), '/post_attention_layernorm', eps)
    gate = matmul_nbits(g, h, w('midlayer.mlp.gate_proj.weight'), '/mlp/gate_proj/MatMul')
    up = matmul_nbits(g, h, w('midlayer.mlp.up_proj.weight'), '/mlp/up_proj/MatMul')
    act = g.node('Mul', [gate, g.node('Sigmoid', [gate], '/mlp/act/Sigmoid')], '/mlp/act/Mul')
    m = matmul_nbits(g, g.node('Mul', [act, up], '/mlp/Mul'), w('midlayer.mlp.down_proj.weight'), '/mlp/down_proj/MatMul')
    row = g.node('Gather', [g.node('Add', [m, residual], '/mlp/residual'), 'last'], '/last/Gather', axis=1)  # [1, 1, H]
    g.nodes.append(helper.make_node('Identity', [row], ['hidden_out'], name='/hidden_out/Identity'))
    logits = matmul_nbits(g, rms_norm(g, row, w('norm.weight'), '/norm', eps), w('lm_head.weight'), '/lm_head/MatMul')
    # Argmax in two levels (ONNX Runtime WebGPU's ArgMax walks a row in one thread: 2.6 ms for 32000 values).
    vocab = sd['lm_head.weight'].shape[0]
    width = next(w for w in (128, 100, 64, 50, 32, 25, 16, 10, 8, 5, 4, 2, 1) if vocab % w == 0)
    grid = g.node('Reshape', [logits, g.const('/token/grid', [vocab // width, width])], '/token/Reshape')
    inner = g.node('ArgMax', [grid], '/token/inner/ArgMax', axis=1, keepdims=0)  # [vocab / width]
    peaks = g.node('ReduceMax', [grid, g.const('/token/axes', [1])], '/token/inner/ReduceMax', keepdims=0)
    row = g.node('ArgMax', [peaks], '/token/outer/ArgMax', axis=0, keepdims=1)  # [1]
    col = g.node('Gather', [inner, row], '/token/col/Gather', axis=0)  # [1]
    flat = g.node('Add', [g.node('Mul', [row, g.const('/token/width', [width])], '/token/Mul'), col], '/token/flat')
    best = g.node('Reshape', [flat, g.const('/token/shape', [1, 1])], '/token/Shape')  # [1, 1] in the head's vocabulary
    d2t = g.init('d2t', sd['d2t'].numpy().astype(np.int64), I64)
    g.nodes.append(helper.make_node('Add', [best, g.node('Gather', [d2t, best], '/token/Gather', axis=0)], ['token'], name='/token/Add'))
    g.close()
    graph = helper.make_graph(g.nodes, spec['repo'], g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    (out / 'eagle3.json').write_text(json.dumps({'taps': taps, 'hidden_size': H, 'kv_heads': kv_heads, 'head_dim': dim}, indent=2) + '\n')
    if spec['licence_file']:
        shutil.copy(head_dir / spec['licence_file'], out / spec['licence_file'])
    print('saved', out, g.offset, 'taps', taps, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=sorted(HEADS))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8])
    args = parser.parse_args()
    Settings.bits = args.bits
    spec = HEADS[args.model]
    build(spec, args.out or ROOT / 'exports' / spec['repo'])
