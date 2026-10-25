"""Exports a Qwen3.5-family model's multi-token prediction (MTP) layer, which the model's own checkpoint ships
(`mtp.*`, config mtp_num_hidden_layers 1), to ONNX for speculative decoding in the app (OnnxArtifact.draft, kind 'mtp').

As vLLM's Qwen3_5MultiTokenPredictor: for target position t, norm(embedding of the token at t + 1) and norm(the
target's final hidden state at t, after its final norm) are concatenated and mapped to the hidden size (fc), then one
full-attention decoder layer (gated attention: q_proj gives each head's query and output gate; per-head q/k RMS norms;
rotary on the first 64 dimensions; its own KV cache at the target's positions), a final norm, and the target's output
layer. Its output hidden state is fed back for the next guess in a chain. Qwen3.5's RMS norms are zero-centred
(x * (1 + w)): the weights get the 1 here.

The graph takes token ids (looked up in a copy of the target's 4-bit embedding table: the same bytes as its embeddings
session) and returns, for one row (`last`), the guessed token and its output hidden state, which a chain's next guess
feeds back, so it stays on the GPU. The output layer's 4-bit weights are copied from the target's ONNX graph (the same
bytes); the MTP layer's matrices are quantised (8-bit by default: the layer is small, and 4-bit cost accuracy).

Guesses come from the first `--vocab` token ids only (32,768 by default; Qwen's ids roughly follow frequency): the output
layer and embedding table copies keep those rows, ~8x smaller and faster to read, for ~3% fewer kept guesses (Qwen3.5
0.8B, English). Ids past them (a token the model chose) are clamped for the lookup: only that row's input suffers.

    tools/export/.venv/bin/python tools/export/export_mtp.py qwen3.5-0.8b [--out DIR]
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

from onnx_export import F16, I32, I64, MS, Graph, Settings, matmul_nbits, qmoe_weights, rms_norm, skip_rms_norm

U8 = TensorProto.UINT8
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = ROOT / 'tools' / 'reference' / '.cache'

# folder: the HF checkpoint under tools/export/.cache/hf (only the shards holding mtp.* are needed); target: the
# catalogue artifact's ONNX files in the local model cache (the output layer is copied from its graph).
MODELS = {
    'qwen3.6-35b-a3b': {
        'folder': 'Qwen3.6-35B-A3B',
        'target': 'amd__Qwen3.6-35B-A3B_gpuep_1.0_gpu_128/c2fdd8966cfde2da5fa780004fe6fad4a1125bb8',
        'graph': 'text.onnx',
        'embeddings': 'embedding.onnx',
    },
    'qwen3.8-27b': {
        'folder': 'Qwen3.8-27B',
        'target': 'amd__Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-onnx/3d00750aaa18e3e09e0b271a82e7762a4586718f',
        'graph': 'text.onnx',
        'embeddings': 'embedding.onnx',
    },
    'qwen3.5-2b': {
        'repo': 'Qwen/Qwen3.5-2B',
        'folder': 'Qwen3.5-2B',
        'target': 'onnx-community__Qwen3.5-2B-ONNX/e988a4ce430d7dc6fd77b4b93341b0aeae289748/onnx',
        'graph': 'decoder_model_merged_q4f16.onnx',
        'embeddings': 'embed_tokens_q4f16.onnx',
    },
    'qwen3.5-4b': {
        'repo': 'Qwen/Qwen3.5-4B',
        'folder': 'Qwen3.5-4B',
        'target': 'onnx-community__Qwen3.5-4B-ONNX/3551704c0c04ee52b5f1bfc27d1ee8d2a94973fc/onnx',
        'graph': 'decoder_model_merged_q4f16.onnx',
        'embeddings': 'embed_tokens_q4f16.onnx',
    },
    'qwen3.5-9b': {
        'repo': 'Qwen/Qwen3.5-9B',
        'folder': 'Qwen3.5-9B',
        'target': 'amd__Qwen3.5-9B_gpuep_1.0_gpu_128/bad679999a75e4883384cdac008972614cfb0d00',
        'graph': 'text.onnx',
        'embeddings': 'embedding.onnx',
    },
    'qwen3.5-0.8b': {
        'folder': 'Qwen3.5-0.8B',
        'target': 'onnx-community__Qwen3.5-0.8B-ONNX/d1ea95774d40a29f0479bf9138cdf488a0394b6b/onnx',
        'graph': 'decoder_model_merged_q4f16.onnx',
        'embeddings': 'embed_tokens_q4f16.onnx',
    },
}


def fetch_mtp(repo: str, folder: Path):
    """Only the checkpoint's mtp.* tensors, by HTTP byte ranges (the safetensors headers give their offsets), into
    folder/mtp.safetensors, with config.json: a few hundred MB instead of every shard holding one of them."""
    import struct

    import requests
    from huggingface_hub import hf_hub_download, hf_hub_url
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo, 'config.json', local_dir=folder)
    try:
        index = json.loads(Path(hf_hub_download(repo, 'model.safetensors.index.json', local_dir=folder)).read_text())['weight_map']
    except Exception:
        index = None
    shards = sorted({v for k, v in index.items() if k.startswith('mtp.')}) if index else ['model.safetensors']
    dtypes = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
    tensors = {}
    for shard in shards:
        url = hf_hub_url(repo, shard)
        get = lambda a, b: requests.get(url, headers={'Range': f'bytes={a}-{b - 1}'}, timeout=600).content
        n = struct.unpack('<Q', get(0, 8))[0]
        header = json.loads(get(8, 8 + n))
        for name, meta in header.items():
            if not name.startswith('mtp.'):
                continue
            a, b = meta['data_offsets']
            raw = bytearray(get(8 + n + a, 8 + n + b))
            tensors[name] = torch.frombuffer(raw, dtype=dtypes[meta['dtype']]).reshape(meta['shape']).clone()
    save_file(tensors, str(folder / 'mtp.safetensors'))
    print('fetched', len(tensors), 'mtp tensors from', shards, flush=True)


def zero_centred(w: torch.Tensor) -> torch.Tensor:
    return 1.0 + w.float()


def copy_initializers(g: Graph, model_path: Path, names: list[str], vocab: int, rows: int) -> list[str]:
    """Copies per-token initializers (with their external data) from another ONNX file into this graph, renamed
    '/copied/<name>', keeping the first `rows` of `vocab` entries (each entry: an equal share of the tensor)."""
    model = onnx.load(str(model_path), load_external_data=False)
    inits = {t.name: t for t in model.graph.initializer}
    out = []
    for name in names:
        if not name:
            out.append('')
            continue
        array = numpy_helper.to_array(inits[name], base_dir=str(model_path.parent))
        if rows < vocab:
            kept = array.reshape(vocab, -1)[:rows]
            array = kept.reshape((rows, *array.shape[1:]) if array.shape[0] == vocab else (-1,))
        out.append(g.init(f'/copied/{name}', np.ascontiguousarray(array), inits[name].data_type))
    return out


def find_head(model_path: Path) -> onnx.NodeProto:
    """The node producing the logits (through Casts): the output layer."""
    model = onnx.load(str(model_path), load_external_data=False)
    producer = {o: n for n in model.graph.node for o in n.output}
    node = producer['logits']
    while node.op_type in ('Cast', 'Identity'):
        node = producer[node.input[0]]
    if node.op_type != 'MatMulNBits':
        raise SystemExit(f'output layer is {node.op_type}, expected MatMulNBits')
    return node


def quantized_table(g: Graph, table: np.ndarray) -> list[str]:
    """A float embedding table as GatherBlockQuantized's 4-bit inputs (symmetric, block 32: codes + 8, the default zero
    point), [codes, scales]."""
    vocab, hidden = table.shape
    block = 32
    codes = np.empty((vocab, hidden // 2), np.uint8)
    scales = np.empty((vocab, hidden // block), np.float16)
    for start in range(0, vocab, 8192):
        w = table[start:start + 8192].astype(np.float32).reshape(-1, hidden // block, block)
        sc = np.abs(w).max(-1, keepdims=True) / 7
        q = np.clip(np.rint(w / np.where(sc == 0, 1, sc)), -8, 7).astype(np.int16) + 8
        q = q.reshape(q.shape[0], hidden)
        codes[start:start + 8192] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        scales[start:start + 8192] = sc[..., 0].astype(np.float16)
    return [g.init('/embed_tokens/weight_Q4', codes, U8), g.init('/embed_tokens/weight_scale', scales, F16)]


def find_embedding(model_path: Path) -> onnx.NodeProto:
    model = onnx.load(str(model_path), load_external_data=False)
    nodes = [n for n in model.graph.node if n.op_type == 'GatherBlockQuantized'] or [n for n in model.graph.node if n.op_type == 'Gather']
    if not nodes:
        raise SystemExit('no embedding lookup found')
    return nodes[0]


def silu_mlp(g: Graph, sd: dict, p: str, x: str, name: str) -> str:
    gate = matmul_nbits(g, x, sd[f'{p}.gate_proj.weight'].float(), f'{name}/gate_proj/MatMul')
    up = matmul_nbits(g, x, sd[f'{p}.up_proj.weight'].float(), f'{name}/up_proj/MatMul')
    act = g.node('Mul', [gate, g.node('Sigmoid', [gate], f'{name}/act/Sigmoid')], f'{name}/act/Mul')
    return matmul_nbits(g, g.node('Mul', [act, up], f'{name}/Mul'), sd[f'{p}.down_proj.weight'].float(), f'{name}/down_proj/MatMul')


def moe(g: Graph, sd: dict, cfg: dict, x: str) -> str:
    """Qwen3.5-MoE's sparse block (Qwen3.6 35B-A3B's MTP layer): softmax router over the experts, top k renormalised
    (QMoE takes the router's logits and does both), SwiGLU experts (gate_up_proj rows: gate then up; interleaved here
    for swiglu_fusion=1), 4-bit like MatMulNBits; plus a shared SwiGLU expert scaled by sigmoid(shared_expert_gate(x))."""
    p = 'mtp.layers.0.mlp'
    experts, k = cfg['num_experts'], cfg['num_experts_per_tok']
    router = matmul_nbits(g, x, sd[f'{p}.gate.weight'].float(), '/moe/router/MatMul')
    router = g.node('Reshape', [router, g.const('/moe/router/shape', [-1, experts])], '/moe/router/Reshape')
    bits, block = Settings.bits, Settings.block
    gate_up, down = sd[f'{p}.experts.gate_up_proj'], sd[f'{p}.experts.down_proj']
    inter = gate_up.shape[1] // 2
    fc1_q, fc1_s, fc2_q, fc2_s = [], [], [], []
    for e in range(experts):
        gate_w, up_w = gate_up[e, :inter].float(), gate_up[e, inter:].float()
        q, sc = qmoe_weights(torch.stack([gate_w, up_w], dim=1).reshape(-1, gate_w.shape[1]), bits, block)
        fc1_q.append(q)
        fc1_s.append(sc)
        q, sc = qmoe_weights(down[e].float(), bits, block)
        fc2_q.append(q)
        fc2_s.append(sc)
    hidden = fc2_q[0].shape[0]
    names = [
        g.init(f'{p}.experts.gate_up_proj.qweight', np.stack(fc1_q), U8),
        g.init(f'{p}.experts.gate_up_proj.scales', np.stack(fc1_s), F16),
        g.init(f'{p}.experts.gate_up_proj.bias', np.zeros((experts, 2 * inter), np.float16), F16),
        g.init(f'{p}.experts.down_proj.qweight', np.stack(fc2_q), U8),
        g.init(f'{p}.experts.down_proj.scales', np.stack(fc2_s), F16),
        g.init(f'{p}.experts.down_proj.bias', np.zeros((experts, hidden), np.float16), F16),
    ]
    flat = g.node('Reshape', [x, g.const('/moe/x_shape', [-1, cfg['hidden_size']])], '/moe/x/Reshape')
    routed = g.node('QMoE', [flat, router, *names], '/moe/QMoE', domain=MS, activation_type='swiglu', swiglu_fusion=1,
                    activation_alpha=1.0, activation_beta=0.0, k=k, normalize_routing_weights=1, expert_weight_bits=bits,
                    block_size=block, use_sparse_mixer=0)
    routed = g.node('Reshape', [routed, g.const('/moe/out_shape', [1, -1, cfg['hidden_size']])], '/moe/out/Reshape')
    shared = silu_mlp(g, sd, f'{p}.shared_expert', x, '/moe/shared_expert')
    shared_gate = g.node('Sigmoid', [matmul_nbits(g, x, sd[f'{p}.shared_expert_gate.weight'].float(), '/moe/shared_expert_gate/MatMul')], '/moe/shared_expert_gate/Sigmoid')
    return g.node('Add', [routed, g.node('Mul', [shared, shared_gate], '/moe/shared/Mul')], '/moe/Add')


def build(spec: dict, out: Path, draft_vocab: int):
    src = HERE / '.cache' / 'hf' / spec['folder']
    if not (src / 'config.json').exists() and 'repo' in spec:
        fetch_mtp(spec['repo'], src)
    cfg = json.loads((src / 'config.json').read_text())
    cfg = cfg.get('text_config', cfg)
    sd = {}
    for f in glob.glob(str(src / '*.safetensors')):
        with safe_open(f, 'pt') as s:
            for k in s.keys():
                if k.startswith('mtp.'):
                    sd[k] = s.get_tensor(k)
    if not sd:
        raise SystemExit(f'no mtp.* tensors in {src}')
    target = CACHE / spec['target']

    H = cfg['hidden_size']
    heads, kv_heads, dim = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
    eps = cfg['rms_norm_eps']
    rope = cfg.get('rope_parameters') or {}
    theta = rope.get('rope_theta', cfg.get('rope_theta', 10000.0))
    rotary_dim = int(dim * rope.get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0)))
    w = lambda name: sd[name].float()
    p = 'mtp.layers.0'

    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, [1, 'sequence_length']),
        helper.make_tensor_value_info('hidden', F16, [1, 'sequence_length', H]),
        helper.make_tensor_value_info('position_ids', I64, [1, 'sequence_length']),
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

    # Rotary tables for the rotated dimensions (text: the three position rows are equal, so mrope is plain rope).
    positions = cfg.get('max_position_embeddings', 262144)
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float64), inv)
    cos = g.init('cos_cache', freqs.cos().numpy().astype(np.float16), F16)
    sin = g.init('sin_cache', freqs.sin().numpy().astype(np.float16), F16)

    # The token embeddings: the target's 4-bit table (GatherBlockQuantized) copied.
    emb = find_embedding(target / spec['embeddings'])
    vocab = cfg['vocab_size']
    rows = min(draft_vocab or vocab, vocab)
    if emb.op_type == 'GatherBlockQuantized':
        emb_inputs = copy_initializers(g, target / spec['embeddings'], [emb.input[0], *emb.input[2:]], vocab, rows)
        emb_attrs = {a.name: helper.get_attribute_value(a) for a in emb.attribute}
    else:
        # A float table (AMD's exports): its first rows, quantised to 4-bit here.
        emb_model = onnx.load(str(target / spec['embeddings']), load_external_data=False)
        table = next(t for t in emb_model.graph.initializer if t.name == emb.input[0])
        emb_inputs = quantized_table(g, numpy_helper.to_array(table, base_dir=str(target))[:rows])
        emb_attrs = {'bits': 4, 'block_size': 32, 'gather_axis': 0, 'quantize_axis': 1}
    ids = 'input_ids' if rows == vocab else g.node('Min', ['input_ids', g.const('/embed_tokens/last_id', [rows - 1])], '/embed_tokens/Min')
    embeds = g.node('GatherBlockQuantized', [emb_inputs[0], ids, *emb_inputs[1:]], '/embed_tokens/GatherBlockQuantized', domain=MS, **emb_attrs)
    e = rms_norm(g, embeds, zero_centred(sd['mtp.pre_fc_norm_embedding.weight']), '/pre_fc_norm_embedding', eps)
    h = rms_norm(g, 'hidden', zero_centred(sd['mtp.pre_fc_norm_hidden.weight']), '/pre_fc_norm_hidden', eps)
    x = matmul_nbits(g, g.node('Concat', [e, h], '/fc/Concat', axis=-1), w('mtp.fc.weight'), '/fc/MatMul')

    a = rms_norm(g, x, zero_centred(sd[f'{p}.input_layernorm.weight']), '/input_layernorm', eps)
    qg = matmul_nbits(g, a, w(f'{p}.self_attn.q_proj.weight'), '/attn/q_proj/MatMul')
    qg = g.node('Reshape', [qg, g.const('/attn/qg_shape', [1, -1, heads, 2 * dim])], '/attn/qg/Reshape')
    q, gate = f'/attn/q_split/output_0', f'/attn/q_split/output_1'
    g.node('Split', [qg, g.const('/attn/q_split/sizes', [dim, dim])], '/attn/q_split', outputs=[q, gate], axis=3)

    def head_norm(t: str, n: int, weight: torch.Tensor, name: str) -> str:
        t = g.node('Reshape', [t, g.const(f'{name}/shape4', [1, -1, n, dim])], f'{name}/Reshape4')
        t = rms_norm(g, t, zero_centred(weight), name, eps)
        return g.node('Reshape', [t, g.const(f'{name}/shape3', [1, -1, n * dim])], f'{name}/Reshape3')

    q = head_norm(q, heads, sd[f'{p}.self_attn.q_norm.weight'], '/attn/q_norm')
    k = head_norm(matmul_nbits(g, a, w(f'{p}.self_attn.k_proj.weight'), '/attn/k_proj/MatMul'), kv_heads, sd[f'{p}.self_attn.k_norm.weight'], '/attn/k_norm')
    v = matmul_nbits(g, a, w(f'{p}.self_attn.v_proj.weight'), '/attn/v_proj/MatMul')
    q = g.node('RotaryEmbedding', [q, 'position_ids', cos, sin], '/attn/q_rotary', domain=MS, num_heads=heads,
               rotary_embedding_dim=rotary_dim, interleaved=0)
    k = g.node('RotaryEmbedding', [k, 'position_ids', cos, sin], '/attn/k_rotary', domain=MS, num_heads=kv_heads,
               rotary_embedding_dim=rotary_dim, interleaved=0)
    attn = '/attn/GroupQueryAttention/output_0'
    g.node('GroupQueryAttention', [q, k, v, 'past_key_values.0.key', 'past_key_values.0.value', 'seqlens_k', 'total_sequence_length'],
           '/attn/GroupQueryAttention', outputs=[attn, 'present.0.key', 'present.0.value'], domain=MS, num_heads=heads,
           kv_num_heads=kv_heads, scale=1 / math.sqrt(dim), local_window_size=-1)
    gate = g.node('Reshape', [gate, g.const('/attn/gate_shape', [1, -1, heads * dim])], '/attn/gate/Reshape')
    attn = g.node('Mul', [attn, g.node('Sigmoid', [gate], '/attn/gate/Sigmoid')], '/attn/gate/Mul')
    o = matmul_nbits(g, attn, w(f'{p}.self_attn.o_proj.weight'), '/attn/o_proj/MatMul')

    m_in, residual = skip_rms_norm(g, o, x, zero_centred(sd[f'{p}.post_attention_layernorm.weight']), '/post_attention_layernorm', eps)
    m = moe(g, sd, cfg, m_in) if f'{p}.mlp.experts.gate_up_proj' in sd else silu_mlp(g, sd, f'{p}.mlp', m_in, '/mlp')
    normed = rms_norm(g, g.node('Add', [m, residual], '/mlp/residual'), zero_centred(sd['mtp.norm.weight']), '/norm', eps)
    row = g.node('Gather', [normed, 'last'], '/last/Gather', axis=1)  # [1, 1, H]
    g.nodes.append(helper.make_node('Identity', [row], ['hidden_out'], name='/hidden_out/Identity'))

    # The target's output layer, its own 4-bit bytes and attributes.
    head = find_head(target / spec['graph'])
    attrs = {a.name: helper.get_attribute_value(a) for a in head.attribute}
    head_inputs = copy_initializers(g, target / spec['graph'], list(head.input[1:]), attrs['N'], rows)
    attrs['N'] = rows
    vocab = rows
    logits = g.node('MatMulNBits', [row, *head_inputs], '/lm_head/MatMul', domain=MS, **attrs)
    # Argmax in two levels (ONNX Runtime WebGPU's ArgMax walks a row in one thread).
    width = next(c for c in (128, 100, 64, 50, 32, 25, 16, 10, 8, 5, 4, 2, 1) if vocab % c == 0)
    grid = g.node('Reshape', [logits, g.const('/token/grid', [vocab // width, width])], '/token/Reshape')
    inner = g.node('ArgMax', [grid], '/token/inner/ArgMax', axis=1, keepdims=0)
    peaks = g.node('ReduceMax', [grid, g.const('/token/axes', [1])], '/token/inner/ReduceMax', keepdims=0)
    best_row = g.node('ArgMax', [peaks], '/token/outer/ArgMax', axis=0, keepdims=1)
    col = g.node('Gather', [inner, best_row], '/token/col/Gather', axis=0)
    flat = g.node('Add', [g.node('Mul', [best_row, g.const('/token/width', [width])], '/token/Mul'), col], '/token/flat')
    g.nodes.append(helper.make_node('Reshape', [flat, g.const('/token/shape', [1, 1])], ['token'], name='/token/Reshape2'))

    g.close()
    graph = helper.make_graph(g.nodes, f"{spec['folder']}-MTP", g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    (out / 'mtp.json').write_text(json.dumps({'hidden_size': H, 'kv_heads': kv_heads, 'head_dim': dim}, indent=2) + '\n')
    print('saved', out, g.offset, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=sorted(MODELS))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--bits', type=int, default=8, choices=[4, 8])
    parser.add_argument('--vocab', type=int, default=32768, help='guess among the first N token ids (0: all)')
    args = parser.parse_args()
    Settings.bits = args.bits
    Settings.block = 32
    spec = MODELS[args.model]
    build(spec, args.out or ROOT / 'exports' / f"{spec['folder']}-MTP", args.vocab)
