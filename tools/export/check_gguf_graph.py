"""Reference for a few-layer export_gguf.py graph (--layers N): a numpy forward pass straight from the GGUF, in
llama.cpp's own conventions (query/key rows as stored, rotary embedding on adjacent pairs), so it also checks the
converter's row re-ordering. Writes an opcheck.html case (tools/reference/.cache/micro/0/<case>.*) whose model is the
export; prints the case name.

    tools/export/.venv/bin/python tools/export/check_gguf_graph.py GGUF EXPORT_DIR_UNDER_MICRO LAYERS
"""

import json
import sys
from pathlib import Path

import gguf
import numpy as np
from gguf import quants

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'reference' / '.cache' / 'micro' / '0'


def main():
    path, folder, layers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    r = gguf.GGUFReader(path)
    T = {t.name: t for t in r.tensors}
    f = lambda k: r.fields[k].parts[r.fields[k].data[0]][0]

    def w(name):
        t = T[name]
        shape = tuple(reversed([int(d) for d in t.shape]))
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            return np.asarray(t.data, np.float32).reshape(shape)
        return quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(shape).astype(np.float32)

    heads, kv, dim = int(f('llama.attention.head_count')), int(f('llama.attention.head_count_kv')), int(f('llama.attention.key_length'))
    eps, theta = float(f('llama.attention.layer_norm_rms_epsilon')), float(f('llama.rope.freq_base'))
    ids = np.array([[166100, 8481, 13, 18215, 19433, 2714, 13, 25389]], np.int64)
    n = ids.shape[1]
    rms = lambda x, g: x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g
    inv = 1.0 / theta ** (np.arange(0, dim, 2) / dim)
    ang = np.arange(n)[:, None] * inv[None]
    cos, sin = np.cos(ang), np.sin(ang)

    def rope(x):  # [n, h, dim], adjacent pairs (llama.cpp's NORM rope on its stored rows)
        a, b = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = a * cos[:, None] - b * sin[:, None]
        out[..., 1::2] = a * sin[:, None] + b * cos[:, None]
        return out

    x = w('token_embd.weight')[ids[0]].astype(np.float64)
    for i in range(layers):
        b = f'blk.{i}'
        h = rms(x, w(f'{b}.attn_norm.weight'))
        q = rope((h @ w(f'{b}.attn_q.weight').T).reshape(n, heads, dim))
        k = rope((h @ w(f'{b}.attn_k.weight').T).reshape(n, kv, dim))
        v = (h @ w(f'{b}.attn_v.weight').T).reshape(n, kv, dim)
        k, v = np.repeat(k, heads // kv, 1), np.repeat(v, heads // kv, 1)
        s = np.einsum('qhd,khd->hqk', q, k) / np.sqrt(dim) + np.triu(np.full((n, n), -np.inf), 1)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        x = x + np.einsum('hqk,khd->qhd', p, v).reshape(n, heads * dim) @ w(f'{b}.attn_output.weight').T
        h = rms(x, w(f'{b}.ffn_norm.weight'))
        g = h @ w(f'{b}.ffn_gate.weight').T
        x = x + ((g / (1 + np.exp(-g))) * (h @ w(f'{b}.ffn_up.weight').T)) @ w(f'{b}.ffn_down.weight').T
    logits = rms(x, w('output_norm.weight')) @ w('output.weight').T

    name = f'{folder}_check'
    inputs = {'input_ids': ids, 'attention_mask': np.ones((1, n), np.int64)}
    for i in range(layers):
        for kind in ('key', 'value'):
            inputs[f'past_key_values.{i}.{kind}'] = np.zeros((1, kv, 0, dim), np.float16)
    types = {np.int64: 'int64', np.float16: 'float16'}
    for k_, v_ in inputs.items():
        v_.tofile(OUT / f'{name}.{k_}.bin')
    logits.astype(np.float32).reshape(1, n, -1).tofile(OUT / f'{name}.logits.bin')
    (OUT / f'{name}.json').write_text(json.dumps({
        'model': f'{folder}/model.onnx', 'externalData': ['model.onnx.data'],
        'inputs': [{'name': k_, 'type': types[v_.dtype.type], 'dims': list(v_.shape)} for k_, v_ in inputs.items()],
        'outputs': [{'name': 'logits', 'type': 'float16', 'dims': [1, n, logits.shape[-1]]}],
    }))
    print(name, 'top tokens', logits.argmax(-1).tolist())


if __name__ == '__main__':
    main()
