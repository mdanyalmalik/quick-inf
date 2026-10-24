"""Checks export_dense.py's Llama 3.3 Nemotron Super graph against a float32 PyTorch reference written from NVIDIA's
DeciLM model code (modeling_decilm.py: layers whose attention is a no-op skip it and its norm; MLP widths per layer),
with Llama 3's rotary scaling written out here from Meta's formula. Compares logits on the first N layers: a prompt,
then a few tokens decoded against the cache. With 8-bit weights the two should agree closely.

    tools/export/.venv/bin/python tools/export/export_dense.py llama-3.3-nemotron-super-49b-v1.5 --layers 8 --bits 8 --out DIR
    tools/export/.venv/bin/python tools/export/check_nemotron_nas.py DIR --layers 8
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from export_dense import HERE, MODELS
from onnx_export import Weights

SPEC = MODELS['llama-3.3-nemotron-super-49b-v1.5']
SRC = HERE / '.cache' / 'hf' / SPEC['folder']


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def llama3_inv_freq(cfg: dict, dim: int) -> torch.Tensor:
    s = cfg['rope_scaling']
    inv = 1.0 / (cfg['rope_theta'] ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    old = s['original_max_position_embeddings']
    low, high = old / s['low_freq_factor'], old / s['high_freq_factor']
    out = []
    for f in inv.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < high:
            out.append(f)
        elif wavelen > low:
            out.append(f / s['factor'])
        else:
            smooth = (old / wavelen - s['low_freq_factor']) / (s['high_freq_factor'] - s['low_freq_factor'])
            out.append((1 - smooth) * f / s['factor'] + smooth * f)
    return torch.tensor(out, dtype=torch.float64)


class Reference:
    """The first `layers` layers in float32, with a KV cache (positions continue from it)."""

    def __init__(self, cfg: dict, W: Weights, layers: int):
        self.cfg, self.W, self.layers = cfg, W, layers
        self.cache = [None] * layers
        self.dim = cfg['hidden_size'] // cfg['num_attention_heads']
        self.inv = llama3_inv_freq(cfg, self.dim)

    def attention(self, i: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cfg, W, p, D = self.cfg, self.W, f'model.layers.{i}.self_attn', self.dim
        H = cfg['num_attention_heads']
        KV = H // cfg['block_configs'][i]['attention']['n_heads_in_group']
        S = x.shape[0]
        q = (x @ W(f'{p}.q_proj.weight').T).view(S, H, D)
        k = (x @ W(f'{p}.k_proj.weight').T).view(S, KV, D)
        v = (x @ W(f'{p}.v_proj.weight').T).view(S, KV, D)
        ang = torch.outer(pos.double(), self.inv)
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None]
        sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None]
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if self.cache[i] is not None:
            k, v = torch.cat([self.cache[i][0], k]), torch.cat([self.cache[i][1], v])
        self.cache[i] = (k, v)
        T = k.shape[0]
        k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
        scores = torch.einsum('shd,thd->hst', q, k) / math.sqrt(D)
        allowed = torch.arange(T)[None, :] <= pos[:, None]
        probs = torch.softmax(scores.masked_fill(~allowed[None], float('-inf')), -1)
        return torch.einsum('hst,thd->shd', probs, v).reshape(S, H * D) @ W(f'{p}.o_proj.weight').T

    def __call__(self, ids: list[int], start: int) -> torch.Tensor:
        cfg, W, eps = self.cfg, self.W, self.cfg['rms_norm_eps']
        pos = torch.arange(start, start + len(ids))
        x = W('model.embed_tokens.weight')[torch.tensor(ids)]
        for i in range(self.layers):
            if not cfg['block_configs'][i]['attention']['no_op']:
                x = x + self.attention(i, rms(x, W(f'model.layers.{i}.input_layernorm.weight'), eps), pos)
            h = rms(x, W(f'model.layers.{i}.post_attention_layernorm.weight'), eps)
            p = f'model.layers.{i}.mlp'
            x = x + (F.silu(h @ W(f'{p}.gate_proj.weight').T) * (h @ W(f'{p}.up_proj.weight').T)) @ W(f'{p}.down_proj.weight').T
            print(f'  reference layer {i}: max |x| {x.abs().max():.1f}', flush=True)
        return rms(x, W('model.norm.weight'), eps) @ W('lm_head.weight').T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_dir', type=Path)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--decode', type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads((SRC / 'config.json').read_text())
    W = Weights(SRC, SPEC['source'], SPEC['revision'])
    ref = Reference(cfg, W, args.layers)
    session = ort.InferenceSession(str(args.onnx_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float16) for i in session.get_inputs() if i.name.startswith('past_key_values')}
    print(f'{len(past) // 2} of {args.layers} layers have a KV cache', flush=True)

    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 128000, args.tokens).tolist()
    total = 0
    for step, chunk in enumerate([ids] + [[int(t)] for t in rng.integers(1000, 128000, args.decode)]):
        with torch.no_grad():
            want = ref(chunk, total).numpy()
        total += len(chunk)
        out = session.run(None, {'input_ids': np.array([chunk], np.int64), 'attention_mask': np.ones((1, total), np.int64), **past})
        names = [o.name for o in session.get_outputs()]
        result = dict(zip(names, out))
        past = {n.replace('present', 'past_key_values'): result[n] for n in names if n.startswith('present')}
        got = result['logits'][0].astype(np.float32)
        for label, rows in (('last', slice(-1, None)), ('all', slice(None))):
            a, b = got[rows].ravel(), want[rows].ravel()
            cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            agree = float((got[rows].argmax(-1) == want[rows].argmax(-1)).mean())
            print(f'step {step} ({len(chunk)} tokens) {label}: cosine {cos:.5f}, same top token {agree:.3f}, max |diff| {np.abs(a - b).max():.3f}', flush=True)


if __name__ == '__main__':
    main()
