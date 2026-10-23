"""Checks export_flint.py's graph against a float32 PyTorch reference written from Flint's SGLang model code
(flint_sglang/model.py), on the first N layers: logits of a long prompt (past the 513-token sliding window), then of a
few tokens decoded against the cache. With 8-bit weights (export_flint.py --bits 8) the two should agree closely, which
checks the graph's structure; 4-bit shows the quantisation's effect.

    tools/export/.venv/bin/python tools/export/export_flint.py --layers 6 --bits 8 --out DIR
    tools/export/.venv/bin/python tools/export/check_flint.py DIR [--tokens 600]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from export_flint import REVISION, SOURCE, SRC
from onnx_export import Weights


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def swiglu(x: torch.Tensor, W: Weights, p: str) -> torch.Tensor:
    return (F.silu(x @ W(f'{p}.gate_proj.weight').T) * (x @ W(f'{p}.up_proj.weight').T)) @ W(f'{p}.down_proj.weight').T


class Reference:
    """Flint's first `layers` layers in float32, with a KV cache (positions continue from it)."""

    def __init__(self, cfg: dict, W: Weights, layers: int):
        self.cfg, self.W, self.layers = cfg, W, layers
        self.cache = [None] * layers

    def attention(self, i: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cfg, W, p = self.cfg, self.W, f'model.layers.{i}.self_attn'
        H, KV, D, eps = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['rms_norm_eps']
        S = x.shape[0]
        q = rms((x @ W(f'{p}.q_proj.weight').T).view(S, H, D), W(f'{p}.q_norm.weight'), eps)
        k = rms((x @ W(f'{p}.k_proj.weight').T).view(S, KV, D), W(f'{p}.k_norm.weight'), eps)
        v = (x @ W(f'{p}.v_proj.weight').T).view(S, KV, D)
        if i not in set(cfg['no_rope_layer_indices']):
            inv = 1.0 / (cfg['rope_theta'] ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
            ang = torch.outer(pos.double(), inv)
            cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None]
            sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if self.cache[i] is not None:
            k, v = torch.cat([self.cache[i][0], k]), torch.cat([self.cache[i][1], v])
        self.cache[i] = (k, v)
        T = k.shape[0]
        k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
        scores = torch.einsum('shd,thd->hst', q, k) / math.sqrt(D)  # [H, S, T]
        qpos, kpos = pos[:, None], torch.arange(T)[None, :]
        allowed = kpos <= qpos
        if cfg['layer_types'][i] == 'sliding_attention':
            allowed &= kpos > qpos - cfg['sliding_window']  # 513 tokens including the current one
        scores = scores.masked_fill(~allowed[None], float('-inf'))
        # Learned sinks: one more logit per head in the softmax denominator, with no value.
        sink = W(f'{p}.softmax_offset')[:, None, None].expand(H, S, 1)
        probs = torch.softmax(torch.cat([scores, sink], -1), -1)[..., :-1]
        out = torch.einsum('hst,thd->shd', probs, v).reshape(S, H * D)
        return out @ W(f'{p}.o_proj.weight').T

    def moe(self, i: int, x: torch.Tensor) -> torch.Tensor:
        cfg, W, p = self.cfg, self.W, f'model.layers.{i}.mlp'
        scores = torch.sigmoid(x @ W(f'{p}.gate.weight').T)
        top = torch.topk(scores, cfg['num_experts_per_tok'], dim=-1).indices
        weights = scores.gather(1, top)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(x)
        for e in top.unique().tolist():
            rows, slot = (top == e).nonzero(as_tuple=True)
            out[rows] += weights[rows, slot, None] * swiglu(x[rows], W, f'{p}.experts.{e}')
        return out + swiglu(x, W, f'{p}.shared_expert')

    def __call__(self, ids: list[int], start: int) -> torch.Tensor:
        cfg, W, eps = self.cfg, self.W, self.cfg['rms_norm_eps']
        pos = torch.arange(start, start + len(ids))
        x = W('model.embed_tokens.weight')[torch.tensor(ids)]
        dense = set(cfg['mlp_only_layers'])
        for i in range(self.layers):
            x = x + self.attention(i, rms(x, W(f'model.layers.{i}.input_layernorm.weight'), eps), pos)
            h = rms(x, W(f'model.layers.{i}.post_attention_layernorm.weight'), eps)
            x = x + (swiglu(h, W, f'model.layers.{i}.mlp') if i in dense else self.moe(i, h))
            print(f'  reference layer {i}: max |x| {x.abs().max():.1f}', flush=True)
        return rms(x, W('model.norm.weight'), eps) @ W('lm_head.weight').T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_dir', type=Path)
    parser.add_argument('--tokens', type=int, default=600)
    parser.add_argument('--decode', type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads((SRC / 'config.json').read_text())
    layers = json.loads((args.onnx_dir / 'genai_config.json').read_text())['model']['decoder']['num_hidden_layers']
    W = Weights(SRC, SOURCE, REVISION)
    ref = Reference(cfg, W, layers)
    session = ort.InferenceSession(str(args.onnx_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    kv, D = cfg['num_key_value_heads'], cfg['head_dim']
    past = {f'past_key_values.{i}.{k}': np.zeros((1, kv, 0, D), np.float16) for i in range(layers) for k in ('key', 'value')}

    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 140000, args.tokens).tolist()
    total = 0
    for step, chunk in enumerate([ids] + [[int(t)] for t in rng.integers(1000, 140000, args.decode)]):
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
        # Positions past the sliding window: a wrong window size shows up here first.
        if step == 0 and len(chunk) > cfg['sliding_window'] + 10:
            late = slice(cfg['sliding_window'] + 10, None)
            a, b = got[late].ravel(), want[late].ravel()
            print(f'  positions past the window: cosine {float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)):.5f}', flush=True)


if __name__ == '__main__':
    main()
