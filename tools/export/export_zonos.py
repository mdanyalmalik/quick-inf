"""Zonos v0.1 (transformer) split into graphs for ONNX Runtime Web.

Zonos is a 26-layer transformer over the 9 codebooks of Descript's DAC 44.1 kHz codec (delay pattern, 86 frames per
second), conditioned on a prefix: eSpeak phonemes of the text, a speaker embedding, emotion, maximum frequency,
pitch spread, speaking rate and language. Graphs (float; quantize with quantize_int4.py):

- `phoneme_embed`: eSpeak phoneme ids [B, P] -> rows [B, P, 2048] (the phoneme conditioner).
- `prefix`: the conditioners' rows concatenated [B, L, 2048] -> prefix [B, L, 2048] (projection + LayerNorm, applied
  per position; the non-phoneme rows depend only on the settings and can be computed once).
- `embed_codes`: codes [B, 9, S] -> the sum of the 9 codebook embeddings [B, S, 2048].
- `backbone`: inputs_embeds [B, S, 2048] + past_key_values.{i}.key/value [B, 4, past, 128] -> logits of the last
  position for the 9 codebooks [B, 9, 1025] + present.*. Positions continue from `past` (RoPE, interleaved pairs).

    zonos-venv/bin/python tools/export/export_zonos.py phoneme_embed|prefix|embed_codes|backbone SRC OUT
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


def load(src: Path):
    from zonos.model import Zonos
    m = Zonos.from_local(str(src / 'config.json'), str(src / 'model.safetensors'), device='cpu')
    return m.float().eval()


def export(module, args, out: Path, inputs, outputs, dynamic=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, args, str(out), input_names=inputs, output_names=outputs, opset_version=17,
                          dynamic_axes=dynamic or {}, dynamo=False)
    print('exported', out.name)


def rope(x, pos, base=10000.0):
    """Interleaved-pair RoPE as zonos' apply_rotary_emb, in real arithmetic. x [B, S, H, D], pos [S]."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    f = pos.float()[:, None] * inv[None]
    c, s = torch.cos(f)[None, :, None], torch.sin(f)[None, :, None]
    x2 = x.float().reshape(*x.shape[:-1], -1, 2)
    a, b = x2[..., 0], x2[..., 1]
    return torch.stack([a * c - b * s, b * c + a * s], -1).flatten(3)


class Backbone(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.bb = m.backbone
        self.heads = m.heads

    def forward(self, x, *past):
        b, s = x.shape[0], x.shape[1]
        p = past[0].shape[2]
        pos = torch.arange(s) + p
        i = torch.arange(s).unsqueeze(1) + p
        j = torch.cat([torch.arange(p), torch.arange(s) + p]).unsqueeze(0)
        mask = torch.where(j <= i, 0.0, float('-inf'))[None, None]
        presents = []
        h = x
        for li, layer in enumerate(self.bb.layers):
            a = layer.mixer
            y = layer.norm(h)
            q_size, kv_size = a.num_heads * a.head_dim, a.num_heads_kv * a.head_dim
            q, k, v = a.in_proj(y).split([q_size, kv_size, kv_size], dim=-1)
            q = rope(q.view(b, s, a.num_heads, a.head_dim), pos).transpose(1, 2)
            k = rope(k.view(b, s, a.num_heads_kv, a.head_dim), pos).transpose(1, 2)
            v = v.view(b, s, a.num_heads_kv, a.head_dim).transpose(1, 2)
            k = torch.cat([past[2 * li], k], dim=2)
            v = torch.cat([past[2 * li + 1], v], dim=2)
            presents += [k, v]
            rep = a.num_heads // a.num_heads_kv
            kk, vv = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            att = torch.softmax(q @ kk.transpose(-1, -2) / (a.head_dim ** 0.5) + mask, dim=-1) @ vv
            h = h + a.out_proj(att.transpose(1, 2).reshape(b, s, q_size))
            h = h + layer.mlp(layer.norm2(h))
        last = self.bb.norm_f(h)[:, -1]
        logits = torch.stack([hd(last) for hd in self.heads], dim=1)
        return (logits, *presents)


def backbone(src, out):
    m = load(src)
    w = Backbone(m).eval()
    n = len(m.backbone.layers)
    kvh, hd = m.config.backbone.attn_cfg['num_heads_kv'], m.config.backbone.d_model // m.config.backbone.attn_cfg['num_heads']
    # Check against the model's own cached forward: prefill 6 + 1 step.
    with torch.no_grad():
        x = torch.randn(2, 7, m.config.backbone.d_model) * 0.1
        ip = m.setup_cache(batch_size=2, max_seqlen=16, dtype=torch.float32)
        ref1 = m.apply_heads(m.backbone(x[:, :6], ip)[:, -1:]).squeeze(2)
        ip.seqlen_offset += 6
        ip.lengths_per_sample[:] += 6
        ref2 = m.apply_heads(m.backbone(x[:, 6:], ip)[:, -1:]).squeeze(2)
        empty = [torch.zeros(2, kvh, 0, hd) for _ in range(2 * n)]
        o1, *pr = w(x[:, :6], *empty)
        o2, *_ = w(x[:, 6:], *pr)
        print('backbone vs zonos: prefill', float((o1 - ref1).abs().max()), 'step', float((o2 - ref2).abs().max()))
    names_in = ['inputs_embeds'] + [f'past_key_values.{i}.{t}' for i in range(n) for t in ('key', 'value')]
    names_out = ['logits'] + [f'present.{i}.{t}' for i in range(n) for t in ('key', 'value')]
    dyn = {'inputs_embeds': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch'}}
    dyn.update({k: {0: 'batch', 2: 'past'} for k in names_in[1:]})
    dyn.update({k: {0: 'batch', 2: 'total'} for k in names_out[1:]})
    export(w, (x[:, :6], *[torch.zeros(2, kvh, 3, hd) for _ in range(2 * n)]), out, names_in, names_out, dyn)


def embed_codes(src, out):
    m = load(src)

    class E(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = m.embeddings

        def forward(self, codes):
            return sum(emb(codes[:, i]) for i, emb in enumerate(self.e))
    export(E().eval(), (torch.randint(0, 1025, (2, 9, 3)),), out, ['codes'], ['embeds'],
           {'codes': {0: 'batch', 2: 'seq'}, 'embeds': {0: 'batch', 1: 'seq'}})


def phoneme_embed(src, out):
    from export_qwen3_tts import gather_graph
    m = load(src)
    c = next(c for c in m.prefix_conditioner.conditioners if c.name == 'espeak')
    out.parent.mkdir(parents=True, exist_ok=True)
    gather_graph(out, {'rows': c.phoneme_embedder.weight.detach().numpy()}, {})


def prefix(src, out):
    m = load(src)
    pc = m.prefix_conditioner

    class P(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pc = pc

        def forward(self, rows):
            return self.pc.norm(self.pc.project(rows))
    export(P().eval(), (torch.randn(2, 9, m.config.backbone.d_model),), out, ['rows'], ['prefix'],
           {'rows': {0: 'batch', 1: 'len'}, 'prefix': {0: 'batch', 1: 'len'}})


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    cmd, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {'backbone': backbone, 'embed_codes': embed_codes, 'phoneme_embed': phoneme_embed, 'prefix': prefix}[cmd](src, out)


if __name__ == '__main__':
    main()
