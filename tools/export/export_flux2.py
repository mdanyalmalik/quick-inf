"""Exports FLUX.2 klein 4B's transformer (Flux2Transformer2DModel) to ONNX for the image engine's 'flux2' pipeline, with
the same inputs and outputs as MarkShark2/flux2-klein-4b-onnx-webgpu-q4's transformer (whose text encoder and decoder
the app uses alongside it):

    x [1, image tokens, 128] float16, x_ids [1, image tokens, 4] float32 (0, row, column, 0), timesteps [1] float16
    (0–1), ctx [1, text tokens, 7680] float16, ctx_ids [1, text tokens, 4] float32 (0, 0, 0, position)
    → pred [1, image tokens, 128] float32 (the flow-matching velocity)

Linear layers are 4-bit MatMulNBits (round to nearest, block size 32); the small embedders, the timestep MLP and the
output projection stay float16 or float32. The residual streams, layer norms, modulation and rotary embedding are
float32; the projections and attention (MultiHeadAttention) run in float16. Weights stream from the checkpoint one
tensor at a time, so the export needs a few GB of memory.

    tools/export/.venv/bin/python tools/export/export_flux2.py [--bits 8] [--out DIR]
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from onnx import helper
from safetensors import safe_open

from onnx_export import F16, F32, I64, MS, Graph, Settings, Weights, matmul_nbits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'FLUX.2-klein-4B'
REPO = 'FLUX.2-klein-4B-ONNX-WebGPU'
SOURCE = 'black-forest-labs/FLUX.2-klein-4B'
REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
CHECKPOINT = 'transformer/diffusion_pytorch_model.safetensors'
import os
EIGHT_BIT = [p for p in os.environ.get('EIGHT_BIT', '').split(',') if p]


def weights() -> Weights:
    """The checkpoint is one file; Weights reads through an index, so one is written for it."""
    path = SRC / CHECKPOINT
    if not path.exists():
        hf_hub_download(SOURCE, CHECKPOINT, revision=REVISION, local_dir=SRC)
    with safe_open(str(path), framework='pt') as f:
        index = {k: CHECKPOINT for k in f.keys()}
    (SRC / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    return Weights(SRC, SOURCE, REVISION)


class Flux2:
    def __init__(self, g: Graph, W: Weights, cfg: dict):
        self.g, self.W, self.cfg = g, W, cfg
        self.heads = cfg['num_attention_heads']
        self.head_dim = cfg['attention_head_dim']
        self.dim = self.heads * self.head_dim
        self.eps = cfg['eps']
        self.ones = g.init('ln.ones', np.ones(self.dim, np.float32), F32)

    def f16(self, x: str, name: str) -> str:
        return self.g.node('Cast', [x], f'{name}/Cast_f16', to=F16)

    def f32(self, x: str, name: str) -> str:
        return self.g.node('Cast', [x], f'{name}/Cast_f32', to=F32)

    def silu(self, x: str, name: str) -> str:
        return self.g.node('Mul', [x, self.g.node('Sigmoid', [x], f'{name}/Sigmoid')], f'{name}/Mul')

    def quantized(self, x: str, w: torch.Tensor, key: str, name: str) -> str:
        """x @ w.T as MatMulNBits, at 8 bits for the layers in EIGHT_BIT, float16 in and out."""
        bits = Settings.bits
        if any(p in key for p in EIGHT_BIT):
            Settings.bits = 8
        try:
            return matmul_nbits(self.g, x, w, name)
        finally:
            Settings.bits = bits

    def linear(self, x: str, key: str, name: str) -> str:
        return self.quantized(x, self.W(key, native=True), key, name)

    def linear_fused(self, x: str, keys: list[str], name: str) -> str:
        """Several linear layers with the same input as one MatMulNBits (their outputs side by side)."""
        return self.quantized(x, torch.cat([self.W(k, native=True) for k in keys]), keys[0], name)

    def split(self, x: str, sizes: list[int], name: str, axis=-1) -> list[str]:
        outs = [f'{name}/output_{i}' for i in range(len(sizes))]
        self.g.node('Split', [x, self.g.const(f'{name}/sizes', sizes)], name, outputs=outs, axis=axis)
        return outs

    def modulate(self, x: str, shift: str, scale: str, name: str) -> str:
        """LayerNorm(x) * (1 + scale) + shift, float32, then float16 for the projections."""
        g = self.g
        n = g.node('LayerNormalization', [x, self.ones], f'{name}/LayerNorm', axis=-1, epsilon=self.eps)
        one = g.const(f'{name}/one', 1.0, F32)
        n = g.node('Mul', [n, g.node('Add', [scale, one], f'{name}/scale')], f'{name}/Mul')
        return self.f16(g.node('Add', [n, shift], f'{name}/Add'), name)

    def gated(self, residual: str, gate: str, y: str, name: str) -> str:
        """residual + gate * y, float32."""
        g = self.g
        return g.node('Add', [residual, g.node('Mul', [gate, self.f32(y, name)], f'{name}/Mul')], f'{name}/Add')

    def modulation(self, temb_act: str, key: str, sets: int, name: str) -> list[str]:
        """The modulation vectors (shift, scale, gate per set) as [1, 1, dim] float32."""
        g = self.g
        m = self.f32(self.linear(temb_act, key, f'{name}/MatMul'), name)
        m = g.node('Reshape', [m, g.const(f'{name}/shape', [1, 1, 3 * sets * self.dim])], f'{name}/Reshape')
        return self.split(m, [self.dim] * 3 * sets, f'{name}/Split')

    def qk_norm(self, x: str, key: str, name: str) -> str:
        """Per-head RMSNorm on a [1, S, heads * head_dim] float16 tensor, as [1, S, heads, head_dim]."""
        g = self.g
        x = g.node('Reshape', [x, g.const(f'{name}/shape', [1, -1, self.heads, self.head_dim])], f'{name}/Reshape')
        w = g.init(f'{name}.weight', self.W(key).numpy().astype(np.float16), F16)
        return g.node('SimplifiedLayerNormalization', [x, w], name, axis=-1, epsilon=self.eps, stash_type=1)

    def rope(self, x: str, cos: str, sin: str, name: str) -> str:
        """Rotary embedding on adjacent pairs of [1, S, heads, head_dim] (float32 maths), back to [1, S, dim] float16."""
        g = self.g
        x = g.node('Reshape', [self.f32(x, name), g.const(f'{name}/pairs', [1, 0, self.heads, self.head_dim // 2, 2])], f'{name}/Reshape', allowzero=0)
        a, b = self.split(x, [1, 1], f'{name}/Split')
        ra = g.node('Sub', [g.node('Mul', [a, cos], f'{name}/Mul_ac'), g.node('Mul', [b, sin], f'{name}/Mul_bs')], f'{name}/Sub')
        rb = g.node('Add', [g.node('Mul', [b, cos], f'{name}/Mul_bc'), g.node('Mul', [a, sin], f'{name}/Mul_as')], f'{name}/Add')
        out = g.node('Concat', [ra, rb], f'{name}/Concat', axis=-1)
        out = g.node('Reshape', [out, g.const(f'{name}/flat', [1, 0, self.dim])], f'{name}/Reshape_flat', allowzero=0)
        return self.f16(out, f'{name}/out')

    def attention(self, q: str, k: str, v: str, name: str) -> str:
        return self.g.node('MultiHeadAttention', [q, k, v], name, domain=MS, num_heads=self.heads)

    def double_block(self, i: int, img: str, txt: str, mods: dict, rope: tuple, sizes: str):
        p = f'transformer_blocks.{i}'
        name = f'/double.{i}'
        g = self.g
        shift_img, scale_img, gate_img, shift_img2, scale_img2, gate_img2 = mods['img']
        shift_txt, scale_txt, gate_txt, shift_txt2, scale_txt2, gate_txt2 = mods['txt']
        d = self.dim

        ni = self.modulate(img, shift_img, scale_img, f'{name}/norm1')
        nt = self.modulate(txt, shift_txt, scale_txt, f'{name}/norm1_context')
        qi, ki, vi = self.split(self.linear_fused(ni, [f'{p}.attn.to_q.weight', f'{p}.attn.to_k.weight', f'{p}.attn.to_v.weight'], f'{name}/qkv'), [d] * 3, f'{name}/qkv/Split')
        qt, kt, vt = self.split(
            self.linear_fused(nt, [f'{p}.attn.add_q_proj.weight', f'{p}.attn.add_k_proj.weight', f'{p}.attn.add_v_proj.weight'], f'{name}/added_qkv'),
            [d] * 3, f'{name}/added_qkv/Split')
        qi = self.qk_norm(qi, f'{p}.attn.norm_q.weight', f'{name}/norm_q')
        ki = self.qk_norm(ki, f'{p}.attn.norm_k.weight', f'{name}/norm_k')
        qt = self.qk_norm(qt, f'{p}.attn.norm_added_q.weight', f'{name}/norm_added_q')
        kt = self.qk_norm(kt, f'{p}.attn.norm_added_k.weight', f'{name}/norm_added_k')
        # Text first, then image, as the rotary embedding's positions.
        q = self.rope(g.node('Concat', [qt, qi], f'{name}/q/Concat', axis=1), *rope, f'{name}/q/rope')
        k = self.rope(g.node('Concat', [kt, ki], f'{name}/k/Concat', axis=1), *rope, f'{name}/k/rope')
        v = g.node('Concat', [vt, vi], f'{name}/v/Concat', axis=1)
        a = self.attention(q, k, v, f'{name}/attn')
        at, ai = [f'{name}/attn/Split/output_{j}' for j in range(2)]
        g.node('Split', [a, sizes], f'{name}/attn/Split', outputs=[at, ai], axis=1)

        img = self.gated(img, gate_img, self.linear(ai, f'{p}.attn.to_out.0.weight', f'{name}/to_out'), f'{name}/attn_residual')
        txt = self.gated(txt, gate_txt, self.linear(at, f'{p}.attn.to_add_out.weight', f'{name}/to_add_out'), f'{name}/attn_residual_context')
        img = self.gated(img, gate_img2, self.ff(self.modulate(img, shift_img2, scale_img2, f'{name}/norm2'), f'{p}.ff', f'{name}/ff'), f'{name}/ff_residual')
        txt = self.gated(txt, gate_txt2, self.ff(self.modulate(txt, shift_txt2, scale_txt2, f'{name}/norm2_context'), f'{p}.ff_context', f'{name}/ff_context'), f'{name}/ff_residual_context')
        return img, txt

    def swiglu(self, x: str, name: str) -> str:
        """silu(first half) * second half; float32 maths."""
        g = self.g
        a, b = self.split(self.f32(x, name), [self.mlp, self.mlp], f'{name}/Split')
        return self.f16(g.node('Mul', [self.silu(a, f'{name}/silu'), b], f'{name}/Mul'), f'{name}/out')

    def ff(self, x: str, p: str, name: str) -> str:
        h = self.swiglu(self.linear(x, f'{p}.linear_in.weight', f'{name}/linear_in'), f'{name}/act')
        return self.linear(h, f'{p}.linear_out.weight', f'{name}/linear_out')

    def single_block(self, i: int, h: str, mods: list, rope: tuple) -> str:
        p = f'single_transformer_blocks.{i}'
        name = f'/single.{i}'
        g = self.g
        shift, scale, gate = mods
        d = self.dim
        n = self.modulate(h, shift, scale, f'{name}/norm')
        q, k, v, mlp = self.split(self.linear(n, f'{p}.attn.to_qkv_mlp_proj.weight', f'{name}/to_qkv_mlp_proj'), [d, d, d, 2 * self.mlp], f'{name}/Split')
        q = self.rope(self.qk_norm(q, f'{p}.attn.norm_q.weight', f'{name}/norm_q'), *rope, f'{name}/q/rope')
        k = self.rope(self.qk_norm(k, f'{p}.attn.norm_k.weight', f'{name}/norm_k'), *rope, f'{name}/k/rope')
        a = self.attention(q, k, v, f'{name}/attn')
        cat = g.node('Concat', [a, self.swiglu(mlp, f'{name}/act')], f'{name}/Concat', axis=-1)
        return self.gated(h, gate, self.linear(cat, f'{p}.attn.to_out.weight', f'{name}/to_out'), f'{name}/residual')

    def rotary(self) -> tuple:
        """cos and sin of every position's angles, [1, S, 1, head_dim / 2, 1] float32: each of the 4 id axes turns
        32 of the head's dimensions (16 pairs) at frequencies theta^(-2j/32)."""
        g = self.g
        axes = self.cfg['axes_dims_rope']
        theta = self.cfg['rope_theta']
        table = np.zeros((len(axes), self.head_dim // 2), np.float64)
        col = 0
        for a, dim in enumerate(axes):
            table[a, col:col + dim // 2] = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
            col += dim // 2
        ids = g.node('Concat', ['ctx_ids', 'x_ids'], '/rope/ids', axis=1)
        angles = g.node('MatMul', [ids, g.init('rope.freqs', table.astype(np.float32), F32)], '/rope/MatMul')
        angles = g.node('Reshape', [angles, g.const('/rope/shape', [1, -1, 1, self.head_dim // 2, 1])], '/rope/Reshape')
        return g.node('Cos', [angles], '/rope/Cos'), g.node('Sin', [angles], '/rope/Sin')

    def timestep(self) -> str:
        """The timestep embedding (sinusoidal, cos first, of t × 1000; then Linear, SiLU, Linear), [1, dim] float32."""
        g = self.g
        half = self.cfg['timestep_guidance_channels'] // 2
        freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half).astype(np.float32)
        t = g.node('Mul', [self.f32('timesteps', '/time'), g.const('/time/scale', 1000.0, F32)], '/time/Mul')
        t = g.node('Unsqueeze', [t, g.const('/time/axes', [1])], '/time/Unsqueeze')
        args = g.node('Mul', [t, g.init('time.freqs', freqs, F32)], '/time/args')
        emb = g.node('Concat', [g.node('Cos', [args], '/time/Cos'), g.node('Sin', [args], '/time/Sin')], '/time/Concat', axis=-1)
        p = 'time_guidance_embed.timestep_embedder'
        w1 = g.init(f'{p}.linear_1.weight_T', self.W(f'{p}.linear_1.weight').T.contiguous().numpy(), F32)
        w2 = g.init(f'{p}.linear_2.weight_T', self.W(f'{p}.linear_2.weight').T.contiguous().numpy(), F32)
        h = self.silu(g.node('MatMul', [emb, w1], '/time/linear_1'), '/time/silu')
        return g.node('MatMul', [h, w2], '/time/linear_2')

    def build(self):
        g, cfg, W = self.g, self.cfg, self.W
        self.mlp = int(self.dim * cfg['mlp_ratio'])
        g.inputs = [
            helper.make_tensor_value_info('x', F16, [1, 'image_seq', cfg['in_channels']]),
            helper.make_tensor_value_info('x_ids', F32, [1, 'image_seq', 4]),
            helper.make_tensor_value_info('timesteps', F16, [1]),
            helper.make_tensor_value_info('ctx', F16, [1, 'text_seq', cfg['joint_attention_dim']]),
            helper.make_tensor_value_info('ctx_ids', F32, [1, 'text_seq', 4]),
        ]
        g.outputs = [helper.make_tensor_value_info('pred', F32, [1, 'image_seq', cfg['out_channels'] or cfg['in_channels']])]

        temb = self.timestep()
        temb_act = self.f16(self.silu(temb, '/time/act'), '/time/act')
        mods = {
            'img': self.modulation(temb_act, 'double_stream_modulation_img.linear.weight', 2, '/mod_img'),
            'txt': self.modulation(temb_act, 'double_stream_modulation_txt.linear.weight', 2, '/mod_txt'),
        }
        single_mods = self.modulation(temb_act, 'single_stream_modulation.linear.weight', 1, '/mod_single')
        rope = self.rotary()

        x_w = g.init('x_embedder.weight_T', W('x_embedder.weight').T.contiguous().numpy().astype(np.float16), F16)
        img = self.f32(g.node('MatMul', ['x', x_w], '/x_embedder/MatMul'), '/x_embedder')
        txt = self.f32(self.linear('ctx', 'context_embedder.weight', '/context_embedder'), '/context_embedder')

        # Split sizes for the joint sequence: [text tokens, image tokens].
        lengths = [g.node('Slice', [g.node('Shape', [t], f'/sizes/Shape_{t}'), g.const(f'/sizes/{t}/start', [1]), g.const(f'/sizes/{t}/end', [2])], f'/sizes/Slice_{t}') for t in ('ctx', 'x')]
        sizes = g.node('Concat', lengths, '/sizes/Concat', axis=0)

        for i in range(cfg['num_layers']):
            print(f'double block {i}', flush=True)
            img, txt = self.double_block(i, img, txt, mods, rope, sizes)
        h = g.node('Concat', [txt, img], '/joint/Concat', axis=1)
        for i in range(cfg['num_single_layers']):
            print(f'single block {i}', flush=True)
            h = self.single_block(i, h, single_mods, rope)
        _, img = [f'/joint/Split/output_{j}' for j in range(2)]
        g.node('Split', [h, sizes], '/joint/Split', outputs=['/joint/Split/output_0', img], axis=1)

        # AdaLayerNormContinuous: (scale, shift) from silu(temb).
        n = self.f32(self.linear(temb_act, 'norm_out.linear.weight', '/norm_out/linear'), '/norm_out')
        n = g.node('Reshape', [n, g.const('/norm_out/shape', [1, 1, 2 * self.dim])], '/norm_out/Reshape')
        scale, shift = self.split(n, [self.dim, self.dim], '/norm_out/Split')
        out = self.modulate(img, shift, scale, '/norm_out')
        out_w = g.init('proj_out.weight_T', W('proj_out.weight').T.contiguous().numpy().astype(np.float16), F16)
        out = g.node('MatMul', [out, out_w], '/proj_out/MatMul')
        g.nodes.append(helper.make_node('Cast', [out], ['pred'], name='/proj_out/Cast', to=F32))


def build(out: Path):
    cfg = json.loads((SRC / 'transformer' / 'config.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'transformer.onnx.data')
    Flux2(g, weights(), cfg).build()
    g.close()
    graph = helper.make_graph(g.nodes, 'FLUX.2-klein-4B transformer', g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'transformer.onnx'))
    print('saved', out, g.offset, flush=True)


def finish(out: Path):
    shutil.copy(SRC / 'transformer' / 'config.json', out / 'config.json')
    shutil.copy(SRC / 'LICENSE.md', out / 'LICENSE.md')
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = """---
license: apache-2.0
base_model:
- black-forest-labs/FLUX.2-klein-4B
tags:
- onnx
- webgpu
- text-to-image
- quick-inference
---

# FLUX.2-klein-4B-ONNX-WebGPU

The transformer of [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
converted to ONNX for running in a web browser with ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- The transformer only: one ONNX graph taking the noisy latents (`x`, one token per 16 × 16 pixels, 128 values each),
  their positions (`x_ids`), the timestep (0–1), the text features (`ctx`, three Qwen3 layers side by side) and their
  positions (`ctx_ids`), and returning the flow-matching velocity (`pred`). Its inputs and outputs match
  [MarkShark2/flux2-klein-4b-onnx-webgpu-q4](https://huggingface.co/MarkShark2/flux2-klein-4b-onnx-webgpu-q4), whose
  text encoder and VAE decoder go with it.
- The linear layers in the blocks are quantised to 4 bits (round to nearest, block size 32); attention and the
  projections run in float16, the residual streams, norms, modulation and rotary embedding in float32.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

Apache 2.0, as the original (LICENSE.md).
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    parser.add_argument('--no-finish', action='store_true')
    args = parser.parse_args()
    Settings.bits = args.bits
    Settings.block = args.block
    build(args.out)
    if not args.no_finish:
        finish(args.out)
