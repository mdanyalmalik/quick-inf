"""Exports FLUX 3 Image Klein (black-forest-labs/flux-3-image-klein, the distilled flow-map checkpoint) to ONNX for the
image engine's 'flux3' pipeline: text-to-image only (no reference images). Three graphs:

    text_encoder.onnx: Qwen3-VL-4B's language model, its first 32 layers: input_ids, attention_mask, position_ids [1, L]
      int64 → ctx [1, L, 20480] (the residual stream after layers 4, 8, …, 32 side by side, as the reference's
      text_encoder.py stacks them). The prompt is padded to a multiple of 80 tokens; padded positions take position 1
      and are masked as keys, as in the reference. Linear layers 4-bit; residual stream and norms float32.
    transformer.onnx: the DiT (dit.py): x [1, image tokens, 128] float32, x_ids [1, image tokens, 4] float32
      (0, row / 2, column / 2, 0), t and t_ref [1] float32 (the time now and the time the flow-map step jumps to), ctx,
      ctx_ids [1, L, 4] float32 (0, 0, 0, position) → pred [1, image tokens, 128] float32 (x(t_ref) = x + (t_ref − t)
      × pred). Five blocks per mode (image tokens attend to image tokens, text to text), then 28 blocks over [image,
      text], each token modulated by its own mode's vectors. Linear layers 4-bit, residual streams, norms, modulation
      and rotary embedding float32, attention float16.
    decoder.onnx: z [1, 128, h, w] float32 (the latent tokens on their grid) → image [1, 3, 16h, 16w] in [−1, 1]: the
      batch-norm statistics undone, 2 × 2 unpatchified to 32 channels, then the VAE decoder, float32.

    tools/export/.venv/bin/python tools/export/export_flux3.py [text_encoder] [transformer] [decoder] [--out DIR]
    tools/export/.venv/bin/python tools/export/export_flux3.py --check text_encoder|transformer|decoder
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper

from onnx_export import F16, F32, I64, MS, Graph, Settings, Weights, matmul_nbits, to_float16

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'flux-3-image-klein'
REFERENCE_CODE = HERE / '.cache' / 'flux3-ref' / 'src'
REPO = 'FLUX-3-Image-Klein-ONNX-WebGPU'
SOURCE = 'black-forest-labs/flux-3-image-klein'
REVISION = '8454017f9ebaada30c998fccd9f67a80c7580f68'

TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128
DEPTH = 5
DEPTH_SINGLE = 28
MLP = int(HIDDEN * 3.0)
AXES = (32, 32, 32, 32)
THETA = 10000
EPS = 1e-6
TXT_IN_SCALE = 16.0
OUT_SCALE = 32.0


def checkpoint() -> Weights:
    """The DiT checkpoint (model.safetensors, one file; Weights reads through an index, so one is written for it)."""
    from safetensors import safe_open

    index = SRC / 'model.safetensors.index.json'
    if not index.exists():
        with safe_open(str(SRC / 'model.safetensors'), framework='pt') as f:
            index.write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in f.keys()}}))
    return Weights(SRC, SOURCE, REVISION)


class Builder:
    def __init__(self, g: Graph, W: Weights):
        self.g, self.W = g, W
        self.stored: set[str] = set()

    def f16(self, x: str, name: str) -> str:
        return self.g.node('Cast', [x], f'{name}/Cast_f16', to=F16)

    def f32(self, x: str, name: str) -> str:
        return self.g.node('Cast', [x], f'{name}/Cast_f32', to=F32)

    def silu(self, x: str, name: str) -> str:
        return self.g.node('Mul', [x, self.g.node('Sigmoid', [x], f'{name}/Sigmoid')], f'{name}/Mul')

    def split(self, x: str, sizes: list[int], name: str, axis=-1) -> list[str]:
        outs = [f'{name}/output_{i}' for i in range(len(sizes))]
        self.g.node('Split', [x, self.g.const(f'{name}/sizes', sizes)], name, outputs=outs, axis=axis)
        return outs

    def linear(self, x: str, keys: list[str], name: str, dim=0) -> str:
        """x (float16) @ [weights side by side].T as one 4-bit MatMulNBits; `dim` 1 joins them along the input."""
        return matmul_nbits(self.g, x, torch.cat([self.W(k, native=True) for k in keys], dim=dim), name)

    def float_matmul(self, x: str, key: str, name: str, dtype=F32) -> str:
        """x @ w.T with w kept in float (small layers); a weight used twice is stored once."""
        np_type = np.float32 if dtype == F32 else np.float16
        w = f'{key}_T'
        if w not in self.stored:
            self.g.init(w, self.W(key).T.contiguous().numpy().astype(np_type), dtype)
            self.stored.add(w)
        return self.g.node('MatMul', [x, w], name)


# ---- Text encoder ----------------------------------------------------------------------------------------------


class TextEncoder(Builder):
    """Qwen3-VL-4B's language model (config text_config), layers 0–31, residual stream float32."""

    def __init__(self, g: Graph, W: Weights, cfg: dict):
        super().__init__(g, W)
        self.cfg = cfg
        self.heads, self.kv_heads, self.head_dim = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
        self.eps = cfg['rms_norm_eps']
        self.prefix = 'model.language_model'

    def rms(self, x: str, key: str, name: str, out16=True) -> str:
        """RMSNorm of float32 x with the stored weight (float32 maths); float16 out for the projections."""
        w = self.g.init(f'{key}', self.W(key).numpy().astype(np.float32), F32)
        n = self.g.node('SimplifiedLayerNormalization', [x, w], name, axis=-1, epsilon=self.eps, stash_type=1)
        return self.f16(n, name) if out16 else n

    def rotary(self) -> tuple[str, str]:
        """cos and sin [1, L, 1, head_dim] of each position (rotate-half layout: the 64 frequencies twice). Qwen3-VL's
        interleaved multimodal sections all take the same position for text, which is the ordinary rotary embedding."""
        g = self.g
        inv = 1.0 / self.cfg['rope_theta'] ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim)
        pos = g.node('Cast', ['position_ids'], '/rope/Cast', to=F32)
        pos = g.node('Unsqueeze', [pos, g.const('/rope/axes', [2])], '/rope/Unsqueeze')
        angles = g.node('Mul', [pos, g.init('rope.inv_freq', np.concatenate([inv, inv]).astype(np.float32), F32)], '/rope/Mul')
        angles = g.node('Unsqueeze', [angles, g.const('/rope/head_axis', [2])], '/rope/Unsqueeze_head')
        return g.node('Cos', [angles], '/rope/Cos'), g.node('Sin', [angles], '/rope/Sin')

    def rope(self, x: str, heads: int, cos: str, sin: str, name: str) -> str:
        """x [1, L, heads, head_dim] float32 → rotated (x cos + rotate_half(x) sin)."""
        g = self.g
        half = self.head_dim // 2
        a, b = self.split(x, [half, half], f'{name}/Split')
        rotated = g.node('Concat', [g.node('Neg', [b], f'{name}/Neg'), a], f'{name}/Concat', axis=-1)
        return g.node('Add', [g.node('Mul', [x, cos], f'{name}/Mul_cos'), g.node('Mul', [rotated, sin], f'{name}/Mul_sin')], f'{name}/Add')

    def head_norm(self, x: str, heads: int, key: str, name: str) -> str:
        g = self.g
        x = g.node('Reshape', [self.f32(x, name), g.const(f'{name}/shape', [1, -1, heads, self.head_dim])], f'{name}/Reshape')
        return self.rms(x, key, f'{name}/norm', out16=False)

    def attention_bias(self) -> str:
        """[1, 1, L, L] float16: 0 where query i may see key j (j ≤ i and j not padding), −65504 elsewhere."""
        g = self.g
        n = g.node('Gather', [g.node('Shape', ['input_ids'], '/mask/Shape'), g.const('/mask/index', 1)], '/mask/Gather', axis=0)
        r = g.node('Range', [g.const('/mask/zero', 0), n, g.const('/mask/one', 1)], '/mask/Range')
        rows = g.node('Unsqueeze', [r, g.const('/mask/rows_axes', [1])], '/mask/rows')
        cols = g.node('Unsqueeze', [r, g.const('/mask/cols_axes', [0])], '/mask/cols')
        future = g.node('Greater', [cols, rows], '/mask/future')
        padding = g.node('Equal', ['attention_mask', g.const('/mask/zero64', [0])], '/mask/padding')
        hidden = g.node('Or', [future, padding], '/mask/Or')
        bias = g.node('Where', [hidden, g.const('/mask/neg', -65504.0, F16), g.const('/mask/ok', 0.0, F16)], '/mask/Where')
        return g.node('Unsqueeze', [bias, g.const('/mask/bias_axes', [0, 1])], '/mask/bias')

    def repeat_kv(self, x: str, name: str) -> str:
        """[1, L, kv_heads, head_dim] → [1, L, heads × head_dim], each key head shared by heads / kv_heads query heads."""
        g = self.g
        group = self.heads // self.kv_heads
        x = g.node('Unsqueeze', [x, g.const(f'{name}/axes', [3])], f'{name}/Unsqueeze')
        shape = g.node('Concat', [g.node('Slice', [g.node('Shape', [x], f'{name}/Shape'), g.const(f'{name}/s0', [0]), g.const(f'{name}/s1', [2])], f'{name}/Slice'),
                                  g.const(f'{name}/rest', [self.kv_heads, group, self.head_dim])], f'{name}/Concat', axis=0)
        x = g.node('Expand', [x, shape], f'{name}/Expand')
        return g.node('Reshape', [x, g.const(f'{name}/flat', [1, -1, self.heads * self.head_dim])], f'{name}/Reshape')

    def layer(self, i: int, h: str, rope: tuple[str, str], bias: str) -> str:
        g = self.g
        p = f'{self.prefix}.layers.{i}'
        name = f'/text/layers.{i}'
        d, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        n = self.rms(h, f'{p}.input_layernorm.weight', f'{name}/input_layernorm')
        q, k, v = self.split(self.linear(n, [f'{p}.self_attn.{x}_proj.weight' for x in 'qkv'], f'{name}/qkv'), [d, kv, kv], f'{name}/qkv/Split')
        q = self.rope(self.head_norm(q, self.heads, f'{p}.self_attn.q_norm.weight', f'{name}/q_norm'), self.heads, *rope, f'{name}/q/rope')
        k = self.rope(self.head_norm(k, self.kv_heads, f'{p}.self_attn.k_norm.weight', f'{name}/k_norm'), self.kv_heads, *rope, f'{name}/k/rope')
        q = g.node('Reshape', [self.f16(q, f'{name}/q'), g.const(f'{name}/q/flat', [1, -1, d])], f'{name}/q/Reshape')
        k = self.repeat_kv(self.f16(k, f'{name}/k'), f'{name}/k/repeat')
        v = g.node('Reshape', [v, g.const(f'{name}/v/heads', [1, -1, self.kv_heads, self.head_dim])], f'{name}/v/Reshape')
        v = self.repeat_kv(v, f'{name}/v/repeat')
        a = g.node('MultiHeadAttention', [q, k, v, '', '', bias], f'{name}/attn', domain=MS, num_heads=self.heads)
        h = g.node('Add', [h, self.f32(self.linear(a, [f'{p}.self_attn.o_proj.weight'], f'{name}/o_proj'), f'{name}/o')], f'{name}/attn_residual')
        n = self.rms(h, f'{p}.post_attention_layernorm.weight', f'{name}/post_attention_layernorm')
        inter = self.cfg['intermediate_size']
        gate, up = self.split(self.f32(self.linear(n, [f'{p}.mlp.gate_proj.weight', f'{p}.mlp.up_proj.weight'], f'{name}/gate_up'), f'{name}/gate_up'),
                              [inter, inter], f'{name}/gate_up/Split')
        act = self.f16(g.node('Mul', [self.silu(gate, f'{name}/silu'), up], f'{name}/act'), f'{name}/act')
        return g.node('Add', [h, self.f32(self.linear(act, [f'{p}.mlp.down_proj.weight'], f'{name}/down'), f'{name}/down')], f'{name}/mlp_residual')

    def build(self, layers: int = max(TEXT_LAYERS)):
        g = self.g
        g.inputs = [helper.make_tensor_value_info(n, I64, [1, 'text_seq']) for n in ('input_ids', 'attention_mask', 'position_ids')]
        g.outputs = [helper.make_tensor_value_info('ctx', F16, [1, 'text_seq', len(TEXT_LAYERS) * self.cfg['hidden_size']])]
        emb = g.init('embed_tokens', to_float16(self.W(f'{self.prefix}.embed_tokens.weight', native=True)), F16)
        h = self.f32(g.node('Gather', [emb, 'input_ids'], '/text/embed'), '/text/embed')
        rope, bias = self.rotary(), self.attention_bias()
        taken = []
        for i in range(layers):
            print(f'text layer {i}', flush=True)
            h = self.layer(i, h, rope, bias)
            # (A shorter encoder, for checks, outputs its last layer too.)
            if i + 1 in TEXT_LAYERS or (i + 1 == layers and layers not in TEXT_LAYERS):
                taken.append(h)
        out = g.node('Concat', taken, '/text/ctx', axis=-1) if len(taken) > 1 else taken[0]
        g.nodes.append(helper.make_node('Cast', [out], ['ctx'], name='/text/ctx/Cast', to=F16))


# ---- Transformer ------------------------------------------------------------------------------------------------


class Transformer(Builder):
    """The DiT (dit.py), text-to-image: modes image and text (no reference images)."""

    def __init__(self, g: Graph, W: Weights):
        super().__init__(g, W)
        self.ones = g.init('ln.ones', np.ones(HIDDEN, np.float32), F32)

    def modulate(self, x: str, shift: str, scale: str, name: str) -> str:
        """LayerNorm(x) × (1 + scale) + shift (float32), float16 out."""
        g = self.g
        n = g.node('LayerNormalization', [x, self.ones], f'{name}/LayerNorm', axis=-1, epsilon=EPS)
        n = g.node('Mul', [n, g.node('Add', [scale, g.const(f'{name}/one', 1.0, F32)], f'{name}/scale')], f'{name}/Mul')
        return self.f16(g.node('Add', [n, shift], f'{name}/Add'), name)

    def qk(self, x: str, key: str, cos: str, sin: str, name: str) -> str:
        """Per-head RMSNorm (float32) then the rotary embedding on adjacent pairs, float16 [1, S, HIDDEN] out."""
        g = self.g
        x = g.node('Reshape', [self.f32(x, name), g.const(f'{name}/shape', [1, -1, HEADS, HEAD_DIM])], f'{name}/Reshape')
        w = g.init(key, self.W(key).numpy().astype(np.float32), F32)
        x = g.node('SimplifiedLayerNormalization', [x, w], f'{name}/norm', axis=-1, epsilon=EPS, stash_type=1)
        x = g.node('Reshape', [x, g.const(f'{name}/pairs', [1, 0, HEADS, HEAD_DIM // 2, 2])], f'{name}/pairs', allowzero=0)
        a, b = self.split(x, [1, 1], f'{name}/pair_split')
        ra = g.node('Sub', [g.node('Mul', [a, cos], f'{name}/ac'), g.node('Mul', [b, sin], f'{name}/bs')], f'{name}/ra')
        rb = g.node('Add', [g.node('Mul', [a, sin], f'{name}/as'), g.node('Mul', [b, cos], f'{name}/bc')], f'{name}/rb')
        out = g.node('Concat', [ra, rb], f'{name}/rot', axis=-1)
        out = g.node('Reshape', [out, g.const(f'{name}/flat', [1, 0, HIDDEN])], f'{name}/flat', allowzero=0)
        return self.f16(out, f'{name}/out')

    def block(self, p: str, name: str, x: str, rope: tuple[str, str], mods: tuple[str, str, str]) -> str:
        """dit.py's Block: attention and a SwiGLU MLP side by side from one modulated input, gated, added."""
        g = self.g
        shift, scale, gate = mods
        xm = self.modulate(x, shift, scale, f'{name}/mod')
        fused = self.linear(xm, [f'{p}.q_proj.weight', f'{p}.k_proj.weight', f'{p}.v_proj.weight', f'{p}.mlp_in.weight'], f'{name}/in')
        q, k, v, mlp = self.split(fused, [HIDDEN, HIDDEN, HIDDEN, 2 * MLP], f'{name}/in/Split')
        q = self.qk(q, f'{p}.norm.query_norm.scale', *rope, f'{name}/q')
        k = self.qk(k, f'{p}.norm.key_norm.scale', *rope, f'{name}/k')
        a = g.node('MultiHeadAttention', [q, k, v], f'{name}/attn', domain=MS, num_heads=HEADS)
        gate_in, up = self.split(self.f32(mlp, f'{name}/mlp'), [MLP, MLP], f'{name}/mlp/Split')
        act = self.f16(g.node('Mul', [self.silu(gate_in, f'{name}/silu'), up], f'{name}/act'), f'{name}/act')
        # attn_out(attn) + mlp_out(act) as one product: [attn, act] @ [attn_out | mlp_out].T. mlp_out's results pass
        # float16's 65,504 in the single-stream blocks (~249,000 in block 25), so the input is scaled down by OUT_SCALE
        # and the float32 result back up (exact: a power of 2).
        cat = g.node('Mul', [g.node('Concat', [a, act], f'{name}/cat', axis=-1), g.const(f'{name}/out/down', 1 / OUT_SCALE, F16)], f'{name}/out/scale_down')
        out = self.linear(cat, [f'{p}.attn_out.weight', f'{p}.mlp_out.weight'], f'{name}/out', dim=1)
        out = g.node('Mul', [self.f32(out, f'{name}/out'), g.const(f'{name}/out/up', OUT_SCALE, F32)], f'{name}/out/scale_up')
        return g.node('Add', [x, g.node('Mul', [gate, out], f'{name}/gated')], f'{name}/residual')

    def rotary(self, ids: str, name: str) -> tuple[str, str]:
        """cos and sin [1, S, 1, 64, 1] float32: each of the 4 id axes turns 16 of the head's 64 pairs, at
        frequencies θ^(−2j/32)."""
        g = self.g
        table = np.zeros((len(AXES), HEAD_DIM // 2), np.float64)
        col = 0
        for a, dim in enumerate(AXES):
            table[a, col:col + dim // 2] = 1.0 / THETA ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
            col += dim // 2
        freqs = 'rope.freqs' if 'rope.freqs' in {i.name for i in g.inits} else g.init('rope.freqs', table.astype(np.float32), F32)
        angles = g.node('MatMul', [ids, freqs], f'{name}/MatMul')
        angles = g.node('Reshape', [angles, g.const(f'{name}/shape', [1, -1, 1, HEAD_DIM // 2, 1])], f'{name}/Reshape')
        return g.node('Cos', [angles], f'{name}/Cos'), g.node('Sin', [angles], f'{name}/Sin')

    def timestep(self, t: str, key: str, name: str) -> str:
        """MLPEmbedder(timestep_embedding(t)): sinusoidal (cos first) of t × 1000, then Linear, SiLU, Linear. [1, HIDDEN]."""
        g = self.g
        half = 128
        freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half).astype(np.float32)
        x = g.node('Unsqueeze', [g.node('Mul', [t, g.const(f'{name}/scale', 1000.0, F32)], f'{name}/Mul'), g.const(f'{name}/axes', [1])], f'{name}/Unsqueeze')
        args = g.node('Mul', [x, g.init(f'{name}.freqs', freqs, F32)], f'{name}/args')
        emb = g.node('Concat', [g.node('Cos', [args], f'{name}/Cos'), g.node('Sin', [args], f'{name}/Sin')], f'{name}/Concat', axis=-1)
        h = self.silu(self.float_matmul(emb, f'{key}.in_layer.weight', f'{name}/in_layer'), f'{name}/silu')
        return self.float_matmul(h, f'{key}.out_layer.weight', f'{name}/out_layer')

    def modulation(self, vec_act: str, key: str, name: str) -> list[str]:
        """ModeModulation: Linear(SiLU(vec)) → shift, scale, gate, each [1, 1, HIDDEN] float32."""
        g = self.g
        m = self.float_matmul(vec_act, key, f'{name}/MatMul')
        m = g.node('Reshape', [m, g.const(f'{name}/shape', [1, 1, 3 * HIDDEN])], f'{name}/Reshape')
        return self.split(m, [HIDDEN] * 3, f'{name}/Split')

    def build(self, depth: int = DEPTH, depth_single: int = DEPTH_SINGLE):
        g = self.g
        g.inputs = [
            helper.make_tensor_value_info('x', F32, [1, 'image_seq', 128]),
            helper.make_tensor_value_info('x_ids', F32, [1, 'image_seq', 4]),
            helper.make_tensor_value_info('t', F32, [1]),
            helper.make_tensor_value_info('t_ref', F32, [1]),
            helper.make_tensor_value_info('ctx', F16, [1, 'text_seq', 20480]),
            helper.make_tensor_value_info('ctx_ids', F32, [1, 'text_seq', 4]),
        ]
        g.outputs = [helper.make_tensor_value_info('pred', F32, [1, 'image_seq', 128])]

        # vec per mode: the image's time t, the text's time 0, both plus the flow map's target time.
        ref = self.timestep('t_ref', 'ref_time_in', '/ref_time')
        vec_img = g.node('Add', [self.timestep('t', 'time_in', '/time_img'), ref], '/vec_img')
        vec_txt = g.node('Add', [self.timestep(g.const('/time_txt/zero', [0.0], F32), 'time_in', '/time_txt'), ref], '/vec_txt')
        act_img, act_txt = self.silu(vec_img, '/vec_img/act'), self.silu(vec_txt, '/vec_txt/act')
        early_img = self.modulation(act_img, 'early_stream_modulations.image.lin.weight', '/early_mod_img')
        early_txt = self.modulation(act_txt, 'early_stream_modulations.txt.lin.weight', '/early_mod_txt')
        single_img = self.modulation(act_img, 'single_stream_modulations.image.lin.weight', '/single_mod_img')
        single_txt = self.modulation(act_txt, 'single_stream_modulations.txt.lin.weight', '/single_mod_txt')

        img = self.f32(self.float_matmul(self.f16('x', '/x'), 'emb_in.image.weight', '/emb_in', F16), '/emb_in')
        # Qwen's features reach ~10,000 (its first-token outliers), and txt_in's products past float16's 65,504
        # (~109,000), which WebGPU computes in float16: scaled down by 16 going in and back up after (exact, a power of 2).
        scaled = g.node('Mul', ['ctx', g.const('/txt_in/down', 1 / TXT_IN_SCALE, F16)], '/txt_in/scale_down')
        txt = g.node('Mul', [self.f32(self.linear(scaled, ['txt_in.weight'], '/txt_in'), '/txt_in'), g.const('/txt_in/up', TXT_IN_SCALE, F32)], '/txt_in/scale_up')
        rope_img, rope_txt = self.rotary('x_ids', '/rope_img'), self.rotary('ctx_ids', '/rope_txt')

        for d in range(depth):
            print(f'mode blocks {d}', flush=True)
            img = self.block(f'content_mode_blocks.image.{d}', f'/image.{d}', img, rope_img, early_img)
            txt = self.block(f'txt_mode_blocks.{d}', f'/text.{d}', txt, rope_txt, early_txt)

        # Single stream over [image, text]; each token's modulation from its own mode.
        h = g.node('Concat', [img, txt], '/joint/Concat', axis=1)
        ids = g.node('Concat', ['x_ids', 'ctx_ids'], '/joint/ids', axis=1)
        rope = self.rotary(ids, '/rope_joint')
        shape = lambda x: g.node('Concat', [g.node('Slice', [g.node('Shape', [x], f'/joint/Shape_{x}'), g.const(f'/joint/{x}/s0', [0]), g.const(f'/joint/{x}/s1', [2])], f'/joint/Slice_{x}'),
                                            g.const(f'/joint/{x}/hidden', [HIDDEN])], f'/joint/shape_{x}', axis=0)
        img_shape, txt_shape = shape('x'), shape('ctx')
        mods = tuple(
            g.node('Concat', [g.node('Expand', [a, img_shape], f'/joint/mod{i}/img'), g.node('Expand', [b, txt_shape], f'/joint/mod{i}/txt')], f'/joint/mod{i}', axis=1)
            for i, (a, b) in enumerate(zip(single_img, single_txt))
        )
        for i in range(depth_single):
            print(f'single block {i}', flush=True)
            h = self.block(f'single_blocks.{i}', f'/single.{i}', h, rope, mods)
        lengths = [g.node('Slice', [g.node('Shape', [t], f'/out/Shape_{t}'), g.const(f'/out/{t}/s0', [1]), g.const(f'/out/{t}/s1', [2])], f'/out/Slice_{t}') for t in ('x', 'ctx')]
        img, _ = self.split_dynamic(h, g.node('Concat', lengths, '/out/sizes', axis=0), '/out/Split')

        # LastLayer: (shift, scale) from SiLU(vec_img), modulated LayerNorm, Linear to 128.
        m = self.float_matmul(act_img, 'final_layer.image.adaLN_modulation.1.weight', '/final/adaLN')
        m = g.node('Reshape', [m, g.const('/final/shape', [1, 1, 2 * HIDDEN])], '/final/Reshape')
        shift, scale = self.split(m, [HIDDEN, HIDDEN], '/final/Split')
        out = self.float_matmul(self.modulate(img, shift, scale, '/final/mod'), 'final_layer.image.linear.weight', '/final/linear', F16)
        g.nodes.append(helper.make_node('Cast', [out], ['pred'], name='/final/Cast', to=F32))

    def split_dynamic(self, x: str, sizes: str, name: str) -> list[str]:
        outs = [f'{name}/output_{i}' for i in range(2)]
        self.g.node('Split', [x, sizes], name, outputs=outs, axis=1)
        return outs


# ---- Decoder ----------------------------------------------------------------------------------------------------


def export_decoder(out: Path):
    """The VAE decoder with autoencoder.py's decode() preprocessing, traced from the reference module (float32)."""
    sys.path.insert(0, str(REFERENCE_CODE))
    from flux3.autoencoder import EMBED_DIM, LATENT_CHANNELS, AutoEncoder
    from safetensors.torch import load_file

    ae = AutoEncoder()
    ae.load_state_dict(load_file(str(SRC / 'ae.safetensors')))
    ae.eval()

    class Decode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = ae.decoder
            self.register_buffer('std', torch.sqrt(ae.bn.running_var + ae.bn_eps).view(1, -1, 1, 1))
            self.register_buffer('mean', ae.bn.running_mean.view(1, -1, 1, 1))

        def forward(self, z):
            z = z * self.std + self.mean
            b, _, h, w = z.shape
            z = z.reshape(1, EMBED_DIM, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, EMBED_DIM, 2 * h, 2 * w)
            return self.decoder(z)

    out.mkdir(parents=True, exist_ok=True)
    z = torch.randn(1, LATENT_CHANNELS, 16, 16)
    torch.onnx.export(Decode(), (z,), str(out / 'decoder.onnx'), input_names=['z'], output_names=['image'],
                      dynamic_axes={'z': {2: 'h', 3: 'w'}, 'image': {2: 'height', 3: 'width'}}, opset_version=21, dynamo=False)
    # One external data file (the app reads weights from it tensor by tensor).
    model = onnx.load(str(out / 'decoder.onnx'))
    onnx.save(model, str(out / 'decoder.onnx'), save_as_external_data=True, all_tensors_to_one_file=True, location='decoder.onnx.data', size_threshold=1024)
    print('saved decoder', flush=True)


# ---- Export -----------------------------------------------------------------------------------------------------


def save(g: Graph, out: Path, file: str, title: str):
    g.close()
    graph = helper.make_graph(g.nodes, title, g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / file))
    print('saved', file, g.offset, flush=True)


def export_text_encoder(out: Path, layers: int = max(TEXT_LAYERS)):
    cfg = json.loads((SRC / 'text_encoder' / 'config.json').read_text())['text_config']
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'text_encoder.onnx.data')
    TextEncoder(g, Weights(SRC / 'text_encoder', SOURCE, REVISION), cfg).build(layers)
    save(g, out, 'text_encoder.onnx', 'FLUX 3 Image Klein text encoder (Qwen3-VL-4B, 32 layers)')


def export_transformer(out: Path, depth: int = DEPTH, depth_single: int = DEPTH_SINGLE):
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'transformer.onnx.data')
    Transformer(g, checkpoint()).build(depth, depth_single)
    save(g, out, 'transformer.onnx', 'FLUX 3 Image Klein transformer')


def finish(out: Path):
    """The tokenizer, BFL's licence and notice (with the statement of modification it requires) and a model card."""
    shutil.copy(SRC / 'text_encoder' / 'tokenizer.json', out / 'tokenizer.json')
    shutil.copy(SRC / 'LICENSE.md', out / 'LICENSE.md')
    shutil.copy(SRC / 'text_encoder' / 'LICENSE', out / 'LICENSE-text-encoder')
    notice = (SRC / 'NOTICE.md').read_text().rstrip()
    (out / 'NOTICE.md').write_text(f"{notice}\n\nThese files were modified by Quick Inference: converted to ONNX, with the text encoder's and the "
                                   "transformer's linear layers quantised to 4 bits (see README.md). They are not an official product of "
                                   "Black Forest Labs and are not endorsed by it.\n")
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = f"""---
license: other
license_name: flux-kommunity-license
license_link: LICENSE.md
base_model:
- {SOURCE}
tags:
- onnx
- webgpu
- text-to-image
---

# {REPO}

[{SOURCE}](https://huggingface.co/{SOURCE}) converted to ONNX for running in a web browser with ONNX Runtime Web's
WebGPU backend. Text-to-image only. Not an official product of Black Forest Labs; see NOTICE.md.

## Changes from the original

- text_encoder.onnx: Qwen3-VL-4B-Instruct's language model, its first 32 layers, returning the residual stream after
  layers 4, 8, …, 32 side by side (as the reference text_encoder.py). Linear layers quantised to 4 bits (round to
  nearest, block size 32); residual stream and norms in float32.
- transformer.onnx: the flow-map DiT. Linear layers quantised to 4 bits; residual streams, norms, modulation and the
  rotary embedding in float32; attention in float16.
- decoder.onnx: the VAE decoder with the latent statistics undone and the 2 × 2 unpatchify, float32.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

FLUX Kommunity License v1.0 (LICENSE.md), as the original: non-commercial use, with the terms set out there. The text
encoder is Qwen3-VL-4B-Instruct, Apache 2.0 (LICENSE-text-encoder).
"""


# ---- Checks against the reference code ----------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_transformer(out: Path, depth: int = 1, depth_single: int = 1):
    """A transformer with `depth` mode blocks and `depth_single` single blocks against dit.py's DiT cut to the same
    blocks (float32 on the CPU): random latents (8 × 12 tokens) and text features (80 tokens)."""
    import onnxruntime as ort
    from safetensors import safe_open

    sys.path.insert(0, str(REFERENCE_CODE))
    from flux3.autoencoder import latent_ids
    from flux3.dit import DiT
    from flux3.pipeline import pack, scaled_ids

    export_transformer(out, depth, depth_single)
    import flux3.dit as reference
    reference.DEPTH = depth  # its forward loops over range(DEPTH) mode blocks
    with torch.device('meta'):
        dit = DiT(flow_map=True)
    dit.content_mode_blocks['image'] = dit.content_mode_blocks['image'][:depth]
    dit.content_mode_blocks['image_cond'] = dit.content_mode_blocks['image_cond'][:depth]
    dit.txt_mode_blocks = dit.txt_mode_blocks[:depth]
    dit.single_blocks = dit.single_blocks[:depth_single]
    wanted = dit.state_dict().keys()
    with safe_open(str(SRC / 'model.safetensors'), framework='pt') as f:
        state = {k: f.get_tensor(k).float() for k in wanted}
    dit.load_state_dict(state, assign=True)
    dit.eval()

    torch.manual_seed(0)
    h, w, length = 8, 12, 80
    x = torch.randn(h * w, 128)
    ids = scaled_ids(latent_ids(h, w))
    # Like Qwen's: small values with outliers of ~10,000 (which overflowed float16 in txt_in before it was scaled).
    ctx = torch.randn(length, 20480) * 4
    ctx[1:, ::997] = 10000.0
    ctx_ids = torch.stack([torch.zeros(length)] * 3 + [torch.arange(length).float()], dim=-1)
    t, t_ref = 0.7, 0.55
    packed = pack([[(ctx, ctx_ids), None, (x, ids)]])
    packed.data = {k: v.float() for k, v in packed.data.items()}
    packed.timesteps = {k: v.float() for k, v in packed.timesteps.items()}
    packed.timesteps['image'].fill_(t)
    with torch.no_grad():
        ref = dit(packed, timesteps_ref=torch.tensor([t_ref])).numpy()

    session = ort.InferenceSession(str(out / 'transformer.onnx'), providers=['CPUExecutionProvider'])
    pred = session.run(None, {
        'x': x[None].numpy(), 'x_ids': ids[None].numpy(), 't': np.array([t], np.float32), 't_ref': np.array([t_ref], np.float32),
        'ctx': ctx[None].numpy().astype(np.float16), 'ctx_ids': ctx_ids[None].numpy(),
    })[0][0]
    print(f'transformer ({depth} + {depth_single} blocks, {Settings.bits}-bit): cosine {cosine(pred, ref):.5f}, '
          f'max diff {np.abs(pred - ref).max():.4f}, reference range {ref.min():.3f}..{ref.max():.3f}')


def check_text_encoder(out: Path, layers: int = 4):
    """The text encoder's layers up to `layers` against transformers' Qwen3-VL (float32), on a prompt prepared as
    text_encoder.py does (chat template, padded to a multiple of 80, padded positions at 1)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    path = str(SRC / 'text_encoder')
    # The processor's tokenizer and chat template (the processor itself needs torchvision for video).
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = 'right'
    template = json.loads((SRC / 'text_encoder' / 'chat_template.json').read_text())['chat_template']
    prompt = [{'role': 'user', 'content': 'A red fox in fresh snow, morning light.'}]
    text = tokenizer.apply_chat_template(prompt, chat_template=template, tokenize=False, add_generation_prompt=True)
    real = tokenizer(text, return_tensors='pt')['input_ids'].shape[1]
    length = math.ceil(real / 80) * 80
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=length)
    mask = inputs['attention_mask']
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)

    # Only the language model's first `layers` layers, float32 (the whole model in float32 needs ~18 GB).
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    config = AutoConfig.from_pretrained(path).text_config
    # One more layer than compared: the last hidden state transformers returns has the final norm applied (the real
    # encoder's layer 32 of 36 doesn't).
    config.num_hidden_layers = layers + 1
    model = Qwen3VLTextModel(config).eval()
    index = json.loads((SRC / 'text_encoder' / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.'
    state = {}
    for key in model.state_dict():
        with safe_open(str(SRC / 'text_encoder' / index[prefix + key]), framework='pt') as f:
            state[key] = f.get_tensor(prefix + key).float()
    model.load_state_dict(state)
    with torch.no_grad():
        result = model(input_ids=inputs['input_ids'], attention_mask=mask, position_ids=positions[None].expand(4, -1, -1),
                       output_hidden_states=True, use_cache=False)
    ref = result.hidden_states[layers][0].numpy()
    del model

    export_text_encoder(out, layers)
    session = ort.InferenceSession(str(out / 'text_encoder.onnx'), providers=['CPUExecutionProvider'])
    ctx = session.run(None, {'input_ids': inputs['input_ids'].numpy(), 'attention_mask': mask.numpy(), 'position_ids': positions.numpy()})[0][0]
    ctx = ctx[:, -ref.shape[-1]:].astype(np.float32)
    print(f'text encoder ({layers} layers, {Settings.bits}-bit, {real} of {length} tokens real): cosine {cosine(ctx, ref):.5f}, '
          f'real tokens {cosine(ctx[:real], ref[:real]):.5f}, padding {cosine(ctx[real:], ref[real:]):.5f}, max |ref| {np.abs(ref).max():.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parts', nargs='*', default=['text_encoder', 'transformer', 'decoder'])
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--check', choices=['text_encoder', 'transformer'])
    parser.add_argument('--layers', type=int, help='checks: text encoder layers / transformer blocks (each kind)')
    args = parser.parse_args()
    Settings.bits = args.bits
    if args.check == 'transformer':
        n = args.layers or 1
        check_transformer(args.out, n, n)
    elif args.check == 'text_encoder':
        check_text_encoder(args.out, args.layers or 4)
    else:
        for part in args.parts:
            {'text_encoder': export_text_encoder, 'transformer': export_transformer, 'decoder': export_decoder}[part](args.out)
        finish(args.out)
