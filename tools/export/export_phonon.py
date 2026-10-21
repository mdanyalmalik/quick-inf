"""Exports Phonon 31M English (Gradium's text-to-speech model, Apache-2.0) to ONNX for ONNX Runtime Web, from a
PyTorch reimplementation of Gradium's Rust runtime (xn-ptts, `ptts/src`), checked against that runtime's own
intermediate tensors.

Three float32 graphs, opset 17, no control flow, written to exports/tts-int4/phonon-31m-en/:
- text_embed.onnx: token ids -> the flow LM's text embeddings (the LUT conditioner, a table lookup).
- flow_lm.onnx: one call of T positions of the flow-matching transformer against an explicit, growing KV cache
  (past_* in, present_* out; rope positions come from the past length). The same graph takes the voice prompt
  (T = 125), the text prompt (T = tokens) and each generated frame (T = 1). Heads on the last position: the EOS
  logit, the latent (one LSD flow step from `noise`, which the caller draws with standard deviation
  sqrt(temperature)) and the next frame's input embedding (input_linear(latent) + the default conditioners).
- mimi_decoder.onnx: one latent frame -> 1920 PCM samples at 24 kHz, with every streaming buffer (conv tails,
  transposed-conv overlaps, the decoder transformer's 250-position KV window, the position) as inputs and outputs.
Rope is applied half-split, with the q/k projection rows permuted once at export (as the Core ML export does);
attention is unchanged by it. The voice embedding (default-voice.safetensors projected by the speaker projection)
and the first frame's input (input_linear(bos_emb) + conditioners) are written as raw float32 files, described in
phonon.json. README.md in the output folder describes the generation loop.

`--check DUMP...` compares the PyTorch model and the ONNX graphs (`--graphs FLOW:MIMI...`, default the float pair)
with dumps of the Rust runtime (its `dump_phonon` example, added to the local xn-ptts clone): voice and text
embeddings, the KV cache after each prompt, every frame's latent and EOS logit (teacher-forced and free-running on
the dump's noise), and Mimi's PCM.

    tools/export/.venv/bin/python -I tools/export/export_phonon.py [--check DUMP.safetensors ...] [--no-export] [--out DIR]
        [--graphs flow_lm_q4:mimi_decoder_q4c8 ...]
    xn-ptts/target/release/examples/dump_phonon --model tools/export/.cache/phonon/src --lang en --out DUMP "text"

The int4 graphs: quantize_int4.py on each (flow_lm, text_embed, mimi_decoder -> *_q4.onnx), and on mimi_decoder with
`--conv-bits 8` (-> mimi_decoder_q4c8.onnx), on flow_lm with `--matmul-bits 8` (-> flow_lm_q8.onnx).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache/phonon/src'
RUST = HERE / '.cache/phonon/xn-ptts'
OUT = ROOT / 'exports/tts-int4/phonon-31m-en'
OPSET = 17
NEG = -1e9  # masked attention scores: exp() of it is exactly 0, like -inf, without inf arithmetic


def inv_freq(head_dim: int, max_period: float) -> torch.Tensor:
    """Rope frequencies as the runtime computes them, in float32: 1 / max_period^(i / half)."""
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    return torch.from_numpy((np.float32(1) / np.power(np.float32(max_period), i / np.float32(half))).astype(np.float32))


def half_split(w: torch.Tensor, d: int, heads: int) -> torch.Tensor:
    """Reorders the q and k rows of a fused [3D, D] in_proj from interleaved rope pairs to half-split order: per head,
    row 2i moves to i and row 2i+1 to HD/2 + i. The same permutation of q and k cancels in q.k; v is untouched."""
    hd = d // heads
    idx = torch.arange(d).view(heads, hd // 2, 2).transpose(1, 2).reshape(-1)
    w = w.clone()
    w[:d] = w[:d][idx]
    w[d:2 * d] = w[d:2 * d][idx]
    return w


def rope(x, cos, sin, half: int):
    """Half-split rotary embedding of x [1, H, T, HD] with cos/sin [T, HD/2]."""
    lo, hi = x[..., :half], x[..., half:]
    return torch.cat([lo * cos - hi * sin, lo * sin + hi * cos], dim=-1)


def layer_norm(x, w, b, eps):
    return F.layer_norm(x, tuple(w.shape), w, b, eps)


class Layer(nn.Module):
    """A pre-norm transformer layer of either model (bias-free projections, GELU exact or tanh as `gelu` names it,
    optional layer scale), whose attention runs over `past` K/V plus this call's rows and returns this call's K/V
    rows."""

    def __init__(self, w: dict, p: str, d: int, heads: int, layer_scale: bool, gelu: str = 'erf'):
        super().__init__()
        self.heads, self.hd = heads, d // heads
        self.approximate = {'erf': 'none', 'tanh': 'tanh'}[gelu]
        for name in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'linear1.weight', 'linear2.weight',
                     'self_attn.out_proj.weight'):
            self.register_buffer(name.replace('.', '_'), w[p + name])
        self.register_buffer('in_proj', half_split(w[p + 'self_attn.in_proj.weight'], d, heads))
        self.ls = layer_scale
        if layer_scale:
            self.register_buffer('ls1', w[p + 'layer_scale_1.scale'])
            self.register_buffer('ls2', w[p + 'layer_scale_2.scale'])

    def forward(self, x, cos, sin, mask, past_k, past_v):
        h = layer_norm(x, self.norm1_weight, self.norm1_bias, 1e-5)
        qkv = F.linear(h, self.in_proj).view(1, -1, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        half = self.hd // 2
        q, k, v = rope(qkv[0], cos, sin, half), rope(qkv[1], cos, sin, half), qkv[2]
        keys, values = torch.cat([past_k, k], 2), torch.cat([past_v, v], 2)
        att = torch.matmul(q, keys.transpose(-1, -2)) * (self.hd ** -0.5) + mask
        o = torch.matmul(torch.softmax(att, -1), values).transpose(1, 2).reshape(1, -1, self.heads * self.hd)
        o = F.linear(o, self.self_attn_out_proj_weight)
        x = x + (o * self.ls1 if self.ls else o)
        h = layer_norm(x, self.norm2_weight, self.norm2_bias, 1e-5)
        f = F.linear(F.gelu(F.linear(h, self.linear1_weight), approximate=self.approximate), self.linear2_weight)
        return x + (f * self.ls2 if self.ls else f), k, v


def time_embed(w: dict, i: int, t: float, norm: str = 'rms') -> torch.Tensor:
    """TimestepEmbedder at a fixed time, ending in its RMS norm: uncentred (`time_rms_norm: rms`, Phonon) or over
    the centred, unbiased variance (`rms_var`, Pocket TTS's `torch.var`)."""
    p = f'flow_lm.flow_net.time_embed.{i}.'
    args = t * w[p + 'freqs']
    x = torch.cat([torch.cos(args), torch.sin(args)])
    x = F.silu(F.linear(x, w[p + 'mlp.0.weight'], w[p + 'mlp.0.bias']))
    x = F.linear(x, w[p + 'mlp.2.weight'], w[p + 'mlp.2.bias'])
    var = (x * x).mean() if norm == 'rms' else x.var()
    return x * torch.rsqrt(var + 1e-5) * w[p + 'mlp.3.alpha']


class FlowLM(nn.Module):
    """emb [1, T, D] + past K/V -> EOS logit, latent and next input embedding for the last position, present K/V."""

    def __init__(self, w: dict, cfg: dict, conditions: torch.Tensor):
        super().__init__()
        c = cfg['flow_lm']
        assert cfg['lsd_decode_steps'] == 1 and c['time_rms_norm'] in ('rms', 'rms_var')
        self.d, self.heads, self.depth = c['d_model'], c['num_heads'], c['flow_depth']
        self.layers = nn.ModuleList(Layer(w, f'flow_lm.transformer.layers.{i}.', self.d, self.heads, False,
                                          c.get('gelu', 'erf')) for i in range(c['num_layers']))
        self.register_buffer('inv_freq', inv_freq(self.d // self.heads, c['max_period']))
        f = 'flow_lm.flow_net.'
        names = {'out_norm_w': 'flow_lm.out_norm.weight', 'out_norm_b': 'flow_lm.out_norm.bias',
                 'eos_w': 'flow_lm.out_eos.weight', 'eos_b': 'flow_lm.out_eos.bias',
                 'input_linear': 'flow_lm.input_linear.weight', 'in_w': f + 'input_proj.weight',
                 'in_b': f + 'input_proj.bias', 'cond_w': f + 'cond_embed.weight', 'cond_b': f + 'cond_embed.bias',
                 'fin_ada_w': f + 'final_layer.adaLN_modulation.1.weight',
                 'fin_ada_b': f + 'final_layer.adaLN_modulation.1.bias', 'fin_w': f + 'final_layer.linear.weight',
                 'fin_b': f + 'final_layer.linear.bias'}
        for k, v in names.items():
            self.register_buffer(k, w[v])
        for i in range(self.depth):
            p = f'{f}res_blocks.{i}.'
            for k, v in (('ada_w', 'adaLN_modulation.1.weight'), ('ada_b', 'adaLN_modulation.1.bias'),
                         ('ln_w', 'in_ln.weight'), ('ln_b', 'in_ln.bias'), ('m0_w', 'mlp.0.weight'),
                         ('m0_b', 'mlp.0.bias'), ('m2_w', 'mlp.2.weight'), ('m2_b', 'mlp.2.bias')):
                self.register_buffer(f'rb{i}_{k}', w[p + v])
        # The single LSD step always goes from s = 0 to t = 1, so the averaged time embedding is a constant.
        norm = c['time_rms_norm']
        self.register_buffer('time', (time_embed(w, 0, 0.0, norm) + time_embed(w, 1, 1.0, norm)) * 0.5)
        self.register_buffer('conditions', conditions.reshape(-1))
        flow_dim = w[f + 'final_layer.linear.weight'].shape[1]  # the flow head's width (Pocket TTS: 512 < d_model)
        self.register_buffer('ones', torch.ones(flow_dim))
        self.register_buffer('zeros', torch.zeros(flow_dim))

    def step_emb(self, latent):
        """A frame's input to the transformer: input_linear(latent) + the conditioners."""
        return F.linear(latent, self.input_linear) + self.conditions

    def flow(self, cond, noise):
        x = F.linear(noise, self.in_w, self.in_b)
        y = self.time + F.linear(cond, self.cond_w, self.cond_b)
        ys = F.silu(y)
        for i in range(self.depth):
            g = lambda k: getattr(self, f'rb{i}_{k}')
            shift, scale, gate = F.linear(ys, g('ada_w'), g('ada_b')).chunk(3, -1)
            h = layer_norm(x, g('ln_w'), g('ln_b'), 1e-6) * (1 + scale) + shift
            h = F.linear(F.silu(F.linear(h, g('m0_w'), g('m0_b'))), g('m2_w'), g('m2_b'))
            x = x + gate * h
        shift, scale = F.linear(ys, self.fin_ada_w, self.fin_ada_b).chunk(2, -1)
        h = layer_norm(x, self.ones, self.zeros, 1e-6) * (1 + scale) + shift
        return noise + F.linear(h, self.fin_w, self.fin_b)

    def forward(self, emb, noise, *past):
        t, p = emb.shape[1], past[0].shape[2]
        q_pos = torch.arange(t, dtype=torch.float32) + p
        angles = q_pos[:, None] * self.inv_freq
        cos, sin = torch.cos(angles), torch.sin(angles)
        k_pos = torch.arange(p + t, dtype=torch.float32)
        mask = (k_pos[None, :] > q_pos[:, None]).to(torch.float32) * NEG
        x, present = emb, []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, mask, past[2 * i], past[2 * i + 1])
            present += [torch.cat([past[2 * i], k], 2), torch.cat([past[2 * i + 1], v], 2)]
        last = layer_norm(x[:, -1], self.out_norm_w, self.out_norm_b, 1e-5)
        eos = F.linear(last, self.eos_w, self.eos_b)
        latent = self.flow(last, noise)
        return (latent.reshape(1, 1, -1), eos, self.step_emb(latent).reshape(1, 1, -1), *present)


def sconv(x, prev, w, b, dilation=1):
    """Streaming causal conv: the carried left context, then this call's samples; returns the new context."""
    full = torch.cat([prev, x], 2)
    return F.conv1d(full, w, b, dilation=dilation), full[:, :, full.shape[2] - prev.shape[2]:]


def sconvtr(x, partial, w, b, stride):
    """Streaming transposed conv: overlap-add the carried tail; the new tail is kept without the bias."""
    y = F.conv_transpose1d(x, w, b, stride=stride)
    pt = partial.shape[2]
    n = y.shape[2] - pt
    out = torch.cat([y[:, :, :pt] + partial, y[:, :, pt:n]], 2)
    tail = y[:, :, n:]
    return out, (tail - b[None, :, None] if b is not None else tail)


class Mimi(nn.Module):
    """One latent frame [1, 1, LDIM] -> PCM [1, 1, 1920], with every streaming buffer in and out."""

    def __init__(self, w: dict, cfg: dict):
        super().__init__()
        m = cfg['mimi']
        assert m['ratios'] == [6, 5, 4] and m['n_residual_layers'] == 1 and m['channels'] == 1
        self.d, self.heads, self.ctx = m['transformer_d_model'], m['transformer_num_heads'], m['transformer_context']
        self.up = round(m['sample_rate'] / np.prod(m['ratios']) / m['frame_rate'])  # 16 transformer steps a frame
        self.layers = nn.ModuleList(Layer(w, f'mimi.decoder_transformer.transformer.layers.{i}.', self.d, self.heads,
                                          True, m.get('gelu', 'erf')) for i in range(m['transformer_num_layers']))
        self.register_buffer('inv_freq', inv_freq(self.d // self.heads, m['transformer_max_period']))
        self.register_buffer('mean', w['flow_lm.emb_mean'])
        self.register_buffer('std', w['flow_lm.emb_std'])
        self.register_buffer('out_proj', w['mimi.quantizer.output_proj.weight'][:, :, 0])
        self.register_buffer('up_w', w['mimi.upsample.convtr.convtr.weight'][:, 0, :][None])  # [1, D, 2 * up]
        d = 'mimi.decoder.model.'
        for i in (0, 11):
            self.register_buffer(f'c{i}_w', w[f'{d}{i}.conv.weight'])
            self.register_buffer(f'c{i}_b', w[f'{d}{i}.conv.bias'])
        for i in (2, 5, 8):
            self.register_buffer(f't{i}_w', w[f'{d}{i}.convtr.weight'])
            self.register_buffer(f't{i}_b', w[f'{d}{i}.convtr.bias'])
        for i in (3, 6, 9):
            for j in (1, 3):
                self.register_buffer(f'r{i}_{j}_w', w[f'{d}{i}.block.{j}.conv.weight'])
                self.register_buffer(f'r{i}_{j}_b', w[f'{d}{i}.block.{j}.conv.bias'])
        # Query q of a call sees cache slots q ..= ctx + q: the ctx previous positions and itself, as the runtime's
        # trimmed cache does. Slot j holds position pos - ctx + j, so it is empty while that is negative.
        # `context_includes_self` (Pocket TTS: a key is seen while query - key < ctx) drops slot q: ctx - 1 previous
        # positions and itself.
        q, j = torch.arange(self.up)[:, None], torch.arange(self.ctx + self.up)[None, :]
        first = q + 1 if m.get('context_includes_self') else q
        self.register_buffer('band', ((j < first) | (j > q + self.ctx)).to(torch.float32) * NEG)
        self.register_buffer('slot', torch.arange(self.ctx + self.up, dtype=torch.float32) - self.ctx)

    def state_spec(self) -> list:
        """(name, shape) of every streaming buffer, in the order the graph takes and returns them; all start at 0."""
        hd = self.d // self.heads
        spec = [('up_partial', [1, self.d, self.up]), ('c0_prev', [1, self.d, 6]), ('t2_partial', [1, 256, 6]),
                ('r3_prev', [1, 256, 2]), ('t5_partial', [1, 128, 5]), ('r6_prev', [1, 128, 2]),
                ('t8_partial', [1, 64, 4]), ('r9_prev', [1, 64, 2]), ('c11_prev', [1, 64, 2])]
        for i in range(len(self.layers)):
            spec += [(f'k{i}', [1, self.heads, self.ctx, hd]), (f'v{i}', [1, self.heads, self.ctx, hd])]
        return spec + [('pos', [1])]

    def resblock(self, i, x, prev):
        h, prev = sconv(F.elu(x), prev, getattr(self, f'r{i}_1_w'), getattr(self, f'r{i}_1_b'))
        h = F.conv1d(F.elu(h), getattr(self, f'r{i}_3_w'), getattr(self, f'r{i}_3_b'))
        return x + h, prev

    def forward(self, latent, up_partial, c0_prev, t2_partial, r3_prev, t5_partial, r6_prev, t8_partial, r9_prev,
                c11_prev, *kv_pos):
        kv, pos = kv_pos[:-1], kv_pos[-1]
        x = F.linear(latent * self.std + self.mean, self.out_proj).transpose(1, 2)  # [1, D, 1]
        # The stride-16 depthwise upsample of one frame is a per-channel outer product with the kernel.
        y = x * self.up_w
        x, up_out = y[:, :, :self.up] + up_partial, y[:, :, self.up:]
        angles = (pos + torch.arange(self.up, dtype=torch.float32))[:, None] * self.inv_freq
        cos, sin = torch.cos(angles), torch.sin(angles)
        mask = self.band + ((pos + self.slot) < 0).to(torch.float32) * NEG
        h, kv_out = x.transpose(1, 2), []
        for i, layer in enumerate(self.layers):
            h, k, v = layer(h, cos, sin, mask, kv[2 * i], kv[2 * i + 1])
            kv_out += [torch.cat([kv[2 * i], k], 2)[:, :, self.up:], torch.cat([kv[2 * i + 1], v], 2)[:, :, self.up:]]
        x, c0_out = sconv(h.transpose(1, 2), c0_prev, self.c0_w, self.c0_b)
        x, t2_out = sconvtr(F.elu(x), t2_partial, self.t2_w, self.t2_b, 6)
        x, r3_out = self.resblock(3, x, r3_prev)
        x, t5_out = sconvtr(F.elu(x), t5_partial, self.t5_w, self.t5_b, 5)
        x, r6_out = self.resblock(6, x, r6_prev)
        x, t8_out = sconvtr(F.elu(x), t8_partial, self.t8_w, self.t8_b, 4)
        x, r9_out = self.resblock(9, x, r9_prev)
        pcm, c11_out = sconv(F.elu(x), c11_prev, self.c11_w, self.c11_b)
        return (pcm, up_out, c0_out, t2_out, r3_out, t5_out, r6_out, t8_out, r9_out, c11_out, *kv_out,
                pos + self.up)


class TextEmbed(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer('table', table)

    def forward(self, ids):
        return F.embedding(ids, self.table)


def load(src: Path):
    cfg = json.loads((src / 'config.json').read_text())
    w = {k: v.float() for k, v in load_file(str(src / 'model.safetensors')).items()
         if not k.startswith(('speaker_mimi', 'mimi.encoder', 'mimi.downsample'))}
    return cfg, w


def conditions(w: dict, cfg: dict) -> torch.Tensor:
    """The conditioners summed at their defaults (num_speakers 1, duration_delta 0), [D]: what every frame adds."""
    total = torch.zeros(cfg['flow_lm']['d_model'])
    defaults = {'num_speakers': '1', 'duration_delta': '0.0', 'padding_bonus': '0.0'}
    for c in cfg['conditioners']:
        p = f'flow_lm.condition_provider.conditioners.{c["name"]}.'
        value = defaults[c['name']]
        if c['type'] == 'lut':
            emb = w[p + 'embed.weight'][c['lut']['possible_values'].index(value)]
        else:
            cc = c['continuous']
            half = cc['dim'] // 2
            pos = np.float32(cc['scale_factor']) * np.float32(value)
            phases = pos / np.power(np.float32(cc.get('max_period', 10000.0)), np.arange(half, dtype=np.float32) / np.float32(half - 1))
            emb = torch.from_numpy(np.concatenate([np.cos(phases), np.sin(phases)]).astype(np.float32))
        total = total + (F.linear(emb, w[p + 'output_proj.weight']) if p + 'output_proj.weight' in w else emb)
    return total


def voice_embedding(w: dict, path: Path) -> torch.Tensor:
    """A voice file's speaker latents [1, C, T] projected to the flow LM's width: [T, D]."""
    latents = load_file(str(path))['speaker_wavs'].float()
    return F.linear(latents[0].T, w['flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight'])


def build(src: Path):
    cfg, w = load(src)
    flow = FlowLM(w, cfg, conditions(w, cfg)).eval()
    mimi = Mimi(w, cfg).eval()
    text = TextEmbed(w['flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight']).eval()
    voice = voice_embedding(w, src / 'default-voice.safetensors')
    bos = flow.step_emb(w['flow_lm.bos_emb']).reshape(1, 1, -1)
    return cfg, flow, mimi, text, voice, bos


def empty_past(flow: FlowLM) -> list:
    hd = flow.d // flow.heads
    return [torch.zeros(1, flow.heads, 0, hd) for _ in range(2 * len(flow.layers))]


def stats(name, ours, ref):
    a, b = np.asarray(ours, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f'  {name:<34} max abs {np.abs(a - b).max():.3e}  rel {np.abs(a - b).max() / (np.abs(b).max() + 1e-30):.2e}  cosine {cos:.8f}')


def rust_kv(t: torch.Tensor, keys: bool) -> np.ndarray:
    """The runtime's cache rows [1, P, H, HD] in this export's layout [1, H, P, HD], keys in half-split rope order."""
    x = t.numpy().transpose(0, 2, 1, 3)
    return np.concatenate([x[..., 0::2], x[..., 1::2]], -1) if keys else x


class Torch:
    """The reimplementation driven like the graphs, for the parity check."""

    def __init__(self, flow, mimi):
        self.flow, self.mimi = flow, mimi

    def flow_call(self, emb, noise, past):
        out = self.flow(torch.as_tensor(emb), torch.as_tensor(noise), *[torch.as_tensor(p) for p in past])
        return [o.numpy() for o in out[:3]], [o.numpy() for o in out[3:]]

    def mimi_call(self, latent, state):
        out = self.mimi(torch.as_tensor(latent), *[torch.as_tensor(s) for s in state])
        return out[0].numpy(), [o.numpy() for o in out[1:]]


class Onnx:
    def __init__(self, out: Path, flow: str = 'flow_lm', mimi: str = 'mimi_decoder'):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.flow = ort.InferenceSession(str(out / f'{flow}.onnx'), so, providers=['CPUExecutionProvider'])
        self.mimi = ort.InferenceSession(str(out / f'{mimi}.onnx'), so, providers=['CPUExecutionProvider'])

    def flow_call(self, emb, noise, past):
        names = [i.name for i in self.flow.get_inputs()]
        out = self.flow.run(None, dict(zip(names, [np.asarray(emb, np.float32), np.asarray(noise, np.float32), *past])))
        return out[:3], out[3:]

    def mimi_call(self, latent, state):
        names = [i.name for i in self.mimi.get_inputs()]
        out = self.mimi.run(None, dict(zip(names, [np.asarray(latent, np.float32), *state])))
        return out[0], out[1:]


def check(dump_path: Path, runner, flow: FlowLM, mimi: Mimi, text: TextEmbed, voice, bos, cfg, label: str, ref=None):
    """Runs the voice and text prompts, every frame (teacher-forced on the runtime's latents, then free-running on its
    noise) and Mimi on the runtime's latents, comparing with the runtime's dump (and with `ref`, another runner's
    outputs, when given). Returns this runner's outputs."""
    d = load_file(str(dump_path))
    zero = np.zeros((1, cfg['flow_lm']['ldim']), np.float32)
    print(f'{label} vs Rust ({dump_path.name}, {d["latents"].shape[0]} frames):')
    stats('voice embedding', voice.numpy(), d['voice_emb'][0].numpy())
    tokens = d['tokens'].long()[None]
    temb = text(tokens).detach().numpy()
    stats('text embedding', temb, d['text_emb'].numpy())
    with torch.no_grad():
        past = [p.numpy() for p in empty_past(flow)]
        _, past = runner.flow_call(voice.numpy()[None], zero, past)
        for i in (0, len(flow.layers) - 1):
            stats(f'K layer {i} after voice', past[2 * i], rust_kv(d[f'after_voice.k{i}'], True))
            stats(f'V layer {i} after voice', past[2 * i + 1], rust_kv(d[f'after_voice.v{i}'], False))
        _, past = runner.flow_call(temb, zero, past)
        for i in (0, len(flow.layers) - 1):
            stats(f'K layer {i} after text', past[2 * i], rust_kv(d[f'after_text.k{i}'], True))
            stats(f'V layer {i} after text', past[2 * i + 1], rust_kv(d[f'after_text.v{i}'], False))
        prompt = past
        noise, lat_ref, eos_ref = d['noise'].numpy(), d['latents'].numpy(), d['eos_logits'].numpy()
        n = len(eos_ref)
        # Teacher-forced: each frame from the runtime's previous latent.
        lats, eoss, emb, past = [], [], bos.numpy(), prompt
        for i in range(n):
            (lat, eos, _), past = runner.flow_call(emb, noise[i:i + 1], past)
            lats.append(lat.reshape(-1))
            eoss.append(float(eos.reshape(-1)[0]))
            emb = flow.step_emb(torch.from_numpy(lat_ref[i])).reshape(1, 1, -1).numpy()
        stats('latents, teacher-forced', np.stack(lats), lat_ref)
        stats('EOS logits, teacher-forced', np.array(eoss), eos_ref)
        # Free-running: the runtime's noise, our own latents fed back.
        free, free_eos, emb, past = [], [], bos.numpy(), prompt
        for i in range(n):
            (lat, eos, emb), past = runner.flow_call(emb, noise[i:i + 1], past)
            free.append(lat.reshape(-1))
            free_eos.append(float(eos.reshape(-1)[0]))
        stats('latents, free-running', np.stack(free), lat_ref)
        stats('EOS logits, free-running', np.array(free_eos), eos_ref)
        first = lambda e: next((i for i, v in enumerate(e) if v > cfg['eos_threshold']), None)
        print(f'  first EOS frame: free-running {first(free_eos)}, Rust {first(eos_ref)}')
        state = [np.zeros(s, np.float32) for _, s in mimi.state_spec()]
        pcm = []
        for i in range(n):
            p, state = runner.mimi_call(lat_ref[i].reshape(1, 1, -1), state)
            pcm.append(p.reshape(-1))
        stats('Mimi PCM (runtime latents)', np.stack(pcm), d['pcm'].numpy())
    out = {'latents_tf': np.stack(lats), 'eos_tf': np.array(eoss), 'free': np.stack(free), 'pcm': np.stack(pcm)}
    if ref is not None:
        print(f'{label} vs PyTorch:')
        for k in out:
            stats(k, out[k], ref[k])
    return out


def export(out: Path, flow: FlowLM, mimi: Mimi, text: TextEmbed, voice, bos, cfg):
    out.mkdir(parents=True, exist_ok=True)
    nl, hd = len(flow.layers), flow.d // flow.heads
    past_names = [f'past_{kind}_{i}' for i in range(nl) for kind in ('key', 'value')]
    present_names = [n.replace('past_', 'present_') for n in past_names]
    dyn = {'emb': {1: 'sequence'}, **{n: {2: 'past'} for n in past_names}, **{n: {2: 'total'} for n in present_names}}
    sample = (torch.randn(1, 3, flow.d), torch.randn(1, 32), *[torch.randn(1, flow.heads, 5, hd) for _ in past_names])
    with torch.no_grad():
        torch.onnx.export(flow, sample, str(out / 'flow_lm.onnx'), opset_version=OPSET, dynamo=False,
                          input_names=['emb', 'noise', *past_names],
                          output_names=['latent', 'eos_logit', 'next_emb', *present_names], dynamic_axes=dyn)
        spec = mimi.state_spec()
        names = [n for n, _ in spec]
        torch.onnx.export(mimi, (torch.randn(1, 1, 32), *[torch.zeros(s) for _, s in spec]),
                          str(out / 'mimi_decoder.onnx'), opset_version=OPSET, dynamo=False,
                          input_names=['latent', *names], output_names=['pcm', *[n + '_out' for n in names]])
        torch.onnx.export(text, (torch.tensor([[1, 2, 3]]),), str(out / 'text_embed.onnx'), opset_version=OPSET,
                          dynamo=False, input_names=['token_ids'], output_names=['text_emb'],
                          dynamic_axes={'token_ids': {1: 'tokens'}, 'text_emb': {1: 'tokens'}})
    voice.numpy().astype('<f4').tofile(out / 'voice_default.bin')
    bos.numpy().astype('<f4').tofile(out / 'bos_emb.bin')
    for f in ('tokenizer.json', 'config.json'):
        shutil.copy(SRC / f, out / f)
    shutil.copy(RUST / 'LICENSE-APACHE', out / 'LICENSE')
    meta = {
        'sample_rate': cfg['mimi']['sample_rate'], 'frame_rate': cfg['mimi']['frame_rate'],
        'samples_per_frame': int(cfg['mimi']['sample_rate'] / cfg['mimi']['frame_rate']),
        'eos_threshold': cfg['eos_threshold'], 'temperature': 0.3, 'ldim': cfg['flow_lm']['ldim'],
        'd_model': flow.d, 'num_layers': nl, 'num_heads': flow.heads, 'head_dim': hd,
        'files': {'voice_default.bin': {'dtype': 'float32', 'shape': list(voice.shape)},
                  'bos_emb.bin': {'dtype': 'float32', 'shape': list(bos.shape)}},
        'flow_lm': {'inputs': ['emb', 'noise', *past_names], 'outputs': ['latent', 'eos_logit', 'next_emb', *present_names]},
        'mimi_decoder': {'state': [{'name': n, 'shape': s} for n, s in spec]},
    }
    (out / 'phonon.json').write_text(json.dumps(meta, indent=1) + '\n')
    for f in sorted(out.glob('*.onnx')):
        print(f'{f.name}: {f.stat().st_size / 1e6:.1f} MB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=Path, nargs='*', default=[])
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--graphs', nargs='*', default=['flow_lm:mimi_decoder'],
                        help='FLOW:MIMI graph pairs in --out to check, e.g. flow_lm_q4:mimi_decoder_q4c8')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    cfg, flow, mimi, text, voice, bos = build(SRC)
    if not args.no_export:
        export(args.out, flow, mimi, text, voice, bos, cfg)
    for dump in args.check:
        ref = check(dump, Torch(flow, mimi), flow, mimi, text, voice, bos, cfg, 'PyTorch')
        for pair in args.graphs:
            f, m = pair.split(':')
            if (args.out / f'{f}.onnx').exists():
                check(dump, Onnx(args.out, f, m), flow, mimi, text, voice, bos, cfg, f'ONNX {f} + {m}', ref)


if __name__ == '__main__':
    main()
