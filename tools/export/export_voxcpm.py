"""VoxCPM2 (OpenBMB, 2B, 48 kHz) split into graphs for ONNX Runtime Web.

VoxCPM2 generates continuous audio latents in patches (4 frames of 64-dim AudioVAE latents per step): a MiniCPM-4
language model ("base LM", 28 layers) and a small residual LM (8 layers, no RoPE) read text and the previous patches;
a local diffusion transformer (LocDiT, conditional flow matching, 10 Euler steps with classifier-free guidance)
samples the next patch; a local encoder (LocEnc) embeds that patch for the next step; a stop head decides when to end;
the AudioVAE decoder turns the latents into 48 kHz audio. All graphs are exported with torch.onnx (float; quantize
with quantize_int4.py). MiniCPM-4 here is a Llama-style decoder with LongRoPE (without muP), so the two LMs are
written out layer by layer with an explicit KV cache:

- `text_embed`: ids -> base-LM input embeddings.
- `base_lm`: inputs_embeds [1, S, 2048] + past_key_values.{i}.key/value [1, 2, past, 128] -> hidden (final norm) +
  present.*. Positions continue from the past length.
- `residual_lm`: the same for the residual LM (no positions).
- `bridge`: base-LM hidden, the step's patch embedding and its audio mask -> the base-LM state passed on (scalar
  quantization layer on audio positions), the residual LM's input, and the stop head's logits.
- `feat_encoder`: patches [1, T, 4, 64] -> their embeddings [1, T, 2048] (LocEnc + projection).
- `dit_step`: one Euler step of the LocDiT sampler (guided velocity, CFG-zero*); the caller runs the 10 steps.
  (`dit_sample`, the 10 steps unrolled, is kept for checks: it repeats the DiT weights 10 times.)
- `vae_decoder`: latents [1, 64, frames] -> audio [1, 1, frames * 1920] at 48 kHz (causal).

    vox-venv/bin/python tools/export/export_voxcpm.py text_embed|base_lm|residual_lm|bridge|feat_encoder|dit_sample|vae_decoder SRC OUT
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STEPS = 10  # inference_timesteps default


def manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """scaled_dot_product_attention written in plain ops (with GQA), for the TorchScript exporter."""
    if enable_gqa and k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    s = (q @ k.transpose(-1, -2)) * (scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    if is_causal:
        i = torch.arange(q.shape[-2]).unsqueeze(1)
        j = torch.arange(k.shape[-2]).unsqueeze(0)
        s = s.masked_fill(j > i + (k.shape[-2] - q.shape[-2]), float('-inf'))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else s + attn_mask
    return torch.softmax(s, dim=-1) @ v


def load(src: Path):
    F.scaled_dot_product_attention = manual_sdpa
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    m = VoxCPM2Model.from_local(str(src), optimize=False, device='cpu')
    return m.float().eval()


def export(module, args, out: Path, inputs, outputs, dynamic=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, args, str(out), input_names=inputs, output_names=outputs, opset_version=17,
                          dynamic_axes=dynamic or {}, dynamo=False)
    print('exported', out.name)


class LM(torch.nn.Module):
    """A MiniCPMModel run with an explicit KV cache (past -> present), positions past..past+S-1."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, x, *past):
        lm = self.lm
        n = len(lm.layers)
        p = past[0].shape[2]
        s = x.shape[1]
        cs = None
        if lm.rope_emb is not None:
            pos = torch.arange(s) + p
            cs = lm.rope_emb(pos)
        i = torch.arange(s).unsqueeze(1) + p
        j = torch.arange(s).unsqueeze(0) + p
        j = torch.cat([torch.arange(p).unsqueeze(0).expand(1, -1), j], dim=1)
        mask = (j <= i)[None, None]
        presents = []
        h = x
        for li, layer in enumerate(lm.layers):
            a = layer.self_attn
            r = h
            y = layer.input_layernorm(h)
            q = a.q_proj(y).view(1, s, a.num_heads, a.head_dim).transpose(1, 2)
            k = a.k_proj(y).view(1, s, a.num_key_value_heads, a.head_dim).transpose(1, 2)
            v = a.v_proj(y).view(1, s, a.num_key_value_heads, a.head_dim).transpose(1, 2)
            if cs is not None:
                from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cs[0], cs[1])
            k = torch.cat([past[2 * li], k], dim=2)
            v = torch.cat([past[2 * li + 1], v], dim=2)
            presents += [k, v]
            o = manual_sdpa(q, k, v, attn_mask=mask, enable_gqa=True)
            h = r + a.o_proj(o.transpose(1, 2).reshape(1, s, a.num_heads * a.head_dim))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            assert not layer.use_mup
        return (lm.norm(h), *presents)


def lm_export(lm, out: Path, check_fn=None):
    w = LM(lm).eval()
    n = len(lm.layers)
    kvh = lm.config.num_key_value_heads
    hd = lm.config.kv_channels or lm.config.hidden_size // lm.config.num_attention_heads
    x = torch.randn(1, 5, lm.config.hidden_size) * 0.1
    past = [torch.randn(1, kvh, 3, hd) * 0.1 for _ in range(2 * n)]
    # Check: 8 tokens at once (the model's own forward) == 3 cached + 5.
    with torch.no_grad():
        full = torch.randn(1, 8, lm.config.hidden_size) * 0.1
        ref, kv = lm(full, is_causal=True)
        cached = [t[:, :, :3] for pair in kv for t in pair]
        mine = w(full[:, 3:], *cached)[0]
        print('LM wrapper vs forward (3 cached + 5): max abs diff', float((mine - ref[:, 3:]).abs().max()))
    names_in = ['inputs_embeds'] + [f'past_key_values.{i}.{t}' for i in range(n) for t in ('key', 'value')]
    names_out = ['hidden'] + [f'present.{i}.{t}' for i in range(n) for t in ('key', 'value')]
    dyn = {'inputs_embeds': {1: 'seq'}, 'hidden': {1: 'seq'}}
    dyn.update({k: {2: 'past'} for k in names_in[1:]})
    dyn.update({k: {2: 'total'} for k in names_out[1:]})
    export(w, (x, *past), out, names_in, names_out, dyn)


def text_embed(src, out):
    from export_qwen3_tts import gather_graph
    m = load(src)
    scale = m.config.lm_config.scale_emb if m.config.lm_config.use_mup else 1.0
    out.parent.mkdir(parents=True, exist_ok=True)
    gather_graph(out, {'embeds': (m.base_lm.embed_tokens.weight.detach() * scale).numpy()}, {})


def base_lm(src, out):
    lm_export(load(src).base_lm, out)


def residual_lm(src, out):
    lm_export(load(src).residual_lm, out)


def bridge(src, out):
    m = load(src)

    class B(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, hidden, feat_embed, audio_mask):
            mk = audio_mask.unsqueeze(-1)
            lm_hidden = self.m.fsq_layer(hidden) * mk + hidden * (1 - mk)
            res_in = self.m.fusion_concat_proj(torch.cat([lm_hidden, mk * feat_embed], dim=-1))
            stop = self.m.stop_head(self.m.stop_actn(self.m.stop_proj(lm_hidden)))
            return lm_hidden, res_in, stop
    h = m.config.lm_config.hidden_size
    export(B().eval(), (torch.randn(1, 3, h), torch.randn(1, 3, h), torch.ones(1, 3)), out,
           ['hidden', 'feat_embed', 'audio_mask'], ['lm_hidden', 'residual_input', 'stop_logits'],
           {k: {1: 'seq'} for k in ('hidden', 'feat_embed', 'audio_mask', 'lm_hidden', 'residual_input', 'stop_logits')})


def feat_encoder(src, out):
    m = load(src)

    class E(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, feats):
            return self.m.enc_to_lm_proj(self.m.feat_encoder(feats))
    p, d = m.patch_size, m.config.feat_dim
    x = torch.randn(1, 3, p, d)
    with torch.no_grad():
        print('feat_encoder sample norm', float(E()(x).norm()))
    export(E().eval(), (x,), out, ['feats'], ['embeds'], {'feats': {1: 'frames'}, 'embeds': {1: 'frames'}})


def dit_sample(src, out):
    m = load(src)
    dec = m.feat_decoder
    p, d = m.patch_size, m.config.feat_dim

    class S(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, lm_hidden, residual_hidden, cond, noise, cfg_value):
            mu = torch.cat([self.m.lm_to_dit_proj(lm_hidden), self.m.res_to_dit_proj(residual_hidden)], dim=-1)
            t_span = torch.linspace(1, 0, STEPS + 1)
            t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
            x = dec.solve_euler(x=noise, t_span=t_span, mu=mu, cond=cond.transpose(1, 2), cfg_value=cfg_value, use_cfg_zero_star=True)
            return x.transpose(1, 2)

    h = m.config.lm_config.hidden_size
    g = torch.Generator().manual_seed(0)
    lh, rh, cond = torch.randn(1, h, generator=g), torch.randn(1, h, generator=g), torch.randn(1, p, d, generator=g) * 0.5
    with torch.no_grad():
        torch.manual_seed(3)
        mu = torch.cat([m.lm_to_dit_proj(lh), m.res_to_dit_proj(rh)], dim=-1)
        ref = dec(mu=mu, patch_size=p, cond=cond.transpose(1, 2), n_timesteps=STEPS, cfg_value=2.0).transpose(1, 2)
        torch.manual_seed(3)
        z = torch.randn(1, d, p)
        mine = S()(lh, rh, cond, z, torch.tensor(2.0))
        print('sampler vs feat_decoder: max abs diff', float((ref - mine).abs().max()))
    export(S().eval(), (lh, rh, cond, z, torch.tensor(2.0)), out, ['lm_hidden', 'residual_hidden', 'cond', 'noise', 'cfg_value'], ['patch'])


def dit_step(src, out):
    """One Euler step's guided velocity: x [1, 64, 4], lm_hidden, residual_hidden [1, H], cond [1, 4, 64], t [1],
    cfg_value [] -> v [1, 64, 4] (CFG-zero* as solve_euler). The caller runs x <- x - dt * v over the schedule in the
    README (velocity 0 on the first step); the whole 10-step sampler unrolled shares its weights 10 times, which the
    quantizer cannot handle."""
    m = load(src)
    dec = m.feat_decoder

    class V(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, x, lm_hidden, residual_hidden, cond, t, cfg_value):
            mu = torch.cat([self.m.lm_to_dit_proj(lm_hidden), self.m.res_to_dit_proj(residual_hidden)], dim=-1)
            x_in = torch.cat([x, x])
            mu_in = torch.cat([mu, torch.zeros_like(mu)])
            t_in = torch.cat([t, t])
            c = cond.transpose(1, 2)
            v = dec.estimator(x_in, mu_in, t_in, torch.cat([c, c]), torch.zeros_like(t_in))
            pos, neg = v[:1], v[1:]
            st = (pos * neg).sum(dim=(1, 2), keepdim=True) / ((neg * neg).sum(dim=(1, 2), keepdim=True) + 1e-8)
            return neg * st + cfg_value * (pos - neg * st)

    p, d, h = m.patch_size, m.config.feat_dim, m.config.lm_config.hidden_size
    g = torch.Generator().manual_seed(0)
    lh, rh, cond = torch.randn(1, h, generator=g), torch.randn(1, h, generator=g), torch.randn(1, p, d, generator=g) * 0.5
    # Check: the step loop reproduces feat_decoder.
    with torch.no_grad():
        torch.manual_seed(3)
        mu = torch.cat([m.lm_to_dit_proj(lh), m.res_to_dit_proj(rh)], dim=-1)
        ref = dec(mu=mu, patch_size=p, cond=cond.transpose(1, 2), n_timesteps=STEPS, cfg_value=2.0)
        torch.manual_seed(3)
        x = torch.randn(1, d, p)
        ts = torch.linspace(1, 0, STEPS + 1)
        ts = ts + (torch.cos(torch.pi / 2 * ts) - 1 + ts)
        for k in range(STEPS):
            dt = ts[k] - ts[k + 1]
            if k > 0:
                x = x - dt * V()(x, lh, rh, cond, ts[k:k + 1], torch.tensor(2.0))
        print('step loop vs feat_decoder: max abs diff', float((ref - x).abs().max()), 'schedule', [round(float(v), 6) for v in ts])
    export(V().eval(), (x, lh, rh, cond, ts[3:4], torch.tensor(2.0)), out,
           ['x', 'lm_hidden', 'residual_hidden', 'cond', 't', 'cfg_value'], ['v'])


def vae_decoder(src, out):
    m = load(src)
    vae = m.audio_vae

    class D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)
    z = torch.randn(1, m.config.feat_dim, 24) * 0.5
    with torch.no_grad():
        a = D()(z)
    print('vae out', tuple(a.shape))
    export(D().eval(), (z,), out, ['latents'], ['audio'], {'latents': {2: 'frames'}, 'audio': {2: 'samples'}})
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    z2 = torch.randn(1, m.config.feat_dim, 40) * 0.5
    with torch.no_grad():
        r = D()(z2).numpy()
    o = s.run(None, {'latents': z2.numpy()})[0]
    print('onnx vs torch (40 frames): max abs diff', float(np.abs(o - r).max()), o.shape)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    cmd, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {'text_embed': text_embed, 'base_lm': base_lm, 'residual_lm': residual_lm, 'bridge': bridge, 'feat_encoder': feat_encoder,
     'dit_sample': dit_sample, 'dit_step': dit_step, 'vae_decoder': vae_decoder}[cmd](src, out)


if __name__ == '__main__':
    main()
