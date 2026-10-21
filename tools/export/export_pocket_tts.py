"""Exports Kyutai's Pocket TTS (English, `kyutai/pocket-tts` `languages/english`, CC BY 4.0) to ONNX for ONNX Runtime
Web, with export_phonon.py's PyTorch modules (Phonon is built on Pocket TTS: same flow LM, LSD head and Mimi decoder),
checked against Kyutai's own `pocket-tts` Python package, the reference implementation.

What differs from Phonon, all read from the checkpoint and the package (4.0.1):
- a wider flow LM (d_model 1024, 16 heads, 6 layers, FFN 4096); tanh GELU in both transformers;
- the flow head's time embedding normalized by the centred, unbiased variance (`rms_var`);
- no conditioners (a frame's input is input_linear(latent) alone);
- the voice prompt is `bos_before_voice` + the speaker projection of the Mimi-encoded clip (127 positions for alba);
- Mimi's decoder transformer sees a key while query - key < 250 (Phonon's runtime: <= 250);
- EOS when the logit exceeds -4.0, ignored before frame 6; the text front end and tokenizer (see the README).

Same graph set and interface as Phonon, written to exports/tts-int4-gated/pocket-tts/: text_embed.onnx,
flow_lm.onnx (explicit KV cache, noise input, next_emb output), mimi_decoder.onnx (streaming state in and out), the
default voice (alba) as voice_alba.bin, bos_emb.bin and pocket_tts.json.

`--check TEXT` drives the reference model and our modules (PyTorch, then each `--graphs FLOW:MIMI` pair) on the same
noise: the voice and text prompts' KV caches, every frame's latent and EOS logit (teacher-forced on the reference's
latents, then free-running) and Mimi's PCM for the reference's latents.

Run with the pocket-tts venv (it holds the reference, torch, onnx, onnxruntime):
    G=tools/export/.cache/tts-gated/pocket-tts
    HF_HOME=$G/hf $G/venv/bin/python -I tools/export/export_pocket_tts.py [--no-export] [--check TEXT ...]
        [--graphs flow_lm_q4:mimi_decoder_q4c8 ...]
The int4 graphs: quantize_int4.py on flow_lm, text_embed, mimi_decoder (-> *_q4.onnx), on mimi_decoder with
`--conv-bits 8` (-> mimi_decoder_q4c8.onnx) and on flow_lm with `--matmul-bits 8` (-> flow_lm_q8.onnx).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GATED = HERE / '.cache/tts-gated/pocket-tts'
os.environ.setdefault('HF_HOME', str(GATED / 'hf'))
os.environ.setdefault('HF_TOKEN_PATH', str(Path.home() / '.cache/huggingface/token'))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from export_phonon import NEG, FlowLM, Mimi, Onnx, TextEmbed, rust_kv, stats  # noqa: E402,F401

OUT = ROOT / 'exports/tts-int4-gated/pocket-tts'
OPSET = 17
VOICE = 'alba'
MIN_FRAMES_BEFORE_EOS = 6  # TTSModel._MIN_FRAMES_BEFORE_EOS
FADE_IN = 120  # samples: the reference fades the first 5 ms of each chunk in


def reference():
    from pocket_tts import TTSModel
    m = TTSModel.load_model()
    return m


def config(m) -> dict:
    """The reference config in the dict shape export_phonon's modules read."""
    c, mc = m.config.flow_lm, m.config.mimi
    t, mt = c.transformer, mc.transformer
    assert c.flow.type == 'lsd' and c.z_dim is None and c.lookup_table.rank is None and t.attn_dim is None
    assert t.num_kv_heads is None and mc.decoder_seanet is None and mc.decoder_transformer is None
    return {
        'lsd_decode_steps': m.sampler_decode_steps,
        'flow_lm': {'d_model': t.d_model, 'num_heads': t.num_heads, 'num_layers': t.num_layers,
                    'flow_depth': c.flow.depth, 'max_period': float(t.max_period), 'ldim': m.flow_lm.ldim,
                    'time_rms_norm': 'rms_var', 'gelu': 'tanh'},
        'mimi': {'ratios': list(mc.seanet.ratios), 'n_residual_layers': mc.seanet.n_residual_layers,
                 'channels': mc.channels, 'gelu': 'tanh', 'transformer_d_model': mt.d_model,
                 'transformer_num_heads': mt.num_heads, 'transformer_num_layers': mt.num_layers,
                 'transformer_context': mt.context, 'transformer_max_period': mt.max_period,
                 'sample_rate': mc.sample_rate, 'frame_rate': mc.frame_rate, 'context_includes_self': True},
    }


def voice_embedding(m, name: str) -> torch.Tensor:
    """A predefined voice's prompt [T, D]: bos_before_voice + speaker_proj(Mimi latents) of its source clip, as
    TTSModel.get_state_for_audio_prompt builds it from audio (the published voice states are this prompt's KV cache)."""
    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio, end_on_pause
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES, download_if_necessary
    audio, rate = audio_read(download_if_necessary(_ORIGINS_OF_PREDEFINED_VOICES[name]))
    audio = end_on_pause(convert_audio(audio, rate, m.sample_rate, 1), m.sample_rate)
    prompt = torch.cat([m.flow_lm.bos_before_voice, m._encode_audio(audio.unsqueeze(0))], 1)
    return prompt[0].detach()


def build(m):
    cfg = config(m)
    w = {k: v.float() for k, v in m.state_dict().items()
         if not k.startswith(('mimi.encoder', 'mimi.downsample', 'mimi.encoder_transformer'))}
    flow = FlowLM(w, cfg, torch.zeros(cfg['flow_lm']['d_model'])).eval()
    mimi = Mimi(w, cfg).eval()
    text = TextEmbed(w['flow_lm.conditioner.embed.weight']).eval()
    voice = voice_embedding(m, VOICE)
    bos = flow.step_emb(w['flow_lm.bos_emb']).reshape(1, 1, -1)
    return cfg, flow, mimi, text, voice, bos


class Torch:
    def __init__(self, flow, mimi):
        self.flow, self.mimi = flow, mimi

    def flow_call(self, emb, noise, past):
        out = self.flow(torch.as_tensor(emb), torch.as_tensor(noise), *[torch.as_tensor(p) for p in past])
        return [o.numpy() for o in out[:3]], [o.numpy() for o in out[3:]]

    def mimi_call(self, latent, state):
        out = self.mimi(torch.as_tensor(latent), *[torch.as_tensor(s) for s in state])
        return out[0].numpy(), [o.numpy() for o in out[1:]]


class Reference:
    """The pocket-tts model stepped by hand, with our noise and its EOS logits captured."""

    def __init__(self, m):
        self.m, self.queue, self.eos = m, [], []
        m.flow_lm.out_eos.register_forward_hook(lambda mod, inp, out: self.eos.append(float(out.reshape(-1)[-1])))
        original = torch.nn.init.normal_

        def normal_(t, mean=0.0, std=1.0, generator=None):
            if self.queue:
                return t.copy_(torch.as_tensor(self.queue.pop(0)).reshape(t.shape))
            return original(t, mean, std, generator)
        torch.nn.init.normal_ = normal_

    def frame(self, state, latent, noise):
        """latent None: the BOS frame. Returns (latent [1, 1, ldim], EOS logit)."""
        self.queue.append(noise)
        x = torch.full((1, 1, self.m.flow_lm.ldim), float('nan')) if latent is None else torch.as_tensor(latent).reshape(1, 1, -1)
        out, _ = self.m._run_flow_lm_and_increment_step(model_state=state, backbone_input_latents=x)
        return out.numpy(), self.eos[-1]


def ref_kv(state, i: int):
    s = state[f'transformer.layers.{i}.self_attn']
    off = int(s['offset'].view(-1)[0])
    return rust_kv(s['cache'][0][:, :off], True), rust_kv(s['cache'][1][:, :off], False)


def check(m, text: str, runners: list, flow, voice, bos, cfg, frames: int = 60, seed: int = 0):
    import copy

    from pocket_tts.models.text_chunking import prepare_text_prompt
    from pocket_tts.modules.stateful_module import increment_steps, init_states
    ref = Reference(m)
    prepared, _ = prepare_text_prompt(text, m.pad_with_spaces_for_short_inputs, m.remove_semicolons,
                                      m.append_terminal_punctuation, m.capitalize_first_letter, m.replace_characters)
    tokens = m.flow_lm.conditioner.prepare(prepared)
    voice_state = m.get_state_for_audio_prompt(VOICE)
    state = copy.deepcopy(voice_state)
    m._expand_kv_cache(state, 127 + tokens.shape[1] + frames + 1)
    m._run_flow_lm_and_increment_step(model_state=state, text_tokens=tokens)
    prompt_state = copy.deepcopy(state)
    gen = torch.Generator().manual_seed(seed)
    std = m.temp ** 0.5
    noise = [torch.empty(1, 32).normal_(0, std, generator=gen).numpy() for _ in range(frames)]
    # Reference, free-running from BOS.
    lat_ref, eos_ref, latent = [], [], None
    for i in range(frames):
        latent, eos = ref.frame(state, latent, noise[i])
        lat_ref.append(latent.reshape(-1))
        eos_ref.append(eos)
    lat_ref = np.stack(lat_ref)
    nl = cfg['flow_lm']['num_layers']
    hd = cfg['flow_lm']['d_model'] // cfg['flow_lm']['num_heads']
    zero = np.zeros((1, 32), np.float32)
    first = lambda e: next((i for i, v in enumerate(e) if i >= MIN_FRAMES_BEFORE_EOS and v > m.eos_threshold), None)
    print(f'Reference: {tokens.shape[1]} tokens "{prepared}", first EOS frame {first(eos_ref)} of {frames}')
    mstate = init_states(m.mimi, batch_size=1, sequence_length=frames * 16)
    pcm_ref = []
    with torch.no_grad():
        for i in range(frames):
            x = torch.from_numpy(lat_ref[i]).reshape(1, 1, -1) * m.flow_lm.emb_std + m.flow_lm.emb_mean
            pcm_ref.append(m.mimi.decode_from_latent(x, mstate).reshape(-1).numpy())
            increment_steps(m.mimi, mstate, increment=16)
    pcm_ref = np.stack(pcm_ref)
    for label, runner in runners:
        print(f'{label} vs reference:')
        with torch.no_grad():
            past = [np.zeros((1, cfg['flow_lm']['num_heads'], 0, hd), np.float32) for _ in range(2 * nl)]
            _, past = runner.flow_call(voice.numpy()[None], zero, past)
            for i in (0, nl - 1):
                k, v = ref_kv(voice_state, i)
                stats(f'K layer {i} after voice', past[2 * i], k)
                stats(f'V layer {i} after voice', past[2 * i + 1], v)
            temb = runner_text(runner, tokens)
            _, past = runner.flow_call(temb, zero, past)
            for i in (0, nl - 1):
                k, v = ref_kv(prompt_state, i)
                stats(f'K layer {i} after text', past[2 * i], k)
                stats(f'V layer {i} after text', past[2 * i + 1], v)
            prompt = past
            lats, eoss, emb, past = [], [], bos.numpy(), prompt
            for i in range(frames):
                (lat, eos, _), past = runner.flow_call(emb, noise[i], past)
                lats.append(lat.reshape(-1))
                eoss.append(float(eos.reshape(-1)[0]))
                emb = flow.step_emb(torch.from_numpy(lat_ref[i])).reshape(1, 1, -1).numpy()
            stats('latents, teacher-forced', np.stack(lats), lat_ref)
            stats('EOS logits, teacher-forced', np.array(eoss), eos_ref)
            free, free_eos, emb, past = [], [], bos.numpy(), prompt
            for i in range(frames):
                (lat, eos, emb), past = runner.flow_call(emb, noise[i], past)
                free.append(lat.reshape(-1))
                free_eos.append(float(eos.reshape(-1)[0]))
            stats('latents, free-running', np.stack(free), lat_ref)
            stats('EOS logits, free-running', np.array(free_eos), eos_ref)
            print(f'  first EOS frame: free-running {first(free_eos)}, reference {first(eos_ref)}')
            st = [np.zeros(s, np.float32) for _, s in runner_mimi_spec(runner)]
            pcm = []
            for i in range(frames):
                p, st = runner.mimi_call(lat_ref[i].reshape(1, 1, -1), st)
                pcm.append(p.reshape(-1))
            stats('Mimi PCM (reference latents)', np.stack(pcm), pcm_ref)


TEXT_EMBED = {}
MIMI_SPEC = []


def runner_text(runner, tokens):
    if isinstance(runner, Torch):
        return TEXT_EMBED['torch'](tokens).numpy()
    import onnxruntime as ort
    sess = TEXT_EMBED.setdefault(runner.text_path, ort.InferenceSession(str(runner.text_path), providers=['CPUExecutionProvider']))
    return sess.run(None, {'token_ids': tokens.numpy().astype(np.int64)})[0]


def runner_mimi_spec(runner):
    return MIMI_SPEC


def export(out: Path, m, flow: FlowLM, mimi: Mimi, text: TextEmbed, voice, bos, cfg):
    from pocket_tts.utils.utils import download_if_necessary
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
    voice.numpy().astype('<f4').tofile(out / f'voice_{VOICE}.bin')
    bos.numpy().astype('<f4').tofile(out / 'bos_emb.bin')
    shutil.copy(download_if_necessary(str(m.config.flow_lm.lookup_table.tokenizer_path)), out / 'tokenizer.json')
    mc = m.config
    meta = {
        'sample_rate': mc.mimi.sample_rate, 'frame_rate': mc.mimi.frame_rate,
        'samples_per_frame': int(mc.mimi.sample_rate / mc.mimi.frame_rate),
        'eos_threshold': m.eos_threshold, 'min_frames_before_eos': MIN_FRAMES_BEFORE_EOS,
        'temperature': m.temp, 'max_tokens_per_chunk': 50, 'fade_in_samples': FADE_IN,
        'replace_characters': mc.replace_characters, 'pad_with_spaces_for_short_inputs': mc.pad_with_spaces_for_short_inputs,
        'ldim': cfg['flow_lm']['ldim'], 'd_model': flow.d, 'num_layers': nl, 'num_heads': flow.heads, 'head_dim': hd,
        'default_voice': VOICE,
        'files': {f'voice_{VOICE}.bin': {'dtype': 'float32', 'shape': list(voice.shape)},
                  'bos_emb.bin': {'dtype': 'float32', 'shape': list(bos.shape)}},
        'flow_lm': {'inputs': ['emb', 'noise', *past_names], 'outputs': ['latent', 'eos_logit', 'next_emb', *present_names]},
        'mimi_decoder': {'state': [{'name': n, 'shape': s} for n, s in spec]},
    }
    (out / 'pocket_tts.json').write_text(json.dumps(meta, indent=1) + '\n')
    for f in sorted(out.glob('*.onnx')):
        print(f'{f.name}: {f.stat().st_size / 1e6:.1f} MB')


class OnnxRunner(Onnx):
    def __init__(self, out: Path, flow: str, mimi: str, text: str):
        super().__init__(out, flow, mimi)
        self.text_path = out / f'{text}.onnx'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', nargs='*', default=[])
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--graphs', nargs='*', default=['flow_lm:mimi_decoder:text_embed'],
                        help='FLOW:MIMI[:TEXT] graph sets in --out to check, e.g. flow_lm_q4:mimi_decoder_q4c8:text_embed_q4')
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    m = reference()
    cfg, flow, mimi, text, voice, bos = build(m)
    TEXT_EMBED['torch'] = text
    MIMI_SPEC[:] = mimi.state_spec()
    if not args.no_export:
        export(args.out, m, flow, mimi, text, voice, bos, cfg)
    for t in args.check:
        runners = [('PyTorch', Torch(flow, mimi))]
        for g in args.graphs:
            f, mm, *tx = g.split(':')
            if (args.out / f'{f}.onnx').exists():
                runners.append((f'ONNX {g}', OnnxRunner(args.out, f, mm, tx[0] if tx else 'text_embed')))
        check(m, t, runners, flow, voice, bos, cfg, args.frames)


if __name__ == '__main__':
    main()
