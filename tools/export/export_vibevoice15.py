"""VibeVoice 1.5B (long-form, multi-speaker TTS) split into graphs for ONNX Runtime Web.

A Qwen2.5-1.5B language model reads a script with voice prompts and emits, per step, one of four tokens: speech
start, speech diffusion (one 64-dim acoustic latent, 7.5 per second), speech end, end of text. At each diffusion step a
small diffusion head samples the latent from the LM's last hidden state (classifier-free guidance against a second,
"negative" LM cache), a causal VAE decoder turns it into 3200 samples of 24 kHz audio, a semantic encoder reads that
audio back, and the two connectors' sum is the LM's next input. Graphs (float; quantize_int4.py makes them int4):

- `hf`: the LM as a Qwen2ForCausalLM checkpoint for ONNX Runtime GenAI's model builder (exclude_embeds,
  include_hidden_states). Generation is constrained to the four control tokens, so its LM head is just their four
  rows of the (tied) embedding: logits [.., 4] in the order speech start, speech end, speech diffusion, end of text.
- `embed`: text ids -> the LM's input embeddings.
- `voice_encoder`: a reference clip (24 kHz, zero-padded to a multiple of 3200 samples) -> acoustic latent means.
- `connector`: acoustic latent [1, n, 64] -> acoustic connector output (prompt latents go in as (x + bias) * scale).
- `diffusion`: one latent from condition, negative condition, noise and cfg scale (10 DPM-Solver++ steps unrolled).
- `decoder_step`: one latent -> 3200 samples, streaming state explicit (as export_vibevoice.py).
- `semantic_step`: 3200 samples -> next LM input = acoustic connector(latent) + semantic connector(semantic encoder of
  the samples), streaming state explicit.

    vv15-venv/bin/python tools/export/export_vibevoice15.py hf|embed|voice_encoder|connector|diffusion|decoder_step|semantic_step SRC OUT
"""

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_vibevoice import export  # noqa: E402

TOKENIZER_FILES = ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'tokenizer.json')
STEPS = 10  # the demo's set_ddpm_inference_steps(num_steps=10)


def load(src: Path):
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    m = VibeVoiceForConditionalGenerationInference.from_pretrained(str(src), torch_dtype=torch.float32, device_map='cpu',
                                                                  attn_implementation='eager')
    return m.eval()


def control_ids(tok_dir: Path):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(tok_dir))
    return [t.convert_tokens_to_ids(x) for x in ('<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|endoftext|>')]


def hf(src: Path, out: Path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    tok = Path(sys.argv[4])
    cfg = json.loads((src / 'config.json').read_text())['decoder_config']
    index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    t = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(src / shard), 'pt') as f:
            for k in f.keys():
                if k.startswith('model.language_model.'):
                    t['model.' + k[len('model.language_model.'):]] = f.get_tensor(k).contiguous()
    ids = control_ids(tok)
    head = t['model.embed_tokens.weight'][ids].clone()
    t['lm_head.weight'] = head
    t['model.embed_tokens.weight'] = head.clone()  # unused (exclude_embeds); the vocab must match the head
    out.mkdir(parents=True, exist_ok=True)
    save_file(t, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    c = {k: cfg[k] for k in ('hidden_act', 'hidden_size', 'intermediate_size', 'max_position_embeddings', 'num_attention_heads',
                             'num_hidden_layers', 'num_key_value_heads', 'rms_norm_eps', 'rope_theta')}
    c.update(architectures=['Qwen2ForCausalLM'], model_type='qwen2', vocab_size=len(ids), tie_word_embeddings=False,
             torch_dtype='bfloat16', use_sliding_window=False, sliding_window=None, max_window_layers=cfg['num_hidden_layers'],
             bos_token_id=0, eos_token_id=3)
    (out / 'config.json').write_text(json.dumps(c, indent=1))
    for name in TOKENIZER_FILES:
        if (tok / name).exists():
            shutil.copy(tok / name, out / name)
    print(len(t), 'tensors; control ids', ids)


def embed(src: Path, out: Path):
    from export_qwen3_tts import gather_graph
    from safetensors import safe_open
    index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    key = 'model.language_model.embed_tokens.weight'
    with safe_open(str(src / index[key]), 'pt') as f:
        e = f.get_tensor(key).float().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    gather_graph(out, {'embeds': e}, {})


def no_extra_padding(module):
    from vibevoice.modular import modular_vibevoice_tokenizer as mt
    mt.get_extra_padding_for_conv1d = lambda *a, **k: 0


def voice_encoder(src: Path, out: Path):
    """audio [1, samples] (24 kHz, normalized as the processor does, zero-padded to a multiple of 3200) -> mean
    [1, samples / 3200, 64]. Sample as the original: x = mean + (randn() * 0.5 / 0.8) * randn_like(mean)."""
    m = load(src)
    enc = m.model.acoustic_tokenizer
    wav = torch.randn(1, 3200 * 20) * 0.1
    with torch.no_grad():
        r = enc.encode(wav[:, None]).mean
    no_extra_padding(enc)  # exact for inputs that are a multiple of 3200 samples

    class E(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = enc

        def forward(self, audio):
            return self.e.encode(audio[:, None]).mean

    with torch.no_grad():
        g = E()(wav)
    print('wrapper vs encode: max abs diff', float((r - g).abs().max()), tuple(g.shape))
    export(E().eval(), (wav,), out, ['audio'], ['mean'], {'audio': {1: 'samples'}, 'mean': {1: 'frames'}})


def connector(src: Path, out: Path):
    m = load(src)

    class Cn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = m.model.acoustic_connector

        def forward(self, latent):
            return self.c(latent)
    export(Cn().eval(), (torch.randn(1, 3, 64),), out, ['latent'], ['embeds'], {'latent': {1: 'n'}, 'embeds': {1: 'n'}})
    print('speech_scaling_factor', m.model.speech_scaling_factor.item(), 'speech_bias_factor', m.model.speech_bias_factor.item())


def diffusion(src: Path, out: Path):
    m = load(src)
    head = m.model.prediction_head
    sched = copy.deepcopy(m.model.noise_scheduler)
    h = m.config.decoder_config.hidden_size

    class Sampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, condition, neg_condition, noise, cfg_scale):
            s = copy.deepcopy(sched)
            s.set_timesteps(STEPS)
            cond = torch.cat([condition, neg_condition], dim=0)
            speech = noise
            for t in s.timesteps:
                both = torch.cat([speech, speech], dim=0)
                eps = self.head(both, t.repeat(2).to(both), condition=cond)
                c, u = eps[:1], eps[1:]
                speech = s.step(u + cfg_scale * (c - u), t, speech).prev_sample
            return speech

    w = Sampler().eval()
    g = torch.Generator().manual_seed(0)
    cond, neg = torch.randn(1, h, generator=g), torch.randn(1, h, generator=g)
    with torch.no_grad():
        m.set_ddpm_inference_steps(STEPS)
        torch.manual_seed(1)
        ref = m.sample_speech_tokens(cond, neg, cfg_scale=1.3)
        torch.manual_seed(1)
        nz = torch.randn(2, 64)[:1]
        print('sampler vs sample_speech_tokens: max abs diff', float((ref - w(cond, neg, nz, torch.tensor([1.3]))).abs().max()))
    export(w, (cond, neg, nz, torch.tensor([1.3])), out, ['condition', 'neg_condition', 'noise', 'cfg_scale'], ['latent'])


class FakeCache:
    def __init__(self, states):
        self.s, self.new = states, {}

    def get(self, layer_id, idx):
        return self.s.get(layer_id)

    def set(self, layer_id, idx, value):
        self.new[layer_id] = value


def stateful_layers(module):
    from vibevoice.modular.modular_vibevoice_tokenizer import SConv1d, SConvTranspose1d
    layers = [l for l in module.modules() if isinstance(l, (SConv1d, SConvTranspose1d)) and (isinstance(l, SConvTranspose1d) or l.context_size > 0)]
    shapes = [(l.in_channels, l.context_size if isinstance(l, SConv1d) else l.kernel_size - 1) for l in layers]
    return layers, shapes


def decoder_step(src: Path, out: Path):
    m = load(src)
    dec = m.model.acoustic_tokenizer.decoder
    scale, bias = m.model.speech_scaling_factor.item(), m.model.speech_bias_factor.item()
    layers, shapes = stateful_layers(dec)

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dec = dec

        def forward(self, latent, *states):
            cache = FakeCache({l.layer_id: s for l, s in zip(layers, states)})
            x = (latent / scale - bias).permute(0, 2, 1)
            audio = self.dec(x, cache=cache, sample_indices=torch.tensor([0]), use_cache=True)
            return (audio, *[cache.new[l.layer_id] for l in layers])
    names = [f'state_{i}' for i in range(len(layers))]
    zeros = [torch.zeros(1, c, n) for c, n in shapes]
    export(Step().eval(), (torch.randn(1, 1, 64), *zeros), out, ['latent'] + names, ['audio'] + [f'new_{n}' for n in names])
    (out.parent / (out.stem + '_states.json')).write_text(json.dumps({'states': [[1, c, n] for c, n in shapes]}))


def semantic_step(src: Path, out: Path):
    """latent [1, 1, 64] (as sampled) + audio [1, 1, 3200] (that latent's decoded samples) + state_i -> embeds
    [1, 1, H] (acoustic connector(latent) + semantic connector(semantic encoder(audio))) + new_state_i."""
    m = load(src)
    enc = m.model.semantic_tokenizer
    layers, shapes = stateful_layers(enc.encoder)

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.ac, self.sc = enc, m.model.acoustic_connector, m.model.semantic_connector

        def forward(self, latent, audio, *states):
            cache = FakeCache({l.layer_id: s for l, s in zip(layers, states)})
            sem = self.enc.encode(audio, cache=cache, sample_indices=torch.tensor([0]), use_cache=True).mean
            return (self.ac(latent) + self.sc(sem), *[cache.new[l.layer_id] for l in layers])

    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
    w = Step().eval()
    lat, aud = torch.randn(4, 1, 1, 64), torch.randn(4, 1, 1, 3200) * 0.1
    with torch.no_grad():
        cache = VibeVoiceTokenizerStreamingCache()
        ref = [m.model.acoustic_connector(l) + m.model.semantic_connector(
            enc.encode(a, cache=cache, sample_indices=torch.tensor([0]), use_cache=True).mean) for l, a in zip(lat, aud)]
        st, err = [torch.zeros(1, c, n) for c, n in shapes], 0.0
        for l, a, r in zip(lat, aud, ref):
            e, *st = w(l, a, *st)
            err = max(err, float((e - r).abs().max()))
    print('step vs streaming encode: max abs diff', err)
    names = [f'state_{i}' for i in range(len(layers))]
    zeros = [torch.zeros(1, c, n) for c, n in shapes]
    export(w, (lat[0], aud[0], *zeros), out, ['latent', 'audio'] + names, ['embeds'] + [f'new_{n}' for n in names])
    (out.parent / (out.stem + '_states.json')).write_text(json.dumps({'states': [[1, c, n] for c, n in shapes]}))


def main():
    cmd, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {'hf': hf, 'embed': embed, 'voice_encoder': voice_encoder, 'connector': connector, 'diffusion': diffusion,
     'decoder_step': decoder_step, 'semantic_step': semantic_step}[cmd](src, out)


if __name__ == '__main__':
    main()
