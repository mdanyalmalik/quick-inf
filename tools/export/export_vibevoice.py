"""VibeVoice Realtime 0.5B (streaming TTS) split into graphs for ONNX Runtime Web.

The model is a Qwen2.5-0.5B cut in two: the lower 4 layers ("language model") read the text, the upper 20 layers
("TTS language model") read the text's hidden states and the generated speech latents, and their last hidden state
conditions a small diffusion head that samples the next 64-dim acoustic latent (7.5 per second), which a causal
convolutional VAE decoder turns into 3200 samples of 24 kHz audio. Graphs (float; quantize_int4.py makes them int4):

- `hf`: the TTS language model as a Qwen2ForCausalLM checkpoint for ONNX Runtime GenAI's model builder (exclude_embeds,
  exclude_lm_head: inputs_embeds -> final-norm hidden states, float16 KV cache).
- `text_lm`: the 4 text layers with an explicit KV cache: input_ids -> hidden states + the "text" type embedding,
  i.e. exactly the TTS language model's input for those tokens. Its final norm is an identity, which the builder
  cannot express, so it is exported directly.
- `diffusion`: one call samples one latent: classifier-free guidance (positive and negative condition) and the 5-step
  DPM-Solver++ schedule unrolled, from the caller's Gaussian noise.
- `connector`: latent -> the TTS language model's next input (acoustic connector + the "speech" type embedding).
- `eos`: final hidden state -> end-of-speech probability.
- `decoder_step`: one latent -> 3200 samples, with every causal convolution's left context as explicit state
  (zeros at the start), i.e. the original streaming decode.
- `voices`: a voice preset (.pt: the prompt's KV caches) -> .npz.

    vv-venv/bin/python tools/export/export_vibevoice.py hf|text_lm|diffusion|connector|eos|decoder_step SRC OUT
    vv-venv/bin/python tools/export/export_vibevoice.py voices PRESET.pt OUT.npz
"""

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

TOKENIZER_FILES = ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'tokenizer.json')


def load(src: Path):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(str(src), torch_dtype=torch.float32, device_map='cpu',
                                                                           attn_implementation='eager')
    return m.eval()


def export(module, args, out: Path, inputs, outputs, dynamic=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, args, str(out), input_names=inputs, output_names=outputs, opset_version=17,
                          dynamic_axes=dynamic or {}, dynamo=False)


def hf(src: Path, out: Path):
    """SRC OUT [TOKENIZER_DIR]: the tokenizer files (Qwen2.5's) are copied in for the builder."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    cfg = json.loads((src / 'config.json').read_text())
    dc = cfg['decoder_config']
    prefix = 'model.tts_language_model.'
    t = {}
    with safe_open(str(src / 'model.safetensors'), 'pt') as f:
        for k in f.keys():
            if k.startswith(prefix):
                t['model.' + k[len(prefix):]] = f.get_tensor(k).contiguous()
    t['lm_head.weight'] = t['model.embed_tokens.weight'].clone()
    out.mkdir(parents=True, exist_ok=True)
    save_file(t, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    c = {k: dc[k] for k in ('hidden_act', 'hidden_size', 'intermediate_size', 'max_position_embeddings', 'num_attention_heads',
                            'num_key_value_heads', 'rms_norm_eps', 'rope_theta', 'vocab_size')}
    c.update(architectures=['Qwen2ForCausalLM'], model_type='qwen2', num_hidden_layers=cfg['tts_backbone_num_hidden_layers'],
             tie_word_embeddings=False, torch_dtype='bfloat16', use_sliding_window=False, sliding_window=None,
             max_window_layers=cfg['tts_backbone_num_hidden_layers'], bos_token_id=151643, eos_token_id=151643)
    (out / 'config.json').write_text(json.dumps(c, indent=1))
    tok = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    for name in TOKENIZER_FILES:
        if tok and (tok / name).exists():
            shutil.copy(tok / name, out / name)
    print(len(t), 'tensors')


def text_lm(src: Path, out: Path):
    """input_ids [1, seq], past_key_values.{i}.key/value [1, 2, past, 64] -> hidden [1, seq, 896] (+ text type
    embedding), present.{i}.key/value."""
    from transformers.cache_utils import DynamicCache
    m = load(src)
    lm = m.model.language_model
    n = len(lm.layers)
    type1 = m.model.tts_input_types.weight[1].detach()

    class TextLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = lm
            self.register_buffer('type1', type1)

        def forward(self, input_ids, *past):
            p = past[0].shape[2]
            s = input_ids.shape[1]
            cache = DynamicCache()
            for i in range(n):
                cache.update(past[2 * i], past[2 * i + 1], i)
            x = self.lm.embed_tokens(input_ids)
            pos = torch.arange(s).unsqueeze(0) + p
            pe = self.lm.rotary_emb(x, pos)
            q = torch.arange(s).unsqueeze(1) + p
            k = torch.cat([torch.arange(p).unsqueeze(0), torch.arange(s).unsqueeze(0) + p], dim=1)
            mask = torch.where(k <= q, torch.zeros((), dtype=x.dtype), torch.full((), torch.finfo(x.dtype).min, dtype=x.dtype))[None, None]
            for layer in self.lm.layers:
                x = layer(x, attention_mask=mask, position_ids=pos, past_key_value=cache, use_cache=True, cache_position=pos[0],
                          position_embeddings=pe)[0]
            outs = [x + self.type1]
            for i in range(n):
                outs += [cache.key_cache[i], cache.value_cache[i]]
            return tuple(outs)

    w = TextLM().eval()
    ids = torch.randint(0, 1000, (1, 5))
    past = [torch.randn(1, 2, 7, 64) for _ in range(2 * n)]
    names_in = ['input_ids'] + [f'past_key_values.{i}.{kv}' for i in range(n) for kv in ('key', 'value')]
    names_out = ['hidden'] + [f'present.{i}.{kv}' for i in range(n) for kv in ('key', 'value')]
    dyn = {'input_ids': {1: 'seq'}, 'hidden': {1: 'seq'}}
    for nm in names_in[1:]:
        dyn[nm] = {2: 'past'}
    for nm in names_out[1:]:
        dyn[nm] = {2: 'total'}
    # Check against the model's own forward (prefill of 12 tokens = 7 + 5 with the cache).
    with torch.no_grad():
        full_ids = torch.randint(0, 1000, (1, 12))
        ref = m.forward_lm(input_ids=full_ids, use_cache=True, return_dict=True)
        legacy = ref.past_key_values
        pk = [legacy.key_cache[i][:, :, :7] for i in range(n)]
        pv = [legacy.value_cache[i][:, :, :7] for i in range(n)]
        mine = w(full_ids[:, 7:], *[t for i in range(n) for t in (pk[i], pv[i])])[0]
        print('text_lm vs forward_lm (cached, 5 of 12 tokens): max abs diff',
              float((mine - (ref.last_hidden_state[:, 7:] + type1)).abs().max()))
    export(w, (ids, *past), out, names_in, names_out, dyn)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    feeds = {'input_ids': full_ids[:, 7:].numpy()}
    for i in range(n):
        feeds[f'past_key_values.{i}.key'] = pk[i].numpy()
        feeds[f'past_key_values.{i}.value'] = pv[i].numpy()
    o = s.run(None, feeds)[0]
    print('onnx vs torch: max abs diff', np.abs(o - mine.numpy()).max())


def diffusion(src: Path, out: Path, steps: int = 5):
    """condition, neg_condition [1, 896], noise [1, 64], cfg_scale [1] -> latent [1, 64]."""
    m = load(src)
    head = m.model.prediction_head
    sched = copy.deepcopy(m.model.noise_scheduler)

    class Sampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, condition, neg_condition, noise, cfg_scale):
            s = copy.deepcopy(sched)
            s.set_timesteps(steps)
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
    cond, neg, noise = torch.randn(1, 896, generator=g), torch.randn(1, 896, generator=g), torch.randn(1, 64, generator=g)
    cfg = torch.tensor([1.5])
    with torch.no_grad():
        # The original samples with a batch of 2 (noise row 1 unused): compare.
        m.set_ddpm_inference_steps(steps)
        torch.manual_seed(1)
        ref = m.sample_speech_tokens(cond, neg, cfg_scale=1.5)
        torch.manual_seed(1)
        nz = torch.randn(2, 64)[:1]
        mine = w(cond, neg, nz, cfg)
        print('sampler vs sample_speech_tokens: max abs diff', float((ref - mine).abs().max()))
    export(w, (cond, neg, noise, cfg), out, ['condition', 'neg_condition', 'noise', 'cfg_scale'], ['latent'])
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    o = s.run(None, {'condition': cond.numpy(), 'neg_condition': neg.numpy(), 'noise': nz.numpy(), 'cfg_scale': cfg.numpy()})[0]
    print('onnx vs torch: max abs diff', np.abs(o - mine.numpy()).max())


def connector(src: Path, out: Path):
    """latent [1, 1, 64] -> next TTS-LM input [1, 1, 896] (acoustic connector + speech type embedding)."""
    m = load(src)

    class C(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = m.model.acoustic_connector
            self.register_buffer('type0', m.model.tts_input_types.weight[0].detach())

        def forward(self, latent):
            return self.c(latent) + self.type0
    export(C().eval(), (torch.randn(1, 1, 64),), out, ['latent'], ['embeds'])


def eos(src: Path, out: Path):
    """hidden [1, 896] -> end-of-speech probability [1, 1] (stop above 0.5)."""
    m = load(src)

    class E(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = m.tts_eos_classifier

        def forward(self, hidden):
            return torch.sigmoid(self.e(hidden))
    export(E().eval(), (torch.randn(1, 896),), out, ['hidden'], ['probability'])


def decoder_step(src: Path, out: Path):
    """latent [1, 1, 64] (as sampled) + state_i -> audio [1, 1, 3200] + new_state_i: one streaming decode step."""
    from vibevoice.modular.modular_vibevoice_tokenizer import SConv1d, SConvTranspose1d
    m = load(src)
    dec = m.model.acoustic_tokenizer.decoder
    scale, bias = m.model.speech_scaling_factor.item(), m.model.speech_bias_factor.item()
    layers = [l for l in dec.modules() if isinstance(l, (SConv1d, SConvTranspose1d)) and (isinstance(l, SConvTranspose1d) or l.context_size > 0)]
    shapes = [(l.in_channels, l.context_size if isinstance(l, SConv1d) else l.kernel_size - 1) for l in layers]

    class Cache:
        def __init__(self, states):
            self.s, self.new = states, {}

        def get(self, layer_id, idx):
            return self.s.get(layer_id)

        def set(self, layer_id, idx, value):
            self.new[layer_id] = value

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dec = dec

        def forward(self, latent, *states):
            cache = Cache({l.layer_id: s for l, s in zip(layers, states)})
            x = (latent / scale - bias).permute(0, 2, 1)
            audio = self.dec(x, cache=cache, sample_indices=torch.tensor([0]), use_cache=True)
            return (audio, *[cache.new[l.layer_id] for l in layers])

    w = Step().eval()
    zeros = [torch.zeros(1, c, n) for c, n in shapes]
    # Check against the original streaming decode over 6 latents.
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
    lat = torch.randn(6, 1, 1, 64) * 0.5
    with torch.no_grad():
        cache = VibeVoiceTokenizerStreamingCache()
        ref = [m.model.acoustic_tokenizer.decode(l / scale - bias, cache=cache, sample_indices=torch.tensor([0]), use_cache=True) for l in lat]
        st, mine = zeros, []
        for l in lat:
            a, *st = w(l, *st)
            mine.append(a)
        print('step vs streaming decode: max abs diff', float(max((a - b).abs().max() for a, b in zip(ref, mine))),
              'peak', float(max(a.abs().max() for a in ref)))
    names = [f'state_{i}' for i in range(len(layers))]
    export(w, (lat[0], *zeros), out, ['latent'] + names, ['audio'] + [f'new_{n}' for n in names])
    (out.parent / (out.stem + '_states.json')).write_text(json.dumps({'states': [[1, c, n] for c, n in shapes], 'samples_per_latent': 3200,
                                                                       'sample_rate': 24000}))
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    st, err = [z.numpy() for z in zeros], 0.0
    for l, r in zip(lat, ref):
        a, *st = s.run(None, {'latent': l.numpy(), **dict(zip(names, st))})
        err = max(err, float(np.abs(a - r.numpy()).max()))
    print('onnx vs streaming decode: max abs diff', err)


def voices(src: Path, out: Path):
    """A voice preset (.pt from the VibeVoice repo) -> .npz: KV caches (float16) and the negative prompt's hidden state."""
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        p = torch.load(str(src), map_location='cpu', weights_only=True)
    arrays = {}
    for part in ('lm', 'tts_lm', 'neg_tts_lm'):
        c = p[part].past_key_values
        for i in range(len(c.key_cache)):
            arrays[f'{part}.{i}.key'] = c.key_cache[i].float().numpy().astype(np.float16)
            arrays[f'{part}.{i}.value'] = c.value_cache[i].float().numpy().astype(np.float16)
    arrays['neg_tts_lm.hidden'] = p['neg_tts_lm'].last_hidden_state[:, -1].float().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print({k: v.shape for k, v in arrays.items() if k.endswith('0.key') or k.endswith('hidden')})


def main():
    cmd, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {'hf': hf, 'text_lm': text_lm, 'diffusion': diffusion, 'connector': connector, 'eos': eos, 'decoder_step': decoder_step,
     'voices': voices}[cmd](src, out)


if __name__ == '__main__':
    main()
