"""Exports Fun-CosyVoice3 0.5B (FunAudioLLM/Fun-CosyVoice3-0.5B-2512, Apache-2.0) to ONNX graphs for ONNX Runtime
Web (WebGPU), from the upstream PyTorch code (github.com/FunAudioLLM/CosyVoice):

- embed.onnx: one Gather over the text LM's token table and the speech-token table, stacked (speech ids are offset
  by 151936), so the JS loop builds the language model's input embeddings with one call.
- llm/: the Qwen2 0.5B speech-token language model, built by ONNX Runtime GenAI's model builder from an HF folder
  this script writes (lm_head = CosyVoice's llm_decoder, 6761 speech logits; input embeddings from embed.onnx).
- flow_encoder.onnx: speech tokens + speaker embedding + prompt mel -> mu, spks, cond for the flow matcher.
- flow_step.onnx: one Euler step of the DiT flow matcher with classifier-free guidance (rate 0.7) folded in.
- hift.onnx: the causal HiFT vocoder, mel -> 24 kHz waveform: F0 predictor, harmonic source, and the STFT / iSTFT
  as convolutions (no complex numbers, no float64, no random ops: the source noise is an input).
- voice_features.onnx: a voice prompt's 16 kHz and 24 kHz waveforms -> the Whisper log-mel for the speech
  tokenizer, the Kaldi fbank for CAM++, and the 80-band mel for the flow prompt.
- campplus.onnx, speech_tokenizer_v3.onnx: the published voice-prompt graphs, moved to opset 17.

Float graphs go to tools/export/.cache/tts-small/cosyvoice3-onnx, the quantized ones (int4, some int8: see QUANT)
with the tokenizer, default voice and README to exports/tts-int4-small/cosyvoice3.

    C=tools/export/.cache/tts-small/cosy-venv/bin/python
    $C -I tools/export/export_cosyvoice3.py noise embed lmdir prompt flow hift voice   # float graphs, voice, noise
    tools/export/.venv/bin/python tools/export/export_cosyvoice3.py llm               # LM: float32 and int4 (builder)
    $C -I tools/export/export_cosyvoice3.py quant [NAME...]                           # the other quantized graphs
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / '.cache/tts-small'
SRC = CACHE / 'cosyvoice3'
CODE = CACHE / 'CosyVoice'
FLOAT = CACHE / 'cosyvoice3-onnx'
Q = ROOT / 'exports/tts-int4-small/cosyvoice3'
PROMPT_WAV = CODE / 'asset/zero_shot_prompt.wav'
PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
TEXT_VOCAB = 151936
SPEECH_VOCAB = 6761
OPSET = 17


def cosy():
    sys.path[:0] = [str(CODE), str(CODE / 'third_party/Matcha-TTS')]
    from cosyvoice.cli.cosyvoice import AutoModel
    return AutoModel(model_dir=str(SRC))


def export(module, args, path, inputs, outputs, dynamic):
    import onnx
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs, dynamic_axes=dynamic,
                          opset_version=OPSET, do_constant_folding=True)
    model = onnx.load(str(path))
    ops = sorted({n.op_type for n in model.graph.node})
    print(path.name, f'{path.stat().st_size / 1e6:.1f} MB', ops, flush=True)


def save_noise(m):
    """The fixed noise the PyTorch pipeline uses: the flow matcher's start noise (seed 0) and the vocoder's uniform
    source noise. The evaluation feeds the same to the graphs; the browser draws its own."""
    flow = m.model.flow.decoder.rand_noise[0, :, :3000].numpy()
    sine = m.model.hift.m_source.l_sin_gen.sine_waves[0, :60 * 24000].numpy()
    FLOAT.mkdir(parents=True, exist_ok=True)
    np.savez(FLOAT / 'ref_noise.npz', flow=flow, sine=sine)
    Q.mkdir(parents=True, exist_ok=True)
    flow.astype(np.float16).tofile(Q / 'flow_noise_f16.bin')


def export_embed(m):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    llm = m.model.llm
    table = np.concatenate([llm.llm.model.model.embed_tokens.weight.detach().numpy(),
                            llm.speech_embedding.weight.detach().numpy()]).astype(np.float32)
    assert table.shape == (TEXT_VOCAB + SPEECH_VOCAB, 896)
    graph = helper.make_graph(
        [helper.make_node('Gather', ['table', 'ids'], ['embeds'], axis=0)], 'cosyvoice3_embed',
        [helper.make_tensor_value_info('ids', TensorProto.INT64, ['batch', 'seq'])],
        [helper.make_tensor_value_info('embeds', TensorProto.FLOAT, ['batch', 'seq', 896])],
        [numpy_helper.from_array(table, 'table')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)])
    model.ir_version = 8
    FLOAT.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(FLOAT / 'embed.onnx'))
    print('embed.onnx', table.shape)


def lm_hf_dir(m):
    """An HF Qwen2 folder for the model builder: CosyVoice's transformer, its speech-token table as the (unused,
    excluded) input embedding and its llm_decoder as lm_head, so the graph's logits are the 6761 speech logits."""
    from safetensors.torch import save_file
    llm = m.model.llm
    out = CACHE / 'cosyvoice3-lm-hf'
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().float().contiguous() for k, v in llm.llm.model.model.state_dict().items() if k != 'embed_tokens.weight'}
    state = {'model.' + k: v for k, v in state.items()}
    state['model.embed_tokens.weight'] = llm.speech_embedding.weight.detach().float().contiguous()
    state['lm_head.weight'] = llm.llm_decoder.weight.detach().float().contiguous()
    save_file(state, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    config = json.loads((SRC / 'CosyVoice-BlankEN/config.json').read_text())
    config.update(vocab_size=SPEECH_VOCAB, tie_word_embeddings=False, torch_dtype='float32', bos_token_id=6561, eos_token_id=6562)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    for name in ('generation_config.json', 'merges.txt', 'tokenizer_config.json', 'vocab.json'):
        shutil.copy(SRC / 'CosyVoice-BlankEN' / name, out / name)
    gen = json.loads((out / 'generation_config.json').read_text())
    gen.update(bos_token_id=6561, eos_token_id=6562)
    (out / 'generation_config.json').write_text(json.dumps(gen, indent=2))
    print('wrote', out)


def build_llm():
    """Runs under tools/export/.venv (onnxruntime-genai's builder): float32 and int4 (k-quant, block 32), both with
    float32 inputs, logits and KV cache."""
    src = CACHE / 'cosyvoice3-lm-hf'
    common = ['exclude_embeds=true', 'use_webgpu_fp32=true']
    for precision, out, extra in (('fp32', FLOAT / 'llm', []), ('int4', Q / 'llm_q4', ['algo_config=k_quant', 'block_size=32'])):
        work = CACHE / 'build/cosyvoice3'
        cmd = [str(HERE / 'guarded.sh'), sys.executable, '-m', 'onnxruntime_genai.models.builder', '-i', str(src), '-o', str(out),
               '-p', precision, '-e', 'webgpu', '-c', str(work), '--extra_options', *common, *extra]
        print('+', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        shutil.rmtree(work, ignore_errors=True)
        for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json', 'added_tokens.json'):
            (out / name).unlink(missing_ok=True)


def remove_weight_norm(module):
    import torch
    for sub in module.modules():
        if hasattr(sub, 'parametrizations') and 'weight' in sub.parametrizations:
            torch.nn.utils.parametrize.remove_parametrizations(sub, 'weight')


def export_flow(m):
    import torch
    import torch.nn.functional as F
    flow = m.model.flow.float().eval().requires_grad_(False)
    remove_weight_norm(flow)

    class Encoder(torch.nn.Module):
        """CausalMaskedDiffWithDiT.inference up to the decoder, finalize=True (the whole utterance)."""

        def __init__(self):
            super().__init__()
            self.flow = flow  # registered, so its weights export as initializers

        def forward(self, tokens, embedding, prompt_feat):
            spks = flow.spk_embed_affine_layer(F.normalize(embedding, dim=1))
            h = flow.pre_lookahead_layer(flow.input_embedding(torch.clamp(tokens, min=0)))
            h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
            mu = h.transpose(1, 2)
            cond = F.pad(prompt_feat.transpose(1, 2), (0, mu.shape[2] - prompt_feat.shape[1]))
            return mu, spks, cond

    estimator = flow.decoder.estimator
    rope = estimator.rotary_embed

    def rope_from_seq_len(seq_len):
        # x_transformers' version uses an Einsum for this outer product.
        t = torch.arange(seq_len, device=rope.inv_freq.device).float()
        freqs = t[None, :, None] * rope.inv_freq[None, None, :]
        return torch.stack((freqs, freqs), dim=-1).reshape(1, seq_len, -1), 1.

    rope.forward_from_seq_len = rope_from_seq_len
    cfg = flow.decoder.inference_cfg_rate

    class Step(torch.nn.Module):
        """One step of CausalConditionalCFM.solve_euler: the estimator on [conditional, unconditional] and the
        guided Euler update x + dt * ((1 + cfg) * v_cond - cfg * v_uncond)."""

        def __init__(self):
            super().__init__()
            self.estimator = estimator

        def forward(self, x, mu, spks, cond, t, dt):
            zeros = torch.zeros_like(mu)
            x_in = torch.cat([x, x])
            mask = torch.ones_like(x_in[:, :1])
            v = estimator(x_in, mask, torch.cat([mu, zeros]), torch.cat([t, t]), torch.cat([spks, torch.zeros_like(spks)]),
                          torch.cat([cond, zeros]), streaming=False)
            return x + dt * ((1.0 + cfg) * v[:1] - cfg * v[1:])

    n_tok, n_prompt = 40, 30
    tokens = torch.randint(0, 6561, (1, n_tok))
    emb = torch.randn(1, 192)
    pf = torch.randn(1, n_prompt, 80)
    export(Encoder(), (tokens, emb, pf), FLOAT / 'flow_encoder.onnx', ['tokens', 'embedding', 'prompt_feat'], ['mu', 'spks', 'cond'],
           {'tokens': {1: 'tokens'}, 'prompt_feat': {1: 'prompt_frames'}, 'mu': {2: 'frames'}, 'cond': {2: 'frames'}})
    mu, spks, cond = Encoder()(tokens, emb, pf)
    x = torch.randn_like(mu)
    export(Step(), (x, mu, spks, cond, torch.tensor([0.1]), torch.tensor([0.05])), FLOAT / 'flow_step.onnx',
           ['x', 'mu', 'spks', 'cond', 't', 'dt'], ['x_next'],
           {'x': {2: 'frames'}, 'mu': {2: 'frames'}, 'cond': {2: 'frames'}, 'x_next': {2: 'frames'}})


def hift_module(hift):
    """CausalHiFTGenerator.inference(finalize=True) in ONNX-friendly form: float32 F0 predictor (the original runs it
    in float64), the harmonic source at frame rate then repeated (exactly what the original's nearest upsampling +
    linear downsampling by 480 computes), the source noise as an input, and the STFT / iSTFT (n_fft 16, hop 4, Hann)
    as a strided Conv and ConvTranspose with DFT kernels."""
    import torch
    import torch.nn.functional as F
    from scipy.signal import get_window
    hift = hift.float().eval().requires_grad_(False)
    remove_weight_norm(hift)
    n_fft, hop = hift.istft_params['n_fft'], hift.istft_params['hop_len']
    up = int(np.prod(hift.upsample_rates) * hop)  # 480 samples per mel frame
    sine_gen = hift.m_source.l_sin_gen
    window = get_window('hann', n_fft, fftbins=True).astype(np.float64)
    n = np.arange(n_fft)
    k = np.arange(n_fft // 2 + 1)
    ang = 2 * np.pi * np.outer(k, n) / n_fft
    stft_kernel = np.concatenate([np.cos(ang) * window, -np.sin(ang) * window])[:, None, :]  # [18, 1, 16]
    coef = np.where((k == 0) | (k == n_fft // 2), 1.0, 2.0)[:, None] / n_fft
    istft_kernel = np.concatenate([coef * np.cos(ang) * window, -coef * np.sin(ang) * window])[:, None, :]  # [18, 1, 16]

    class HiFT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hift = hift
            self.register_buffer('stft_kernel', torch.tensor(stft_kernel, dtype=torch.float32))
            self.register_buffer('istft_kernel', torch.tensor(istft_kernel, dtype=torch.float32))
            self.register_buffer('window_sq', torch.tensor(window ** 2, dtype=torch.float32)[None, None, :])
            self.register_buffer('harmonics', torch.arange(1, sine_gen.harmonic_num + 2, dtype=torch.float32))

        def forward(self, mel, noise):
            # The F0 head (512 -> 1) and the harmonic mix (9 -> 1) as Mul + ReduceSum, not MatMul, so that int4
            # quantization leaves these two tiny, pitch-critical layers in float32.
            h = mel
            for layer in hift.f0_predictor.condnet:
                h = layer(h)
            head = hift.f0_predictor.classifier
            f0 = torch.abs((h.transpose(1, 2) * head.weight[0]).sum(-1) + head.bias[0])  # [1, F]
            frames = f0.shape[1]
            fn = f0[:, :, None] * self.harmonics  # [1, F, 9]
            rad = torch.remainder(fn / sine_gen.sampling_rate, 1.0)
            # The original's phase, cumsum(rad) * 2 pi * 480, grows past 1e5 radians; GPU sin() loses accuracy on
            # arguments that large, so the whole cycles are dropped first (sin(2 pi (x mod 1)) = sin(2 pi x)).
            phase = torch.remainder(torch.cumsum(rad, dim=1) * up, 1.0) * (2 * np.pi)
            phase = phase[:, :, None, :].expand(-1, -1, up, -1).reshape(1, frames * up, -1)
            uv = (f0 > sine_gen.voiced_threshold).float()
            uv = uv[:, :, None].expand(-1, -1, up).reshape(1, frames * up, 1)
            noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
            sines = torch.sin(phase) * sine_gen.sine_amp * uv + noise_amp * noise
            mix = hift.m_source.l_linear
            s = torch.tanh((sines * mix.weight[0]).sum(-1, keepdim=True) + mix.bias[0]).transpose(1, 2)  # [1, 1, L]
            spec = F.conv1d(F.pad(s, (n_fft // 2, n_fft // 2), mode='reflect'), self.stft_kernel, stride=hop)
            x = hift.conv_pre(mel)
            for i in range(hift.num_upsamples):
                x = F.leaky_relu(x, hift.lrelu_slope)
                x = hift.ups[i](x)
                if i == hift.num_upsamples - 1:
                    x = hift.reflection_pad(x)
                si = hift.source_resblocks[i](hift.source_downs[i](spec))
                x = x + si
                xs = None
                for j in range(hift.num_kernels):
                    r = hift.resblocks[i * hift.num_kernels + j](x)
                    xs = r if xs is None else xs + r
                x = xs / hift.num_kernels
            x = hift.conv_post(F.leaky_relu(x))
            bins = n_fft // 2 + 1
            magnitude = torch.clamp(torch.exp(x[:, :bins]), max=1e2)
            phase = torch.sin(x[:, bins:])
            ri = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
            y = F.conv_transpose1d(ri, self.istft_kernel, stride=hop)
            env = F.conv_transpose1d(torch.ones_like(ri[:, :1]), self.window_sq, stride=hop)
            y = y / env
            length = hop * (ri.shape[2] - 1)
            y = y[:, 0, n_fft // 2:n_fft // 2 + length]
            return torch.clamp(y, -hift.audio_limit, hift.audio_limit)

    return HiFT()


def export_hift(m):
    import torch
    model = hift_module(m.model.hift)
    mel = torch.randn(1, 80, 50) - 5
    noise = torch.rand(1, 50 * 480, 9)
    export(model, (mel, noise), FLOAT / 'hift.onnx', ['mel', 'noise'], ['audio'],
           {'mel': {2: 'frames'}, 'noise': {1: 'samples'}, 'audio': {1: 'samples'}})


def voice_module():
    """A voice prompt's features, as the upstream frontend computes them, with each STFT as one strided Conv whose
    kernel folds in the frame processing:
    - whisper_mel [1, 128, T]: whisper.log_mel_spectrogram(wav_16k, n_mels=128) (n_fft 400, hop 160, centred,
      reflect padding, Slaney mel, log10, max - 8 floor, (x + 4) / 4), for speech_tokenizer_v3.
    - fbank [1, T, 80]: torchaudio's kaldi.fbank(wav_16k, num_mel_bins=80, dither=0) (25 ms Povey window, 10 ms
      hop, DC removal, pre-emphasis 0.97, 512-point power spectrum, log) minus its mean over time, for CAM++.
    - prompt_feat [1, T, 80]: Matcha's mel_spectrogram(wav_24k) (n_fft 1920, hop 480, reflect padding 720, magnitude,
      Slaney mel, natural log floored at 1e-5), the flow's prompt mel."""
    import librosa
    import torch
    import torch.nn.functional as F
    import torchaudio
    import whisper

    def dft(n_fft, size, frame_matrix=None):
        n = np.arange(size)
        k = np.arange(n_fft // 2 + 1)
        ang = 2 * np.pi * np.outer(k, n) / n_fft
        kern = np.concatenate([np.cos(ang), -np.sin(ang)])
        if frame_matrix is not None:
            kern = kern @ frame_matrix
        return torch.tensor(kern[:, None, :], dtype=torch.float32)

    # Whisper
    w_win = torch.hann_window(400).double().numpy()
    w_kernel = dft(400, 400, np.diag(w_win))
    w_mel = torch.tensor(np.asarray(whisper.audio.mel_filters('cpu', 128)), dtype=torch.float32)
    # Kaldi: frame -> remove DC -> pre-emphasis (first sample against itself) -> Povey window -> 512-point DFT
    L = 400
    dc = np.eye(L) - np.ones((L, L)) / L
    pre = np.eye(L)
    pre[np.arange(1, L), np.arange(L - 1)] = -0.97
    pre[0, 0] = 1 - 0.97
    povey = torch.hann_window(L, periodic=False, dtype=torch.float64).pow(0.85).numpy()
    k_kernel = dft(512, L, np.diag(povey) @ pre @ dc)
    k_mel, _ = torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    k_mel = F.pad(k_mel, (0, 1)).float()  # [80, 257]
    # Matcha
    m_win = torch.hann_window(1920).double().numpy()
    m_kernel = dft(1920, 1920, np.diag(m_win))
    m_mel = torch.tensor(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=None), dtype=torch.float32)
    eps = torch.finfo(torch.float32).eps

    def power(frames, bins):
        return frames[:, :bins] ** 2 + frames[:, bins:] ** 2

    class Voice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, t in (('w_kernel', w_kernel), ('w_mel', w_mel), ('k_kernel', k_kernel), ('k_mel', k_mel),
                            ('m_kernel', m_kernel), ('m_mel', m_mel)):
                self.register_buffer(name, t)

        def forward(self, wav_16k, wav_24k):
            x = F.pad(wav_16k[:, None], (200, 200), mode='reflect')
            spec = power(F.conv1d(x, self.w_kernel, stride=160), 201)[:, :, :-1]
            log_spec = torch.clamp(torch.matmul(self.w_mel, spec), min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            whisper_mel = (log_spec + 4.0) / 4.0
            spec = power(F.conv1d(wav_16k[:, None], self.k_kernel, stride=160), 257)
            fbank = torch.log(torch.clamp(torch.matmul(self.k_mel, spec), min=eps)).transpose(1, 2)
            fbank = fbank - fbank.mean(dim=1, keepdim=True)
            x = F.pad(wav_24k[:, None], (720, 720), mode='reflect')
            mag = torch.sqrt(power(F.conv1d(x, self.m_kernel, stride=480), 961) + 1e-9)
            prompt_feat = torch.log(torch.clamp(torch.matmul(self.m_mel, mag), min=1e-5)).transpose(1, 2)
            return whisper_mel, fbank, prompt_feat

    return Voice()


def export_voice():
    import torch
    export(voice_module(), (torch.randn(1, 16000) * 0.1, torch.randn(1, 24000) * 0.1), FLOAT / 'voice_features.onnx',
           ['wav_16k', 'wav_24k'], ['whisper_mel', 'fbank', 'prompt_feat'],
           {'wav_16k': {1: 'samples_16k'}, 'wav_24k': {1: 'samples_24k'}, 'whisper_mel': {2: 'whisper_frames'},
            'fbank': {1: 'fbank_frames'}, 'prompt_feat': {1: 'mel_frames'}})
    import onnx
    from onnx import version_converter
    for name in ('campplus.onnx', 'speech_tokenizer_v3.onnx'):
        model = version_converter.convert_version(onnx.load(str(SRC / name)), OPSET)
        onnx.save(model, str(FLOAT / name))
        print(name, 'opset', OPSET)


def export_prompt(m):
    """The default voice (the upstream zero-shot prompt clip) as JSON: the LM prompt's text ids, its speech tokens,
    the speaker embedding and the prompt mel, and the tokenizer with CosyVoice 3's special tokens."""
    fe = m.frontend
    inputs = fe.frontend_zero_shot('x', fe.text_normalize(PROMPT_TEXT, split=False), str(PROMPT_WAV), 24000, '')
    voice = {
        'prompt_text': PROMPT_TEXT,
        'prompt_text_ids': inputs['prompt_text'][0].tolist(),
        'prompt_speech_tokens': inputs['llm_prompt_speech_token'][0].tolist(),
        'embedding': [round(float(v), 6) for v in inputs['flow_embedding'][0]],
        'prompt_feat_shape': list(inputs['prompt_speech_feat'][0].shape),
        'prompt_feat': [round(float(v), 5) for v in inputs['prompt_speech_feat'][0].reshape(-1)],
    }
    Q.mkdir(parents=True, exist_ok=True)
    (Q / 'voice_default.json').write_text(json.dumps(voice, separators=(',', ':')))
    tok = fe.tokenizer.tokenizer
    tok.save_pretrained(str(Q / 'tokenizer'))
    for p in (Q / 'tokenizer').iterdir():
        if p.name not in ('tokenizer.json', 'tokenizer_config.json'):
            p.unlink()
    print('voice', len(voice['prompt_text_ids']), 'text ids', len(voice['prompt_speech_tokens']), 'speech tokens',
          voice['prompt_feat_shape'])


def quantize(src, dst, eight=(), keep=(), conv_bits=4, matmul_bits=4, block=32):
    """quantize_int4.py's recipe, with per-node exceptions: MatMul / Gather nodes whose name matches a pattern in
    `eight` become 8-bit MatMulNBits / GatherBlockQuantized, those matching `keep` stay float32."""
    import re

    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    sys.path.insert(0, str(HERE))
    import quantize_int4 as qi
    model = qi.upgrade(onnx.load(str(src)))
    qi.float64_to_float32(model)
    qi.gemm_to_matmul(model)
    names = [n.name for n in model.graph.node if n.op_type in ('MatMul', 'Gather')]
    match = lambda patterns: [n for n in names if any(re.search(p, n) for p in patterns)]
    kw = dict(block_size=block, is_symmetric=True, op_types_to_quantize=('MatMul', 'Gather'), quant_axes=(('MatMul', 0), ('Gather', 1)))
    if match(eight):
        # (nodes_to_include only adds nodes to the op types quantized, so the others are excluded instead)
        q = MatMulNBitsQuantizer(model, bits=8, nodes_to_exclude=sorted(set(names) - set(match(eight))), **kw)
        q.process()
        model = q.model.model
    q = MatMulNBitsQuantizer(model, bits=matmul_bits, nodes_to_exclude=match(keep) + match(eight), **kw)
    q.process()
    model = q.model.model
    convs = qi.dequantize_weights(model, block, conv_bits)
    counts = {op: sum(1 for n in model.graph.node if n.op_type == op) for op in ('MatMul', 'MatMulNBits', 'GatherBlockQuantized')}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    print(f'{Path(src).name} -> {Path(dst).name}: {counts}, {convs} int{conv_bits} conv weights, '
          f'{len(match(eight))} 8-bit, {len(match(keep))} float; {Path(dst).stat().st_size / 1e6:.1f} MB', flush=True)


QUANT = {  # output (in exports/tts-int4-small/cosyvoice3) -> (float graph, options)
    'embed_q4': ('embed', {}),
    'flow_encoder_q4': ('flow_encoder', {}),
    'flow_step_q4': ('flow_step', {}),
    'flow_step_q8': ('flow_step', {'matmul_bits': 8}),
    'hift_q4': ('hift', {}),
    'hift_q4c8': ('hift', {'conv_bits': 8}),
    'speech_tokenizer_v3_q4': ('speech_tokenizer_v3', {}),
    'speech_tokenizer_v3_q8': ('speech_tokenizer_v3', {'matmul_bits': 8, 'conv_bits': 8}),
    'campplus_q8': ('campplus', {'conv_bits': 8}),  # at 4 bits its embedding drifts (cosine 0.80)
}


def quantize_all(names):
    for out in names or QUANT:
        src, options = QUANT[out]
        quantize(FLOAT / f'{src}.onnx', Q / f'{out}.onnx', **options)


if __name__ == '__main__':
    steps = sys.argv[1:]
    if steps[:1] == ['quant']:
        quantize_all(steps[1:])
        sys.exit()
    if steps == ['llm']:
        build_llm()
        sys.exit()
    model = cosy() if set(steps) - {'voice'} else None
    for step in steps:
        {'noise': save_noise, 'embed': export_embed, 'lmdir': lm_hf_dir, 'flow': export_flow, 'hift': export_hift,
         'voice': lambda _: export_voice(), 'prompt': export_prompt}[step](model)
