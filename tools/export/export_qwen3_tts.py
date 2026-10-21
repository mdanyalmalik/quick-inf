"""Qwen3-TTS (12 Hz: 0.6B / 1.7B, CustomVoice / VoiceDesign / Base) split into graphs for ONNX Runtime Web.

The talker (the 28-layer language model that predicts the first codebook) and the code predictor (the 5-layer model
that fills in the other 15 codebooks of each frame) are plain Qwen3 decoders: their multimodal RoPE always gets the
same position in all three sections for speech, which is ordinary RoPE. `hf` writes each as a Qwen3ForCausalLM
checkpoint for ONNX Runtime GenAI's model builder (run with exclude_embeds, since their inputs are sums of embeddings):

- talker: lm_head = codec_head (3072 codes), embed_tokens = the codec embedding (unused by the graph).
- code predictor: lm_head = its 15 per-codebook heads stacked (15 x 2048 rows; step j reads rows j*2048..j*2048+2047).

`embeds` writes the lookup graphs (float; quantize_int4.py makes them int4):

- text_embed.onnx: ids -> text_projection(text_embedding(ids)), precomputed per token into one table.
- codec_embed.onnx: ids into one stacked table (rows 0..3071 the talker's codec embedding, then 15 x 2048 rows, the
  code predictor's embeddings for codebooks 1..15) -> `embeds` in the talker's width; for 1.7B also `cp_embeds`, the
  same rows through small_to_mtp_projection (the code predictor's input width).
- cp_project.onnx (1.7B only): small_to_mtp_projection for the talker's hidden state.

`decoder` exports the speech tokenizer's decoder (16 codebooks -> 24 kHz audio) with a dynamic number of frames, and
`encoder` the tokenizer's encoder (24 kHz audio -> 16 codebooks, for Base's voice cloning); `speaker` exports Base's
speaker encoder (128-bin log-mel -> x-vector).

    qwen-venv/bin/python tools/export/export_qwen3_tts.py hf|embeds|decoder|encoder|speaker SRC OUT
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = ('vocab.json', 'merges.txt', 'tokenizer_config.json')


def weights(src: Path) -> dict:
    out = {}
    with safe_open(str(src / 'model.safetensors'), 'pt') as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def qwen3_config(c: dict, vocab: int) -> dict:
    return {
        'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3', 'hidden_size': c['hidden_size'],
        'intermediate_size': c['intermediate_size'], 'num_hidden_layers': c['num_hidden_layers'],
        'num_attention_heads': c['num_attention_heads'], 'num_key_value_heads': c['num_key_value_heads'],
        'head_dim': c['head_dim'], 'hidden_act': c['hidden_act'], 'rms_norm_eps': c['rms_norm_eps'],
        'rope_theta': c['rope_theta'], 'max_position_embeddings': c['max_position_embeddings'], 'vocab_size': vocab,
        'tie_word_embeddings': False, 'attention_bias': False, 'torch_dtype': 'bfloat16', 'bos_token_id': 0,
        'eos_token_id': 0, 'use_sliding_window': False, 'sliding_window': None,
    }


def hf(src: Path, out: Path):
    w = weights(src)
    tc = json.loads((src / 'config.json').read_text())['talker_config']
    cc = tc['code_predictor_config']
    parts = {
        'talker': ('talker.model.', w['talker.codec_head.weight'], w['talker.model.codec_embedding.weight'], tc),
        'code_predictor': ('talker.code_predictor.model.',
                           torch.cat([w[f'talker.code_predictor.lm_head.{i}.weight'] for i in range(cc['num_code_groups'] - 1)]),
                           None, cc),
    }
    for name, (prefix, head, embed, c) in parts.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        t = {'model.' + k[len(prefix):]: v.contiguous() for k, v in w.items()
             if k.startswith(prefix) and ('.layers.' in k or k.endswith('norm.weight') and '.layers.' not in k)}
        t['lm_head.weight'] = head.contiguous()
        t['model.embed_tokens.weight'] = (embed if embed is not None else head).contiguous().clone()
        save_file(t, str(d / 'model.safetensors'), metadata={'format': 'pt'})
        (d / 'config.json').write_text(json.dumps(qwen3_config(c, head.shape[0]), indent=1))
        for f in TOKENIZER_FILES:
            shutil.copy(src / f, d / f)
        print(name, len(t), 'tensors, vocab', head.shape[0])


def gather_graph(path: Path, tables: dict, out_dim_names: dict):
    """ids [batch, seq] -> one Gather per table (output name -> float32 table)."""
    nodes, inits, outputs = [], [], []
    for name, table in tables.items():
        inits.append(numpy_helper.from_array(table.astype(np.float32), f'{name}_table'))
        nodes.append(helper.make_node('Gather', [f'{name}_table', 'ids'], [name], axis=0, name=f'{name}/Gather'))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', 'seq', table.shape[1]]))
    g = helper.make_graph(nodes, path.stem, [helper.make_tensor_value_info('ids', TensorProto.INT64, ['batch', 'seq'])], outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 21)])
    m.ir_version = 10
    onnx.save(m, str(path), save_as_external_data=m.ByteSize() > 2_000_000_000)


def embeds(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    w = {k: v.float() for k, v in weights(src).items() if 'layers.' not in k}
    tc = json.loads((src / 'config.json').read_text())['talker_config']
    n = tc['code_predictor_config']['num_code_groups'] - 1
    # text_projection(text_embedding(id)) for every id, in chunks.
    e = w['talker.model.text_embedding.weight']
    fc1, b1 = w['talker.text_projection.linear_fc1.weight'], w['talker.text_projection.linear_fc1.bias']
    fc2, b2 = w['talker.text_projection.linear_fc2.weight'], w['talker.text_projection.linear_fc2.bias']
    rows = []
    with torch.no_grad():
        for i in range(0, e.shape[0], 8192):
            rows.append(torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(e[i:i + 8192], fc1, b1)), fc2, b2))
    text = torch.cat(rows).numpy()
    gather_graph(out / 'text_embed.onnx', {'embeds': text}, {})
    stacked = torch.cat([w['talker.model.codec_embedding.weight']] + [w[f'talker.code_predictor.model.codec_embedding.{i}.weight'] for i in range(n)])
    tables = {'embeds': stacked.numpy()}
    proj = 'talker.code_predictor.small_to_mtp_projection.weight'
    if proj in w:
        pw, pb = w[proj], w['talker.code_predictor.small_to_mtp_projection.bias']
        tables['cp_embeds'] = torch.nn.functional.linear(stacked, pw, pb).numpy()
        x = helper.make_tensor_value_info('hidden', TensorProto.FLOAT, ['batch', 'seq', pw.shape[1]])
        y = helper.make_tensor_value_info('cp_hidden', TensorProto.FLOAT, ['batch', 'seq', pw.shape[0]])
        g = helper.make_graph([helper.make_node('MatMul', ['hidden', 'w'], ['mm']), helper.make_node('Add', ['mm', 'b'], ['cp_hidden'])],
                              'cp_project', [x], [y], [numpy_helper.from_array(pw.T.contiguous().numpy(), 'w'), numpy_helper.from_array(pb.numpy(), 'b')])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 21)])
        m.ir_version = 10
        onnx.save(m, str(out / 'cp_project.onnx'))
    gather_graph(out / 'codec_embed.onnx', tables, {})
    print('text', text.shape, 'codec', {k: v.shape for k, v in tables.items()})


def tokenizer_model(src: Path):
    from qwen_tts.core import Qwen3TTSTokenizerV2Model
    m = Qwen3TTSTokenizerV2Model.from_pretrained(str(src), attn_implementation='eager', dtype=torch.float32)
    return m.eval()


def sliding_causal_mask(t: int, window: int, dtype) -> torch.Tensor:
    i = torch.arange(t).unsqueeze(1)
    j = torch.arange(t).unsqueeze(0)
    ok = (j <= i) & (i - j < window)
    return torch.where(ok, torch.zeros((), dtype=dtype), torch.full((), torch.finfo(dtype).min, dtype=dtype))[None, None]


def decoder(src: Path, out: Path):
    """codes [batch, 16, frames] int64 -> audio [batch, 1, frames * 1920] (24 kHz)."""
    model = tokenizer_model(src)
    dec = model.decoder
    tr = dec.pre_transformer

    def transformer_forward(inputs_embeds=None, **_):
        x = tr.input_proj(inputs_embeds)
        t = x.shape[1]
        pos = torch.arange(t).unsqueeze(0)
        pe = tr.rotary_emb(x, pos)
        mask = sliding_causal_mask(t, tr.window_size, x.dtype)
        for layer in tr.layers:
            x = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        x = tr.output_proj(tr.norm(x))

        class R:
            last_hidden_state = x
        return R

    tr.forward = transformer_forward
    # Every causal conv here has stride 1, where the extra right padding is always 0; the original computes it
    # with math.ceil on the length, which the tracer would bake in as a constant.
    from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as mt
    for mod in dec.modules():
        if isinstance(mod, mt.Qwen3TTSTokenizerV2CausalConvNet):
            assert mod.stride == 1
            mod._get_extra_padding_for_conv1d = lambda h: 0

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dec = dec

        def forward(self, codes):
            return self.dec(codes)

    codes = torch.randint(0, 2048, (1, 16, 40))
    with torch.no_grad():
        ref = model.decoder.chunked_decode(codes)
        got = Wrap()(codes)
        print('wrapper vs chunked_decode max abs diff', float((ref - got).abs().max()))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Wrap(), (codes,), str(out), input_names=['codes'], output_names=['audio'], opset_version=17,
                      dynamic_axes={'codes': {0: 'batch', 2: 'frames'}, 'audio': {0: 'batch', 2: 'samples'}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    for t in (7, 60):
        c = torch.randint(0, 2048, (1, 16, t))
        with torch.no_grad():
            r = model.decoder.chunked_decode(c).numpy()
        o = s.run(None, {'codes': c.numpy()})[0]
        print(f'onnx vs torch, {t} frames: max abs diff {np.abs(o - r).max():.2e}, shape {o.shape}')


def windowed_transformer(tr, window: int):
    """Replaces a Mimi transformer's forward (create_causal_mask does not trace) with an explicit sliding causal mask."""
    def forward(hidden_states=None, **_):
        x = hidden_states
        t = x.shape[1]
        pos = torch.arange(t).unsqueeze(0)
        mask = sliding_causal_mask(t, window, x.dtype)
        for layer in tr.layers:
            x = layer(x, attention_mask=mask, position_ids=pos)[0]
        return (x,)
    tr.forward = forward


def encoder(src: Path, out: Path):
    """audio [batch, samples] at 24 kHz, zero-padded by the caller to a multiple of 1920 samples ->
    codes [batch, 16, samples / 1920]. With that padding every strided convolution's extra right padding is 0, which
    is what is traced; the caller trims the codes to ceil(true samples / 1920) frames, as the original does."""
    model = tokenizer_model(src)
    enc = model.encoder
    from transformers.models.mimi import modeling_mimi as mm
    for mod in enc.modules():
        if isinstance(mod, mm.MimiConv1d):
            mod._get_extra_padding_for_conv1d = lambda h: 0
        if isinstance(mod, mm.MimiEuclideanCodebook):
            def quantize(x, cb=mod):
                e = cb.embed
                d = (x * x).sum(-1, keepdim=True) - 2 * x @ e.T + (e * e).sum(-1)[None]
                return d.argmin(dim=-1)
            mod.quantize = quantize
    windowed_transformer(enc.encoder_transformer, enc.config.sliding_window)
    n_q = model.encoder_valid_num_quantizers

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, audio):
            codes, _, _ = self.enc._encode_frame(audio[:, None], n_q, None, return_dict=False)
            return codes

    ref_model = tokenizer_model(src)
    wav = torch.from_numpy(np.random.default_rng(0).standard_normal(1920 * 30).astype(np.float32) * 0.1)
    with torch.no_grad():
        ref = ref_model.encoder.encode(wav[None, None], num_quantizers=32, return_dict=True).audio_codes[:, :n_q]
        got = Wrap()(wav[None])
    print('wrapper vs encode: codes equal', float((ref == got).float().mean()))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Wrap(), (wav[None],), str(out), input_names=['audio'], output_names=['codes'], opset_version=17,
                      dynamic_axes={'audio': {0: 'batch', 1: 'samples'}, 'codes': {0: 'batch', 2: 'frames'}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    for n in (13, 52):
        w = torch.from_numpy(np.random.default_rng(n).standard_normal(1920 * n).astype(np.float32) * 0.1)
        with torch.no_grad():
            r = ref_model.encoder.encode(w[None, None], num_quantizers=32, return_dict=True).audio_codes[:, :n_q].numpy()
        o = s.run(None, {'audio': w[None].numpy()})[0]
        print(f'onnx vs torch, {n} frames: codes equal {np.mean(o == r):.4f} (first codebook {np.mean(o[:, 0] == r[:, 0]):.4f}), shape {o.shape}')


def speaker(src: Path, out: Path):
    """audio [batch, samples] at 24 kHz -> x-vector [batch, 1024]: the 128-bin log-mel front end (n_fft 1024, hop 256,
    Hann window, reflect padding, as mel_spectrogram) is built from a Conv1d DFT so the whole graph is ordinary ops."""
    from librosa.filters import mel as librosa_mel
    from qwen_tts.core.models import modeling_qwen3_tts as mq
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    cfg = Qwen3TTSConfig.from_pretrained(str(src))
    se = mq.Qwen3TTSSpeakerEncoder(cfg.speaker_encoder_config)
    sd = {k[len('speaker_encoder.'):]: v.float() for k, v in weights(src).items() if k.startswith('speaker_encoder.')}
    missing, unexpected = se.load_state_dict(sd, strict=True), None
    se.eval()
    n_fft, hop = 1024, 256
    window = torch.hann_window(n_fft)
    k = torch.arange(n_fft // 2 + 1)[:, None].float()
    n = torch.arange(n_fft)[None].float()
    ang = 2 * np.pi * k * n / n_fft
    basis = torch.cat([torch.cos(ang) * window, -torch.sin(ang) * window])[:, None]  # [2 * 513, 1, 1024]
    mel = torch.from_numpy(librosa_mel(sr=24000, n_fft=n_fft, n_mels=128, fmin=0, fmax=12000)).float()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.se = se
            self.register_buffer('basis', basis)
            self.register_buffer('mel', mel)

        def forward(self, audio):
            x = torch.nn.functional.pad(audio[:, None], ((n_fft - hop) // 2, (n_fft - hop) // 2), mode='reflect')
            spec = torch.nn.functional.conv1d(x, self.basis, stride=hop)
            re, im = spec[:, :n_fft // 2 + 1], spec[:, n_fft // 2 + 1:]
            mag = torch.sqrt(re * re + im * im + 1e-9)
            m = torch.log(torch.clamp(torch.matmul(self.mel, mag), min=1e-5))
            return self.se(m.transpose(1, 2))

    wav = torch.from_numpy(np.random.default_rng(0).standard_normal(24000 * 3).astype(np.float32) * 0.1)
    with torch.no_grad():
        ref_mel = mq.mel_spectrogram(wav[None], n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        ref = se(ref_mel)
        got = Wrap()(wav[None])
    print('wrapper vs original x-vector: max abs diff', float((ref - got).abs().max()), 'norm', float(ref.norm()))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Wrap(), (wav[None],), str(out), input_names=['audio'], output_names=['xvector'], opset_version=17,
                      dynamic_axes={'audio': {0: 'batch', 1: 'samples'}, 'xvector': {0: 'batch'}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    w = torch.from_numpy(np.random.default_rng(5).standard_normal(24000 * 5).astype(np.float32) * 0.1)
    with torch.no_grad():
        r = se(mq.mel_spectrogram(w[None], n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)).numpy()
    o = s.run(None, {'audio': w[None].numpy()})[0]
    print('onnx vs torch (5 s): max abs diff', np.abs(o - r).max(), 'cos', float((o * r).sum() / np.linalg.norm(o) / np.linalg.norm(r)))


def fixhead(src: Path, out: Path = None):
    """The builder (with include_hidden_states) wires the LM head to a stale tensor (a layer-0 norm output) instead of
    the final norm's output, now named hidden_states. Rewires it in SRC/model.onnx (graph only)."""
    m = onnx.load(str(src / 'model.onnx'), load_external_data=False)
    n = 0
    for node in m.graph.node:
        if node.name.startswith('/lm_head/') and node.input[0] != 'hidden_states':
            node.input[0] = 'hidden_states'
            n += 1
    onnx.save(m, str(src / 'model.onnx'))
    print(src.name, 'lm_head inputs rewired:', n)


def main():
    cmd, src = sys.argv[1], Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    {'hf': hf, 'embeds': embeds, 'decoder': decoder, 'encoder': encoder, 'speaker': speaker, 'fixhead': fixhead}[cmd](src, out)


if __name__ == '__main__':
    main()
