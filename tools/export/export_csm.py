"""Sesame CSM-1B split into graphs for ONNX Runtime Web.

CSM is two Llama decoders over Mimi audio frames (32 codebooks, 12.5 per second): the backbone (16 layers, 2048 wide)
reads the text and the previous frames (each frame's input is the sum of its 32 codebook embeddings) and predicts
codebook 0; the depth decoder (4 layers, 1024 wide) then predicts codebooks 1..31 of that frame from the backbone's
last hidden state. Both are plain Llama 3 decoders, so `hf` writes them as LlamaForCausalLM checkpoints for ONNX
Runtime GenAI's model builder (exclude_embeds, include_hidden_states):

- backbone: lm_head = codebook-0 head (2051 rows).
- depth decoder: lm_head = its 31 position heads stacked (31 x 2051 rows: position j reads rows (j-1)*2051..).

`embeds` writes the lookup graphs (float; quantize_int4.py makes them int4):

- text_embed.onnx: text ids -> backbone input (embed_text_tokens).
- audio_embed.onnx: ids into the 32 x 2051 audio table (row = codebook * 2051 + code) -> `embeds` (the backbone's
  width; a frame's backbone input is the sum of its 32 rows) and `depth_embeds` (the same table through the depth
  decoder's input projection; the depth decoder's audio embedding is tied to the backbone's).
- depth_project.onnx: the depth decoder's input projection for the backbone's hidden state (position 0).

`decoder` exports the Mimi decoder (32 codebooks -> 24 kHz audio) with a dynamic number of frames.

    qwen-venv/bin/python tools/export/export_csm.py hf|embeds|decoder SRC OUT
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_qwen3_tts import gather_graph, sliding_causal_mask, windowed_transformer  # noqa: E402

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')


def weights(src: Path, prefix=None) -> dict:
    index = json.loads((src / 'transformers.safetensors.index.json').read_text())['weight_map']
    out = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(src / shard), 'pt') as f:
            for k in f.keys():
                if prefix is None or k.startswith(prefix):
                    out[k] = f.get_tensor(k)
    return out


def llama_config(c: dict, vocab: int) -> dict:
    return {
        'architectures': ['LlamaForCausalLM'], 'model_type': 'llama', 'hidden_size': c['hidden_size'],
        'intermediate_size': c['intermediate_size'], 'num_hidden_layers': c['num_hidden_layers'],
        'num_attention_heads': c['num_attention_heads'], 'num_key_value_heads': c['num_key_value_heads'],
        'head_dim': c['head_dim'], 'hidden_act': c['hidden_act'], 'rms_norm_eps': c['rms_norm_eps'],
        'rope_theta': c['rope_theta'], 'rope_scaling': c['rope_scaling'], 'max_position_embeddings': c['max_position_embeddings'],
        'vocab_size': vocab, 'tie_word_embeddings': False, 'attention_bias': False, 'mlp_bias': False,
        'torch_dtype': 'float32', 'bos_token_id': 128000, 'eos_token_id': 128001,
    }


def hf(src: Path, out: Path):
    cfg = json.loads((src / 'config.json').read_text())
    dc = cfg['depth_decoder_config']
    w = weights(src)
    heads = w['depth_decoder.codebooks_head.weight']  # [31, 1024, 2051]
    parts = {
        'backbone': ('backbone_model.', w['lm_head.weight'], cfg),
        'depth_decoder': ('depth_decoder.model.', heads.permute(0, 2, 1).reshape(-1, heads.shape[1]), dc),
    }
    for name, (prefix, head, c) in parts.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        t = {'model.' + k[len(prefix):]: v.contiguous() for k, v in w.items()
             if k.startswith(prefix) and ('.layers.' in k or k == prefix + 'norm.weight')}
        t['lm_head.weight'] = head.contiguous()
        t['model.embed_tokens.weight'] = head.contiguous().clone()  # unused (exclude_embeds); the vocab must match
        save_file(t, str(d / 'model.safetensors'), metadata={'format': 'pt'})
        (d / 'config.json').write_text(json.dumps(llama_config(c, head.shape[0]), indent=1))
        for f in TOKENIZER_FILES:
            shutil.copy(src / f, d / f)
        print(name, len(t), 'tensors, vocab', head.shape[0])


def embeds(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    w = {k: v.float() for k, v in weights(src).items() if 'layers.' not in k and not k.startswith('codec_model')}
    gather_graph(out / 'text_embed.onnx', {'embeds': w['embed_text_tokens.weight'].numpy()}, {})
    audio = w['backbone_model.embed_tokens.embed_audio_tokens.weight']
    assert torch.equal(audio, w['depth_decoder.model.embed_tokens.weight']), 'depth embeddings are tied to the backbone\'s'
    proj = w['depth_decoder.model.inputs_embeds_projector.weight']  # [1024, 2048], no bias
    gather_graph(out / 'audio_embed.onnx', {'embeds': audio.numpy(), 'depth_embeds': (audio @ proj.T).numpy()}, {})
    x = helper.make_tensor_value_info('hidden', TensorProto.FLOAT, ['batch', 'seq', proj.shape[1]])
    y = helper.make_tensor_value_info('depth_hidden', TensorProto.FLOAT, ['batch', 'seq', proj.shape[0]])
    g = helper.make_graph([helper.make_node('MatMul', ['hidden', 'w'], ['depth_hidden'])], 'depth_project', [x], [y],
                          [numpy_helper.from_array(proj.T.contiguous().numpy(), 'w')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 21)])
    m.ir_version = 10
    onnx.save(m, str(out / 'depth_project.onnx'))
    print('text', tuple(w['embed_text_tokens.weight'].shape), 'audio', tuple(audio.shape))


def decoder(src: Path, out: Path):
    """codes [batch, 32, frames] int64 -> audio [batch, 1, frames * 1920] (24 kHz)."""
    from transformers import MimiConfig, MimiModel
    from transformers.models.mimi import modeling_mimi as mm
    cfg = json.loads((src / 'config.json').read_text())['codec_config']
    model = MimiModel(MimiConfig(**{k: v for k, v in cfg.items() if k not in ('architectures', '_name_or_path')}))
    sd = {k[len('codec_model.'):]: v for k, v in weights(src, 'codec_model.').items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('mimi: missing', len(missing), 'unexpected', len(unexpected))
    model.eval()
    ref_model = MimiModel(model.config)
    ref_model.load_state_dict(sd, strict=False)
    ref_model.eval()
    for mod in model.modules():
        if isinstance(mod, mm.MimiConv1d):
            assert int(mod.stride) == 1 or mod.causal
            if int(mod.stride) == 1:
                mod._get_extra_padding_for_conv1d = lambda h: 0
    windowed_transformer(model.decoder_transformer, model.config.sliding_window)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, codes):
            emb = self.m.quantizer.decode(codes)
            emb = self.m.upsample(emb)
            h = self.m.decoder_transformer(emb.transpose(1, 2))[0]
            return self.m.decoder(h.transpose(1, 2))

    codes = torch.randint(0, 2048, (1, 32, 40))
    with torch.no_grad():
        ref = ref_model.decode(codes).audio_values
        got = Wrap()(codes)
    print('wrapper vs decode: max abs diff', float((ref - got).abs().max()), 'shape', tuple(got.shape))
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Wrap(), (codes,), str(out), input_names=['codes'], output_names=['audio'], opset_version=17,
                          dynamic_axes={'codes': {0: 'batch', 2: 'frames'}, 'audio': {0: 'batch', 2: 'samples'}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    for t in (9, 75):
        c = torch.randint(0, 2048, (1, 32, t))
        with torch.no_grad():
            r = ref_model.decode(c).audio_values.numpy()
        o = s.run(None, {'codes': c.numpy()})[0]
        print(f'onnx vs torch, {t} frames: max abs diff {np.abs(o - r).max():.2e}, shape {o.shape}')


def main():
    cmd, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {'hf': hf, 'embeds': embeds, 'decoder': decoder}[cmd](src, out)


if __name__ == '__main__':
    main()
