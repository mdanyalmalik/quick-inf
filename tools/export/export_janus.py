"""Exports Janus Pro 7B's text-to-image generation to ONNX for the image engine's 'janus' pipeline:

- language_model.onnx (+ .data): the Llama decoder, input embeddings in, final hidden states out (hidden_states), 4-bit
  (export_dense.py janus-pro-7b, run first).
- embed_tokens.onnx: prompt token ids → embeddings (float16).
- gen_head.onnx: a hidden state → logits over the 16,384 image tokens (float16).
- gen_img_embeds.onnx: an image token → the next input embedding (codebook entry through the generation aligner).
- image_decode.onnx: the 24 × 24 image tokens → a 384 × 384 image in [-1, 1] (the VQ decoder, float32).

The vision encoder (image understanding) is left out. Each step of generation runs the prompt and an unconditional
copy (the prompt's tokens replaced by padding) as a batch of two for classifier-free guidance.

    tools/export/.venv/bin/python tools/export/export_dense.py janus-pro-7b
    tools/export/.venv/bin/python tools/export/export_janus.py
"""

import shutil
from pathlib import Path

import onnx
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import AutoConfig
from transformers.models.janus.modeling_janus import JanusVQVAE, JanusVQVAEAlignerMLP, JanusVQVAEHead

from onnx_export import Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'Janus-Pro-7B'
SOURCE = 'deepseek-community/Janus-Pro-7B'
REVISION = '8f23139ed8d37346501091b3690291cb7b16f9fc'
OUT = ROOT / 'exports' / 'Janus-Pro-7B-ONNX-WebGPU'
OPSET = 17


def load(module: torch.nn.Module, W, prefix: str) -> torch.nn.Module:
    module.load_state_dict({k: W(prefix + k) for k in module.state_dict()}, strict=True)
    return module.float().eval()


def export(module, args, name: str, inputs, outputs, dynamic, half: bool):
    path = OUT / name
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs, dynamic_axes=dynamic, opset_version=OPSET, dynamo=False)
    if half:
        # Exported in float32 and converted afterwards (PyTorch's exporter mixes types when tracing float16).
        model = convert_float_to_float16(onnx.load(str(path)), keep_io_types=False, disable_shape_infer=True)
        onnx.save(model, str(path))
    print('saved', path, path.stat().st_size, flush=True)


class EmbedTokens(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.embed = torch.nn.Embedding.from_pretrained(weight)

    def forward(self, input_ids):
        return self.embed(input_ids)


class GenImgEmbeds(torch.nn.Module):
    def __init__(self, embeddings, aligner):
        super().__init__()
        self.embeddings, self.aligner = embeddings, aligner

    def forward(self, image_ids):
        return self.aligner(self.embeddings(image_ids))


class ImageDecode(torch.nn.Module):
    def __init__(self, vq):
        super().__init__()
        self.vq = vq

    def forward(self, image_tokens):
        return self.vq.decode(image_tokens)


def main():
    config = AutoConfig.from_pretrained(SRC)
    W = Weights(SRC, SOURCE, REVISION)
    OUT.mkdir(parents=True, exist_ok=True)

    embed = EmbedTokens(W('model.language_model.embed_tokens.weight'))
    export(embed, (torch.zeros(2, 8, dtype=torch.int64),), 'embed_tokens.onnx', ['input_ids'], ['inputs_embeds'],
           {'input_ids': {0: 'batch', 1: 'sequence'}, 'inputs_embeds': {0: 'batch', 1: 'sequence'}}, half=True)
    del embed

    head = load(JanusVQVAEHead(config.vq_config), W, 'model.generation_head.')
    export(head, (torch.zeros(2, 1, config.vq_config.image_token_embed_dim),), 'gen_head.onnx', ['hidden_states'], ['logits'],
           {'hidden_states': {0: 'batch'}, 'logits': {0: 'batch'}}, half=True)

    embeddings = torch.nn.Embedding.from_pretrained(W('model.generation_embeddings.weight'))
    aligner = load(JanusVQVAEAlignerMLP(config.vq_config), W, 'model.generation_aligner.')
    export(GenImgEmbeds(embeddings, aligner).eval(), (torch.zeros(2, 1, dtype=torch.int64),), 'gen_img_embeds.onnx', ['image_ids'],
           ['inputs_embeds'], {'image_ids': {0: 'batch', 1: 'sequence'}, 'inputs_embeds': {0: 'batch', 1: 'sequence'}}, half=True)

    vq = JanusVQVAE(config.vq_config)
    vq_state = {k: W('model.vqmodel.' + k) for k in vq.state_dict() if not k.startswith('encoder.') and not k.startswith('quant_conv')}
    vq.load_state_dict(vq_state, strict=False)
    export(ImageDecode(vq.float().eval()), (torch.zeros(1, config.vision_config.num_image_tokens, dtype=torch.int64),), 'image_decode.onnx',
           ['generated_tokens'], ['decoded_image'], None, half=False)

    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json', 'generation_config.json'):
        shutil.copy(SRC / name, OUT / name)
    shutil.copy(HERE / 'licences' / 'DEEPSEEK_MODEL_LICENSE.md', OUT / 'LICENSE-MODEL.md')
    shutil.copy(HERE / 'licences' / 'JANUS_LICENSE_CODE', OUT / 'LICENSE-CODE')
    (OUT / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = """---
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/Janus/blob/main/LICENSE-MODEL
base_model:
- deepseek-ai/Janus-Pro-7B
tags:
- onnx
- webgpu
- text-to-image
- quick-inference
---

# Janus-Pro-7B-ONNX-WebGPU

The text-to-image part of [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) (converted from
[deepseek-community/Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B), the same weights in
transformers' format), converted to ONNX for running in a web browser with ONNX Runtime Web's WebGPU backend, as used
by Quick Inference.

## Changes from the original

- Split into five ONNX graphs: `language_model` (input embeddings → hidden states, with a KV cache),
  `embed_tokens`, `gen_head`, `gen_img_embeds` and `image_decode` (the VQ decoder). The vision encoder used for image
  understanding is left out.
- The language model's linear layers are quantised to 4 bits (round to nearest, block size 32); its activations and
  KV cache are float16. `embed_tokens`, `gen_head` and `gen_img_embeds` are float16; `image_decode` is float32.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence and use restrictions

The code of Janus is MIT-licensed (LICENSE-CODE); the model weights are licensed under the DeepSeek License
(LICENSE-MODEL.md), as the original. Its use-based restrictions (section 5 and Attachment A) apply to these files and
to anything made from them: you may not use them for the uses listed in Attachment A, and if you distribute them or a
derivative you must include those restrictions as an enforceable provision and give recipients a copy of the licence.
"""


if __name__ == '__main__':
    main()
