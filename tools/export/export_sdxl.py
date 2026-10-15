"""Exports an SDXL checkpoint in diffusers' layout to ONNX in the layout the image engine's 'sdxl' pipeline reads,
ready to upload to Hugging Face:

- text_encoder/model.onnx: input_ids (int32, [1, 77]) → hidden_states.11 (the penultimate layer), float16 weights.
- text_encoder_2/model.onnx: input_ids (int64, [1, 77]) → hidden_states.31 and text_embeds, float16 weights.
- unet/model.onnx (+ model.onnx.data): sample, timestep, encoder_hidden_states, text_embeds, time_ids → out_sample;
  float16 weights and maths, float32 inputs and outputs.
- vae_decoder/model.onnx: latent_sample → sample, float32 throughout (SDXL's VAE overflows in float16).
- tokenizer/, tokenizer_2/: copied.

    tools/export/.venv/bin/python tools/export/export_sdxl.py animagine-xl-3.1
"""

import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import CLIPTextModel, CLIPTextModelWithProjection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / '.cache'
EXPORTS = ROOT / 'exports'
OPSET = 17

MODELS = {
    'animagine-xl-3.1': {
        'source': 'cagliostrolab/animagine-xl-3.1',
        'repo': 'animagine-xl-3.1-ONNX-WebGPU',
        'licenceFile': 'OPENRAIL_PLUS_PLUS_M.md',
    },
}


class TextEncoder(torch.nn.Module):
    def __init__(self, model, layer: int, pooled: bool):
        super().__init__()
        self.model, self.layer, self.pooled = model, layer, pooled

    def forward(self, input_ids):
        out = self.model(input_ids.long(), output_hidden_states=True)
        hidden = out.hidden_states[self.layer].float()
        return (hidden, out.text_embeds.float()) if self.pooled else hidden


class Unet(torch.nn.Module):
    """Float32 in and out, float16 inside."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        h = torch.float16
        out = self.unet(
            sample.to(h),
            timestep.to(h),
            encoder_hidden_states.to(h),
            added_cond_kwargs={'text_embeds': text_embeds.to(h), 'time_ids': time_ids.to(h)},
            return_dict=False,
        )[0]
        return out.float()


class VaeDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample, return_dict=False)[0]


def export(module, args, path: Path, inputs, outputs, dynamic, to_half=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / 'build' / path.parent.name
    tmp.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module, args, str(tmp / 'model.onnx'), input_names=inputs, output_names=outputs,
            dynamic_axes=dynamic, opset_version=OPSET, dynamo=False,
        )
    model = onnx.load(str(tmp / 'model.onnx'))
    if to_half:
        model = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    big = sum(t.ByteSize() for t in model.graph.initializer) > 1.9e9
    if big:
        onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location='model.onnx.data')
    else:
        onnx.save(model, str(path))
    shutil.rmtree(tmp)
    print('saved', path, flush=True)


def main(key: str):
    spec = MODELS[key]
    src = CACHE / 'hf' / spec['source'].split('/')[1]
    out = EXPORTS / spec['repo']
    if not (src / 'model_index.json').exists():
        subprocess.run([HERE / '.venv/bin/hf', 'download', spec['source'], '--exclude', '*.safetensors', '--local-dir', src], check=True)
    half = torch.float16
    # The text encoders are exported in float32 and converted to float16 afterwards: PyTorch's exporter folds
    # float16 weights into float32 constants, which leaves the graph's types inconsistent.
    if not (out / 'text_encoder/model.onnx').exists():
        te = CLIPTextModel.from_pretrained(src / 'text_encoder', dtype=torch.float32).eval()
        export(TextEncoder(te, 11, False), (torch.zeros(1, 77, dtype=torch.int32),), out / 'text_encoder/model.onnx',
               ['input_ids'], ['hidden_states.11'], None, to_half=True)
        del te
    if not (out / 'text_encoder_2/model.onnx').exists():
        te2 = CLIPTextModelWithProjection.from_pretrained(src / 'text_encoder_2', dtype=torch.float32).eval()
        export(TextEncoder(te2, 31, True), (torch.zeros(1, 77, dtype=torch.int64),), out / 'text_encoder_2/model.onnx',
               ['input_ids'], ['hidden_states.31', 'text_embeds'], None, to_half=True)
        del te2
    if not (out / 'unet/model.onnx').exists():
        unet = UNet2DConditionModel.from_pretrained(src / 'unet', dtype=half).eval()
        args = (torch.randn(2, 4, 64, 64), torch.tensor([999.0, 999.0]), torch.randn(2, 77, 2048), torch.randn(2, 1280),
                torch.tensor([[512.0, 512, 0, 0, 512, 512]] * 2))
        dyn = {'sample': {0: 'batch', 2: 'height', 3: 'width'}, 'timestep': {0: 'batch'}, 'encoder_hidden_states': {0: 'batch'},
               'text_embeds': {0: 'batch'}, 'time_ids': {0: 'batch'}, 'out_sample': {0: 'batch', 2: 'height', 3: 'width'}}
        export(Unet(unet), args, out / 'unet/model.onnx', ['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids'],
               ['out_sample'], dyn)
        del unet
    if not (out / 'vae_decoder/model.onnx').exists():
        vae = AutoencoderKL.from_pretrained(src / 'vae', dtype=torch.float32).eval()
        export(VaeDecoder(vae), (torch.randn(1, 4, 64, 64),), out / 'vae_decoder/model.onnx', ['latent_sample'], ['sample'],
               {'latent_sample': {0: 'batch', 2: 'height', 3: 'width'}, 'sample': {0: 'batch', 2: 'height', 3: 'width'}})
        del vae
    for folder in ['tokenizer', 'tokenizer_2']:
        shutil.copytree(src / folder, out / folder, dirs_exist_ok=True)
    shutil.copy(src / 'model_index.json', out / 'model_index.json')
    shutil.copytree(src / 'scheduler', out / 'scheduler', dirs_exist_ok=True)
    shutil.copy(HERE / 'licences' / spec['licenceFile'], out / 'LICENSE.md')
    (out / 'README.md').write_text(model_card(spec))


def model_card(spec: dict) -> str:
    source = spec['source']
    return f"""---
license: openrail++
base_model:
- {source}
tags:
- onnx
- webgpu
- stable-diffusion-xl
- text-to-image
- quick-inference
---

# {spec['repo']}

[{source}](https://huggingface.co/{source}) converted to ONNX for running in a web browser with
ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to ONNX (opset {OPSET}) with PyTorch's exporter: `text_encoder`, `text_encoder_2`, `unet` and `vae_decoder`.
- The text encoders and the UNet are float16; the UNet takes and returns float32. The VAE decoder is float32.
- The text encoders return the penultimate hidden layer (and the second its pooled projection), as SDXL uses them.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence and use restrictions

Licensed under the CreativeML Open RAIL++-M License, as the original (LICENSE.md). Its use-based restrictions
(paragraph 5 and Attachment A) apply to these files and to anything made from them: you may not use them for the
uses listed in Attachment A, and if you distribute them or a derivative you must include those restrictions as an
enforceable provision and give recipients a copy of the licence.
"""


if __name__ == '__main__':
    main(sys.argv[1])
