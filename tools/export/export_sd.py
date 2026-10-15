"""Exports a Stable Diffusion 1.x checkpoint in diffusers' layout (including the distilled small ones) to ONNX in the
layout the image engine's 'sd' pipeline reads, ready to upload to Hugging Face:

- text_encoder/model.onnx: input_ids (int32, [batch, 77]) → last_hidden_state; float16 weights, float32 output.
- unet/model.onnx: sample, timestep (float32, [batch]), encoder_hidden_states → out_sample; float16 weights and maths,
  float32 inputs and outputs.
- vae_decoder/model.onnx: latent_sample → sample, float32 (AutoencoderKL: divide latents by 0.18215 first;
  AutoencoderTiny/TAESD: latents as they are).
- tokenizer/: copied.

    tools/export/.venv/bin/python tools/export/export_sd.py tiny-sd          # export
    tools/export/.venv/bin/python tools/export/export_sd.py tiny-sd --check  # compare with diffusers (CPU, float32)
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import CLIPTextModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / '.cache'
EXPORTS = ROOT / 'exports'
OPSET = 17

CREATIVEML = {'licence': 'creativeml-openrail-m', 'licenceName': 'CreativeML Open RAIL-M', 'licenceFile': 'CREATIVEML_OPENRAIL_M.md'}
MODELS = {
    'sdxs-512-dreamshaper': {
        'source': 'IDKiro/sdxs-512-dreamshaper',
        'revision': '76f720262bb051da75666b22c902a78c8e16c763',
        'repo': 'sdxs-512-dreamshaper-ONNX-WebGPU',
        'weights': 'safetensors',
        'vae': 'tiny',
        'licence': 'openrail++',
        'licenceName': 'CreativeML Open RAIL++-M',
        'licenceFile': 'OPENRAIL_PLUS_PLUS_M.md',
    },
    'tiny-sd': {'source': 'segmind/tiny-sd', 'revision': 'cad0bd7495fa6c4bcca01b19a723dc91627fe84f', 'repo': 'tiny-sd-ONNX-WebGPU', 'weights': 'bin', 'vae': 'kl', **CREATIVEML},
    'small-sd': {'source': 'segmind/small-sd', 'revision': 'f8b087e68071a29bbe85a4b07e6e1e8998d59aa7', 'repo': 'small-sd-ONNX-WebGPU', 'weights': 'bin', 'vae': 'kl', **CREATIVEML},
    'bk-sdm-small-2m': {
        'source': 'nota-ai/bk-sdm-small-2m',
        'revision': '33411b0f1dd8beb66aa5f431973ea006ea7611e6',
        'repo': 'bk-sdm-small-2m-ONNX-WebGPU',
        'weights': 'safetensors',
        'vae': 'kl',
        **CREATIVEML,
    },
}
WEIGHT_FILES = {
    'safetensors': ['text_encoder/model.safetensors', 'unet/diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.safetensors'],
    'bin': ['text_encoder/pytorch_model.bin', 'unet/diffusion_pytorch_model.bin', 'vae/diffusion_pytorch_model.bin'],
}


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids.long()).last_hidden_state.float()


class Unet(torch.nn.Module):
    """Float32 in and out, float16 inside."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        h = torch.float16
        return self.unet(sample.to(h), timestep.to(h), encoder_hidden_states.to(h), return_dict=False)[0].float()


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
    onnx.save(model, str(path))
    shutil.rmtree(tmp)
    print('saved', path, f'{path.stat().st_size / 1e6:.0f} MB', flush=True)


def source(spec: dict) -> Path:
    src = CACHE / 'hf' / spec['source'].split('/')[1]
    if not all((src / f).exists() for f in WEIGHT_FILES[spec['weights']]):
        includes = ['model_index.json', '*/config.json', 'scheduler/*', 'tokenizer/*', *WEIGHT_FILES[spec['weights']]]
        cmd = [HERE / '.venv/bin/hf', 'download', spec['source'], '--revision', spec['revision'], '--local-dir', src]
        subprocess.run([*cmd, *[a for i in includes for a in ('--include', i)]], check=True)
    return src


def load_vae(spec: dict, src: Path):
    cls = AutoencoderTiny if spec['vae'] == 'tiny' else AutoencoderKL
    return cls.from_pretrained(src / 'vae', dtype=torch.float32).eval()


def main(key: str):
    spec = MODELS[key]
    src = source(spec)
    out = EXPORTS / spec['repo']
    # The text encoder is exported in float32 and converted to float16 afterwards: PyTorch's exporter folds float16
    # weights into float32 constants, which leaves the graph's types inconsistent.
    if not (out / 'text_encoder/model.onnx').exists():
        te = CLIPTextModel.from_pretrained(src / 'text_encoder', dtype=torch.float32).eval()
        export(TextEncoder(te), (torch.zeros(1, 77, dtype=torch.int32),), out / 'text_encoder/model.onnx', ['input_ids'],
               ['last_hidden_state'], {'input_ids': {0: 'batch'}, 'last_hidden_state': {0: 'batch'}}, to_half=True)
        del te
    if not (out / 'unet/model.onnx').exists():
        unet = UNet2DConditionModel.from_pretrained(src / 'unet', dtype=torch.float16).eval()
        # Small trace inputs (the sizes are dynamic): PyTorch runs the trace in float16 on the CPU, which is slow.
        args = (torch.randn(1, 4, 16, 16), torch.tensor([999.0]), torch.randn(1, 77, 768))
        dyn = {'sample': {0: 'batch', 2: 'height', 3: 'width'}, 'timestep': {0: 'batch'}, 'encoder_hidden_states': {0: 'batch'},
               'out_sample': {0: 'batch', 2: 'height', 3: 'width'}}
        export(Unet(unet), args, out / 'unet/model.onnx', ['sample', 'timestep', 'encoder_hidden_states'], ['out_sample'], dyn)
        del unet
    if not (out / 'vae_decoder/model.onnx').exists():
        vae = load_vae(spec, src)
        export(VaeDecoder(vae), (torch.randn(1, 4, 64, 64),), out / 'vae_decoder/model.onnx', ['latent_sample'], ['sample'],
               {'latent_sample': {0: 'batch', 2: 'height', 3: 'width'}, 'sample': {0: 'batch', 2: 'height', 3: 'width'}})
        del vae
    shutil.copytree(src / 'tokenizer', out / 'tokenizer', dirs_exist_ok=True)
    shutil.copy(src / 'model_index.json', out / 'model_index.json')
    shutil.copytree(src / 'scheduler', out / 'scheduler', dirs_exist_ok=True)
    shutil.copy(HERE / 'licences' / spec['licenceFile'], out / 'LICENSE.md')
    (out / 'README.md').write_text(model_card(spec))


def check(key: str):
    """Each exported graph (ONNX Runtime, CPU) against the original in float32 on the same inputs: cosine similarity."""
    import onnxruntime as ort

    spec = MODELS[key]
    src = source(spec)
    out = EXPORTS / spec['repo']
    run = lambda name, feeds: ort.InferenceSession(str(out / name / 'model.onnx'), providers=['CPUExecutionProvider']).run(None, feeds)[0]
    cos = lambda a, b: float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 49407, (1, 77), generator=g, dtype=torch.int32)
    with torch.no_grad():
        te = CLIPTextModel.from_pretrained(src / 'text_encoder', dtype=torch.float32).eval()
        ref = te(ids.long()).last_hidden_state.numpy()
        del te
        hidden = run('text_encoder', {'input_ids': ids.numpy()})
        print('text_encoder cos', cos(hidden, ref), flush=True)
        sample, t = torch.randn(1, 4, 64, 64, generator=g), torch.tensor([500.0])
        unet = UNet2DConditionModel.from_pretrained(src / 'unet', dtype=torch.float32).eval()
        ref = unet(sample, t, torch.from_numpy(ref), return_dict=False)[0].numpy()
        del unet
        eps = run('unet', {'sample': sample.numpy(), 'timestep': t.numpy(), 'encoder_hidden_states': hidden})
        print('unet cos', cos(eps, ref), 'nan', bool(np.isnan(eps).any()), flush=True)
        latents = torch.randn(1, 4, 64, 64, generator=g)
        vae = load_vae(spec, src)
        ref = vae.decode(latents, return_dict=False)[0].numpy()
        img = run('vae_decoder', {'latent_sample': latents.numpy()})
        print('vae_decoder cos', cos(img, ref), 'range', float(img.min()), float(img.max()), flush=True)


def model_card(spec: dict) -> str:
    source = spec['source']
    decoder = 'The VAE decoder (TAESD, a tiny decoder)' if spec['vae'] == 'tiny' else 'The VAE decoder'
    return f"""---
license: {spec['licence']}
base_model:
- {source}
tags:
- onnx
- webgpu
- stable-diffusion
- text-to-image
- quick-inference
---

# {spec['repo']}

[{source}](https://huggingface.co/{source}) converted to ONNX for running in a web browser with
ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to ONNX (opset {OPSET}) with PyTorch's exporter: `text_encoder`, `unet` and `vae_decoder`.
- The text encoder and the UNet are float16; both take and return float32. {decoder} is float32.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence and use restrictions

Licensed under the {spec['licenceName']} License, as the original (LICENSE.md). Its use-based restrictions (paragraph 5
and Attachment A) apply to these files and to anything made from them: you may not use them for the uses listed in
Attachment A, and if you distribute them or a derivative you must include those restrictions as an enforceable
provision and give recipients a copy of the licence.
"""


if __name__ == '__main__':
    check(sys.argv[1]) if '--check' in sys.argv else main(sys.argv[1])
