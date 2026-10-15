"""Exports aMUSEd (a masked image-token transformer, diffusers' AmusedPipeline) to ONNX in the layout the image engine's
'amused' pipeline reads, ready to upload to Hugging Face:

- text_encoder/model.onnx: input_ids (int32, [batch, 77]) → encoder_hidden_states (the penultimate layer) and
  text_embeds (the pooled projection); float16 weights, float32 outputs.
- transformer/model.onnx: input_ids (int64, [1, h, w]; the mask token for unknown positions), micro_conds (float32,
  [1, 5]), text_embeds and encoder_hidden_states (float32, the negative prompt's then the prompt's: batch 2), guidance
  (float32, [1]) → logits (float16, [h · w, codebook]): the guided mix of the two rows, so only one row comes back.
  Float16 weights and maths.
- vq_decoder/model.onnx: input_ids (int64, [1, h, w]) → sample ([1, 3, 16h, 16w] in [−1, 1]); float32.
- tokenizer/, scheduler/: copied.

    tools/export/.venv/bin/python tools/export/export_amused.py amused-512          # export
    tools/export/.venv/bin/python tools/export/export_amused.py amused-512 --check  # compare with diffusers (CPU)
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from diffusers import UVit2DModel, VQModel
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import CLIPTextModelWithProjection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / '.cache'
EXPORTS = ROOT / 'exports'
OPSET = 17

MODELS = {
    'amused-512': {
        'source': 'amused/amused-512',
        'revision': '6f3f309ba188a08e347bff9691a0ac791817ca97',
        'repo': 'amused-512-ONNX-WebGPU',
        'licenceFile': 'OPENRAIL_PLUS_PLUS_M.md',
    },
}
WEIGHT_FILES = ['text_encoder/model.safetensors', 'transformer/diffusion_pytorch_model.safetensors', 'vqvae/diffusion_pytorch_model.safetensors']


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids.long(), output_hidden_states=True)
        return out.hidden_states[-2].float(), out.text_embeds.float()


class Transformer(torch.nn.Module):
    """One image's logits with classifier-free guidance applied: rows are (negative prompt, prompt)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, micro_conds, text_embeds, encoder_hidden_states, guidance):
        h = torch.float16
        ids = torch.cat([input_ids, input_ids])
        out = self.transformer(ids, micro_conds=micro_conds.expand(2, -1).to(h), pooled_text_emb=text_embeds.to(h),
                               encoder_hidden_states=encoder_hidden_states.to(h))
        uncond, cond = out.float().chunk(2)
        mixed = uncond + guidance.reshape(1, 1, 1, 1) * (cond - uncond)
        return mixed.reshape(mixed.shape[1], -1).transpose(0, 1).to(h)


class VqDecoder(torch.nn.Module):
    def __init__(self, vq):
        super().__init__()
        self.vq = vq

    def forward(self, input_ids):
        b, hh, ww = input_ids.shape
        image = self.vq.decode(input_ids, force_not_quantize=True, shape=(b, hh, ww, self.vq.config.latent_channels)).sample
        return image.clamp(0, 1) * 2 - 1


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
    if not all((src / f).exists() for f in WEIGHT_FILES):
        includes = ['model_index.json', '*/config.json', 'scheduler/*', 'tokenizer/*', *WEIGHT_FILES]
        cmd = [HERE / '.venv/bin/hf', 'download', spec['source'], '--revision', spec['revision'], '--local-dir', src]
        subprocess.run([*cmd, *[a for i in includes for a in ('--include', i)]], check=True)
    return src


def trace_inputs(grid: int):
    g = torch.Generator().manual_seed(0)
    return (torch.randint(0, 8192, (1, grid, grid), generator=g), torch.tensor([[512.0, 512, 0, 0, 6]]),
            torch.randn(2, 768, generator=g), torch.randn(2, 77, 768, generator=g), torch.tensor([10.0]))


def main(key: str):
    spec = MODELS[key]
    src = source(spec)
    out = EXPORTS / spec['repo']
    # Exported in float32 and converted afterwards: PyTorch's exporter folds float16 weights into float32 constants.
    if not (out / 'text_encoder/model.onnx').exists():
        te = CLIPTextModelWithProjection.from_pretrained(src / 'text_encoder', dtype=torch.float32).eval()
        export(TextEncoder(te), (torch.zeros(1, 77, dtype=torch.int32),), out / 'text_encoder/model.onnx', ['input_ids'],
               ['encoder_hidden_states', 'text_embeds'],
               {'input_ids': {0: 'batch'}, 'encoder_hidden_states': {0: 'batch'}, 'text_embeds': {0: 'batch'}}, to_half=True)
        del te
    if not (out / 'transformer/model.onnx').exists():
        tr = UVit2DModel.from_pretrained(src / 'transformer', dtype=torch.float16).eval()
        # A small grid for the trace (the sizes are dynamic): PyTorch traces float16 on the CPU slowly.
        export(Transformer(tr), trace_inputs(8), out / 'transformer/model.onnx',
               ['input_ids', 'micro_conds', 'text_embeds', 'encoder_hidden_states', 'guidance'], ['logits'],
               {'input_ids': {1: 'height', 2: 'width'}, 'logits': {0: 'positions'}})
        del tr
    if not (out / 'vq_decoder/model.onnx').exists():
        vq = VQModel.from_pretrained(src / 'vqvae', dtype=torch.float32).eval()
        export(VqDecoder(vq), (torch.randint(0, 8192, (1, 8, 8)),), out / 'vq_decoder/model.onnx', ['input_ids'], ['sample'],
               {'input_ids': {1: 'height', 2: 'width'}, 'sample': {2: 'height', 3: 'width'}})
        del vq
    for folder in ['tokenizer', 'scheduler']:
        shutil.copytree(src / folder, out / folder, dirs_exist_ok=True)
    shutil.copy(src / 'model_index.json', out / 'model_index.json')
    shutil.copy(HERE / 'licences' / spec['licenceFile'], out / 'LICENSE.md')
    (out / 'README.md').write_text(model_card(spec))


def check(key: str):
    """Each exported graph (ONNX Runtime, CPU) against the original in float32 on the same inputs: cosine similarity."""
    import onnxruntime as ort

    spec = MODELS[key]
    src = source(spec)
    out = EXPORTS / spec['repo']
    run = lambda name, feeds: ort.InferenceSession(str(out / name / 'model.onnx'), providers=['CPUExecutionProvider']).run(None, feeds)
    cos = lambda a, b: float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 49407, (2, 77), generator=g, dtype=torch.int32)
    with torch.no_grad():
        te = CLIPTextModelWithProjection.from_pretrained(src / 'text_encoder', dtype=torch.float32).eval()
        ref = te(ids.long(), output_hidden_states=True)
        hidden, pooled = run('text_encoder', {'input_ids': ids.numpy()})
        print('text_encoder cos', cos(hidden, ref.hidden_states[-2].numpy()), cos(pooled, ref.text_embeds.numpy()), flush=True)
        del te
        tokens, conds, _, _, guidance = trace_inputs(32)
        tokens[0, :, 16:] = 8255  # half the grid still masked, as midway through a generation
        tr = UVit2DModel.from_pretrained(src / 'transformer', dtype=torch.float32).eval()
        logits = tr(torch.cat([tokens, tokens]), micro_conds=conds.expand(2, -1), pooled_text_emb=torch.from_numpy(pooled),
                    encoder_hidden_states=torch.from_numpy(hidden)).float()
        uncond, cond = logits.chunk(2)
        ref = (uncond + guidance * (cond - uncond)).reshape(8192, -1).T.numpy()
        del tr
        got = run('transformer', {'input_ids': tokens.numpy(), 'micro_conds': conds.numpy(), 'text_embeds': pooled,
                                  'encoder_hidden_states': hidden, 'guidance': guidance.numpy()})[0].astype(np.float32)
        print('transformer cos', cos(got, ref), 'argmax agreement', float((got.argmax(1) == ref.argmax(1)).mean()), flush=True)
        vq = VQModel.from_pretrained(src / 'vqvae', dtype=torch.float32).eval()
        ref = (VqDecoder(vq)(tokens.clamp(max=8191))).numpy()
        img = run('vq_decoder', {'input_ids': tokens.clamp(max=8191).numpy()})[0]
        print('vq_decoder cos', cos(img, ref), 'range', float(img.min()), float(img.max()), flush=True)


def model_card(spec: dict) -> str:
    source = spec['source']
    return f"""---
license: openrail++
base_model:
- {source}
tags:
- onnx
- webgpu
- text-to-image
- quick-inference
---

# {spec['repo']}

[{source}](https://huggingface.co/{source}) converted to ONNX for running in a web browser with
ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to ONNX (opset {OPSET}) with PyTorch's exporter: `text_encoder`, `transformer` and `vq_decoder`.
- The text encoder returns its penultimate hidden layer and pooled projection, as aMUSEd uses them.
- The transformer runs the prompt and the negative prompt together and returns their guided mix
  (`uncond + guidance · (cond − uncond)`), one row of logits per image token; float16 weights and output.
- The VQ decoder takes token ids and returns the image in [−1, 1]; float32.
- Sampling and the masking schedule run outside the graphs (as diffusers' AmusedScheduler).

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence and use restrictions

Licensed under the CreativeML Open RAIL++-M License, as the original (LICENSE.md). Its use-based restrictions
(paragraph 5 and Attachment A) apply to these files and to anything made from them: you may not use them for the
uses listed in Attachment A, and if you distribute them or a derivative you must include those restrictions as an
enforceable provision and give recipients a copy of the licence.
"""


if __name__ == '__main__':
    check(sys.argv[1]) if '--check' in sys.argv else main(sys.argv[1])
