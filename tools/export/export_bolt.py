"""NVIDIA's Nemotron 3.5 Bolt E4B (a post-trained Gemma 4 E4B) as ONNX for WebGPU, by putting its weights into
onnx-community's Gemma 4 E4B q4f16 graphs: the text model is the same architecture, so the graphs stay as they are and
every weight keeps its shape, place and quantisation (4-bit, asymmetric, block 32, float16 scales):

- MatMulNBits weights (attention, MLP with gate and up concatenated, per-layer gate/projection, the per-layer model
  projection, and the output layer, tied to the token table) and the two GatherBlockQuantized token tables are
  re-quantised from Bolt's bfloat16 weights, written at the same offsets in copies of the data files.
- Norm weights and layer scalars are copied as float16; the rotary tables and other constants stay.
- The vision and audio towers are left out, as in the app's Gemma 4 E4B (text only), and so are the k/v projections
  of the 18 layers that reuse an earlier layer's cache (Bolt's model-frozen-shared-kv.safetensors).

    tools/export/.venv/bin/python tools/export/export_bolt.py [--check]

`--check` first re-quantises onnx-community's own (dequantised) Gemma weights and reports how many codes come out the
same, to confirm the quantiser matches theirs.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / '.cache' / 'hf' / 'bolt-e4b'          # nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
REVISION = '49843eed8db1f1c616c47af75cfd36e07aec0667'
BASE = ROOT / '.cache' / 'hf' / 'gemma4-e4b-onnx'  # onnx-community/gemma-4-E4B-it-ONNX
BASE_REVISION = 'd298100958fdf3c60581158fe201fb240634fb28'
REPO = 'NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX-WebGPU'
GRAPHS = ('embed_tokens_q4f16', 'decoder_model_merged_q4f16')
BLOCK = 32
ROWS = 8192  # rows quantised at a time, to keep memory low on the 262,144-row tables
L = 'model.language_model.'
KEEP = re.compile(r'^(/model/|cos_cache_|sin_cache_)')


class Bolt:
    def __init__(self):
        self.map = json.loads((SRC / 'model.safetensors.index.json').read_text())['weight_map']
        self.files = {}

    def slice(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(str(SRC / f), 'pt')
        return self.files[f].get_slice(name)

    def rows(self, names, start, stop):
        """Rows start:stop of the weights `names` stacked along the output dimension, as float32."""
        parts, offset = [], 0
        for n in names:
            s = self.slice(n)
            count = s.get_shape()[0]
            a, b = max(start - offset, 0), min(stop - offset, count)
            if a < b:
                parts.append(s[a:b].float().numpy())
            offset += count
        return np.concatenate(parts)

    def tensor(self, name):
        return self.slice(name)[:].float().numpy()


def source(name: str):
    """Bolt's weight(s) for an initializer of the ONNX graphs, without the _quant/_scales/_zp suffix."""
    if name == 'model_embed_tokens_weight' or name == 'lm_head_MatMul_weight':
        return [L + 'embed_tokens.weight']
    if name == 'model_embed_tokens_per_layer_weight':
        return [L + 'embed_tokens_per_layer.weight']
    if name == 'model_per_layer_projection_MatMul_weight':
        return [L + 'per_layer_model_projection.weight']
    if name == 'model.per_layer_projection_norm.weight':
        return [L + 'per_layer_projection_norm.weight']
    if name == 'model.layers.42.final_norm_layernorm.weight':
        return [L + 'norm.weight']
    m = re.fullmatch(r'model_layers_(\d+)_(.+)_MatMul_weight', name)
    if m:
        layer, part = m.groups()
        p = f'{L}layers.{layer}.'
        if part == 'mlp_gate_up_proj':
            return [p + 'mlp.gate_proj.weight', p + 'mlp.up_proj.weight']
        return [p + {'attn_q_proj': 'self_attn.q_proj', 'attn_k_proj': 'self_attn.k_proj', 'attn_v_proj': 'self_attn.v_proj',
                     'attn_o_proj': 'self_attn.o_proj', 'mlp_down_proj': 'mlp.down_proj',
                     'per_layer_per_layer_input_gate': 'per_layer_input_gate',
                     'per_layer_per_layer_projection': 'per_layer_projection'}[part] + '.weight']
    m = re.fullmatch(r'model\.layers\.(\d+)\.(attn\.[qk]_norm\.layernorm|[a-z_]+)\.weight', name) or re.fullmatch(r'model\.layers\.(\d+)\.(layer_scalar)', name)
    if m:
        layer, part = m.groups()
        part = {'attn.q_norm.layernorm': 'self_attn.q_norm', 'attn.k_norm.layernorm': 'self_attn.k_norm'}.get(part, part)
        return [f'{L}layers.{layer}.{part}' + ('' if part == 'layer_scalar' else '.weight')]
    raise KeyError(name)


def quantize(w: np.ndarray):
    """4-bit asymmetric blocks of 32 along each row (ONNX Runtime's MatMulNBits / GatherBlockQuantized layout):
    codes [rows, blocks, 16] (two per byte, low nibble first), float16 scales [rows, blocks], zero points packed two
    per byte [rows, ceil(blocks / 2)]."""
    rows, k = w.shape
    blocks = k // BLOCK
    b = w.reshape(rows, blocks, BLOCK)
    lo, hi = np.minimum(b.min(-1), 0), np.maximum(b.max(-1), 0)
    scale = ((hi - lo) / 15).astype(np.float16)
    s = scale.astype(np.float32)
    safe = np.where(s == 0, 1, s)
    zp = np.where(s == 0, 8, np.clip(np.round(-lo / safe), 0, 15)).astype(np.uint8)
    q = np.clip(np.round(b / safe[..., None]) + zp[..., None], 0, 15).astype(np.uint8)
    codes = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    if blocks % 2:
        zp = np.concatenate([zp, np.zeros((rows, 1), np.uint8)], 1)
    zps = (zp[:, 0::2] | (zp[:, 1::2] << 4)).astype(np.uint8)
    return codes, scale, zps


def dequantize(codes, scale, zps, k):
    rows, blocks = scale.shape
    q = np.stack([codes & 15, codes >> 4], -1).reshape(rows, blocks, BLOCK).astype(np.float32)
    z = np.stack([zps & 15, zps >> 4], -1).reshape(rows, -1)[:, :blocks].astype(np.float32)
    return ((q - z[..., None]) * scale.astype(np.float32)[..., None]).reshape(rows, k)


def location(t: onnx.TensorProto):
    info = {e.key: e.value for e in t.external_data}
    return info['location'], int(info.get('offset', 0)), int(info['length'])


def check(model_dir: Path):
    """Re-quantise a few of onnx-community's tensors from their own dequantised values: the codes should come back."""
    model = onnx.load(str(model_dir / 'decoder_model_merged_q4f16.onnx'), load_external_data=False)
    inits = {t.name: t for t in model.graph.initializer}
    for name in ('model_layers_0_attn_q_proj_MatMul_weight', 'model_layers_7_mlp_down_proj_MatMul_weight'):
        parts = []
        for suffix in ('quant', 'scales', 'zp'):
            t = onnx.TensorProto()
            t.CopyFrom(inits[f'{name}_{suffix}'])
            onnx.external_data_helper.load_external_data_for_tensor(t, str(model_dir))
            parts.append(numpy_helper.to_array(t))
        codes, scale, zps = parts
        k = scale.shape[1] * BLOCK
        c2, s2, z2 = quantize(dequantize(codes.reshape(scale.shape[0], -1, 16), scale, zps, k))
        print(f'{name}: codes {np.mean(c2.reshape(codes.shape) == codes):.4f}, scales {np.mean(s2 == scale):.4f}, '
              f'zero points {np.mean(z2 == zps):.4f} the same')


def export(out: Path):
    bolt = Bolt()
    onnx_dir = out / 'onnx'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for graph in GRAPHS:
        model = onnx.load(str(BASE / 'onnx' / f'{graph}.onnx'), load_external_data=False)
        files = sorted({location(t)[0] for t in model.graph.initializer if t.data_location == 1})
        for f in files:
            shutil.copyfile(BASE / 'onnx' / f, onnx_dir / f)
        handles = {f: open(onnx_dir / f, 'r+b') for f in files}
        inits = {t.name: t for t in model.graph.initializer}
        done = set()

        def write(t, data: np.ndarray, row0=0):
            """Write data (rows row0.. of tensor t) into its place in the copied data file."""
            f, offset, length = location(t)
            row_bytes = length // t.dims[0]
            raw = np.ascontiguousarray(data).tobytes()
            assert len(raw) % row_bytes == 0 and row0 * row_bytes + len(raw) <= length, t.name
            handles[f].seek(offset + row0 * row_bytes)
            handles[f].write(raw)

        for name, t in inits.items():
            if name in done or KEEP.match(name):
                continue
            if name.endswith('_quant'):
                base = name[: -len('_quant')]
                names = source(base)
                tq, ts, tz = inits[name], inits[base + '_scales'], inits[base + '_zp']
                rows = tq.dims[0]
                for start in range(0, rows, ROWS):
                    w = bolt.rows(names, start, min(start + ROWS, rows))
                    codes, scale, zps = quantize(w)
                    write(tq, codes.reshape((w.shape[0],) + tuple(tq.dims[1:])), start)
                    write(ts, scale, start)
                    write(tz, zps, start)
                done |= {name, base + '_scales', base + '_zp'}
                print(f'{graph}: {base} <- {", ".join(n[len(L):] for n in names)}', flush=True)
            elif name.endswith(('_scales', '_zp')):
                continue
            else:
                (hf,) = source(name)
                value = bolt.tensor(hf).astype(np.float16).reshape(tuple(t.dims))
                if t.data_location == 1:
                    write(t, value)
                else:
                    t.CopyFrom(numpy_helper.from_array(value, name))
                done.add(name)
        left = [n for n in inits if n not in done and not KEEP.match(n) and not n.endswith(('_scales', '_zp'))]
        assert not left, left
        for h in handles.values():
            h.close()
        onnx.save(model, str(onnx_dir / f'{graph}.onnx'))
        print(f'{graph}: {len(done)} weights from Bolt', flush=True)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'config.json'):
        shutil.copy(SRC / name, out / name)
    shutil.copy(ROOT / 'tools' / 'export' / 'licences' / 'OPENMDW_1.1.txt', out / 'LICENSE')
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = f"""---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
pipeline_tag: text-generation
tags:
- onnx
- webgpu
- quick-inference
---

# {REPO}

NVIDIA's [Nemotron 3.5 Bolt E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) (post-trained from
Google's Gemma 4 E4B) converted to ONNX for running in a web browser with ONNX Runtime Web's WebGPU backend, as used by
Quick Inference. Text only.

## Changes from the original

- Bolt's text weights are placed into the graphs of
  [onnx-community/gemma-4-E4B-it-ONNX](https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX) (revision
  `{BASE_REVISION[:8]}`), which share its architecture, by Quick Inference's `tools/export/export_bolt.py`
  ([Quick Inference](https://github.com/mdanyalmalik/quick-inf)): `onnx/embed_tokens_q4f16.onnx` (token and
  per-layer embeddings) and `onnx/decoder_model_merged_q4f16.onnx` (the decoder with its KV cache).
- Linear layers and both embedding tables are quantised to 4 bits (round to nearest, asymmetric, block size 32,
  float16 scales); norms and activations are float16.
- The vision and audio encoders are not included.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

[OpenMDW-1.1](https://openmdw.ai/license/1-1/), as the original (LICENSE). Bolt is post-trained from Google's Gemma 4
E4B, which is released under the Apache License 2.0; the ONNX graphs come from onnx-community's Apache 2.0 conversion
of it.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check(BASE / 'onnx')
    export(args.out)
