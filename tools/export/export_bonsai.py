"""Converts PrismML's Ternary Bonsai 2 (prism-ml/Ternary-Bonsai-2-{2B,4B,9B}-gguf, Apache 2.0) to ONNX for the app, keeping
its ternary weights exact. Bonsai 2 is Qwen3.5 (architecture unchanged) with every matrix ternary, −1, 0 or +1 times a
float16 scale per 128 weights (PQ2_0), so the graph is the catalogue's Qwen3.5 ONNX (onnx-community/Qwen3.5-{2B,4B}-ONNX,
and for 9B AMD's build, of which only the kept weights are fetched) with all its weights replaced from the GGUF:

- Each matrix becomes a 2-bit MatMulNBits, block size 128, no zero points: code q (weight (q − 1) × scale) stored as
  q + 1, read with the default zero point 2. Exact.
- PrismML folded a blockwise Hadamard rotation into the stored matrices (prism.hadamard.*): each one's input is
  multiplied by a ±1 sign vector of its width and then by the normalized Walsh–Hadamard matrix in blocks of 512, as their
  runtime (runtime.py: fwht) does. Here that is a Mul and, since the Sylvester matrix of a × b is the Kronecker
  product of those of a and b, two MatMuls by small ones (16 and 32, float32) on each block seen as a 16 × 32 matrix,
  before the layer. (9B: blocks of 4096, 64 × 64; one 4096 × 4096 matrix, 64 MB read for every rotation, made it
  run at half speed. It also has its own output layer instead of the tied one below.)
- The token embeddings are stored rotated: a 4-bit GatherBlockQuantized (codes q + 7, default zero point 8, block 128,
  exact) followed by the inverse transform (Hadamard, then the signs). The output layer is tied to them: the same codes
  as a 2-bit MatMulNBits with the forward transform.
- The linear attention's alpha and beta projections are float32 in the GGUF: float16 MatMuls here. Norms, the
  convolution, A and dt bias are taken as stored (they use the ONNX graph's conventions: verified against Qwen3.5).
- With more value heads than key heads (4B: 32 and 16), the GGUF orders the value heads differently from the ONNX
  graph (transformers' order): reordered as PrismML's runtime does (vperm).

    tools/export/.venv/bin/python tools/export/export_bonsai.py 2b|4b|9b [--out DIR]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from gguf_read import Gguf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / '.cache' / 'hf' / 'bonsai-2'
OWN_REPO = 'danyalmalik'

MODELS = {
    '2b': {'gguf': 'Ternary-Bonsai-2-2B-PQ2_0.gguf', 'source': 'prism-ml/Ternary-Bonsai-2-2B-gguf', 'base': 'onnx-community/Qwen3.5-2B-ONNX', 'layout': 'community'},
    '4b': {'gguf': 'Ternary-Bonsai-2-4B-PQ2_0.gguf', 'source': 'prism-ml/Ternary-Bonsai-2-4B-gguf', 'base': 'onnx-community/Qwen3.5-4B-ONNX', 'layout': 'community'},
    # The catalogue's Qwen3.5 9B is AMD's ONNX Runtime GenAI build (text.onnx and embedding.onnx).
    '9b': {'gguf': 'Ternary-Bonsai-2-9B-PQ2_0.gguf', 'source': 'prism-ml/Ternary-Bonsai-2-9B-gguf', 'base': 'amd/Qwen3.5-9B_gpuep_1.0_gpu_128', 'layout': 'amd',
           'revision': 'bad679999a75e4883384cdac008972614cfb0d00'},
}
# Graph files per layout: (decoder, embeddings), and what the converted ones are called.
FILES = {
    'community': (('onnx/decoder_model_merged_q4f16.onnx', 'onnx/decoder_model_merged_q2f16.onnx'), ('onnx/embed_tokens_q4f16.onnx', 'onnx/embed_tokens_q2f16.onnx')),
    'amd': (('text.onnx', 'text.onnx'), ('embedding.onnx', 'embedding.onnx')),
}
BLOCK = 128

# ONNX MatMulNBits node (per layer) → GGUF matrix.
# (onnx-community names the linear attention gdn, AMD linear_attn.)
LINEARS = {
    f'{g}/{k}': v
    for g in ('gdn', 'linear_attn')
    for k, v in {'in_proj_qkv': 'attn_qkv', 'in_proj_z': 'attn_gate', 'out_proj': 'ssm_out'}.items()
} | {
    'mlp/gate_proj': 'ffn_gate', 'mlp/up_proj': 'ffn_up', 'mlp/down_proj': 'ffn_down',
    'attn/q_proj': 'attn_q', 'attn/k_proj': 'attn_k', 'attn/v_proj': 'attn_v', 'attn/o_proj': 'attn_output',
}
FLOAT_LINEARS = {f'{g}/{k}': v for g in ('gdn', 'linear_attn') for k, v in {'in_proj_a': 'ssm_alpha', 'in_proj_b': 'ssm_beta'}.items()}
# ONNX initializer (per layer) → GGUF tensor.
SMALL = {
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'post_attention_norm.weight',
    'gdn.norm.weight': 'ssm_norm.weight', 'gdn.A_neg_exp': 'ssm_a', 'gdn.dt_bias': 'ssm_dt.bias',
    'gdn.conv1d.weight': 'ssm_conv1d.weight', 'gdn.conv1d.weight_3d': 'ssm_conv1d.weight',
    'linear_attn.norm.weight': 'ssm_norm.weight', 'linear_attn.neg_exp_A': 'ssm_a', 'linear_attn.dt_bias': 'ssm_dt.bias',
    'linear_attn.conv1d.weight': 'ssm_conv1d.weight',
    'attn.q_norm.layernorm.weight': 'attn_q_norm.weight', 'attn.k_norm.layernorm.weight': 'attn_k_norm.weight',
}
# Kept as they are: AMD's convolution bias (Qwen3.5 has none: zeros, checked).
ZERO = {'linear_attn.conv1d.bias'}


def slim_base(repo: str, revision: str, file: str, dest: Path):
    """The base graph with only the weights the conversion keeps: those are fetched by range from Hugging Face and
    stored inline, and the rest (the 4-bit matrices and the token table, all replaced) become empty placeholders. Saves
    downloading AMD's 6.5 GB for about 70 MB."""
    import urllib.request

    from onnx.external_data_helper import ExternalDataInfo, uses_external_data

    if dest.exists():
        return
    url = f'https://huggingface.co/{repo}/resolve/{revision}/'
    with urllib.request.urlopen(url + file) as r:
        model = onnx.ModelProto.FromString(r.read())
    replaced = {i for n in model.graph.node if n.op_type == 'MatMulNBits' for i in n.input[1:]}
    replaced |= {n.input[0] for n in model.graph.node if n.op_type == 'Gather' and 'embed_tokens' in n.input[0]}
    for init in model.graph.initializer:
        if not uses_external_data(init):
            continue
        info = ExternalDataInfo(init)
        del init.external_data[:]
        init.data_location = TensorProto.DEFAULT
        if init.name in replaced:
            continue
        request = urllib.request.Request(url + info.location, headers={'Range': f'bytes={info.offset}-{info.offset + info.length - 1}'})
        with urllib.request.urlopen(request) as r:
            init.raw_data = r.read()
        if len(init.raw_data) != info.length:
            raise ValueError(f'{init.name}: short read')
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))


def hadamard(n: int) -> np.ndarray:
    """The normalized Sylvester Walsh–Hadamard matrix (symmetric, its own inverse)."""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


class Converter:
    def __init__(self, gguf: Gguf):
        self.g = gguf
        meta = gguf.meta
        self.block = meta['prism.hadamard.block_size']
        if self.block not in (512, 4096) or meta.get('prism.hadamard.sign_mode') != 'explicit':
            raise ValueError('Unexpected Hadamard settings')
        self.signs, offset = {}, 0
        for width in meta['prism.hadamard.sign_widths']:
            self.signs[width] = np.array(meta['prism.hadamard.sign_values'][offset:offset + width], np.float32)
            offset += width
        self.folded = set(meta['prism.hadamard.weight_names'])
        self.tied = bool(meta.get('prism.hadamard.tied_output'))
        if set(meta['prism.hadamard.inverse_weight_names']) != {'token_embd.weight'} or self.tied == ('output.weight' in gguf.tensors):
            raise ValueError('Expected a rotated token embedding and either a tied or a separate output layer')
        self.nv, self.nk = meta['qwen35.ssm.time_step_rank'], meta['qwen35.ssm.group_count']
        if self.nv != self.nk and not meta.get('prism.hadamard.gdn_v_grouped'):
            raise ValueError('Ungrouped value heads (not handled by PrismML either)')
        self.hd = meta['qwen35.ssm.inner_size'] // self.nv
        self.hk = meta['qwen35.ssm.state_size']
        self.nodes, self.inits, self.transformed = [], [], {}

    def vperm(self, unit: int) -> np.ndarray:
        return np.arange(self.nv * unit).reshape(self.nv // self.nk, self.nk, unit).transpose(1, 0, 2).reshape(-1)

    def reorder(self, a: np.ndarray, stem: str) -> np.ndarray:
        """PrismML's runtime.py reorder: the GGUF's value-head order to transformers'."""
        if self.nv == self.nk:
            return a
        qk = 2 * self.nk * self.hk
        if stem in ('attn_qkv.weight', 'ssm_conv1d.weight'):
            return np.concatenate([a[:qk], a[qk:][self.vperm(self.hd)]], axis=0)
        if stem == 'attn_gate.weight':
            return a[self.vperm(self.hd)]
        if stem in ('ssm_alpha.weight', 'ssm_beta.weight', 'ssm_a', 'ssm_dt.bias'):
            return a[self.vperm(1)]
        return a

    def init(self, name: str, array: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(array, name))
        return name

    def ternary(self, gguf_name: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
        codes, scales = self.g.ternary(gguf_name)
        return self.reorder(codes, stem), self.reorder(scales, stem)

    def rotate(self, x: str, width: int, inverse=False) -> str:
        """x [.., width] float16 → its Hadamard transform (signs first; with `inverse`, signs last), float16."""
        key = (x, width, inverse)
        if key in self.transformed:
            return self.transformed[key]
        name = f'/hadamard{"_inv" if inverse else ""}/{x.strip("/").replace("/", "_")}'
        signs = self.init(f'hadamard.signs.{width}', self.signs[width]) if f'hadamard.signs.{width}' not in {i.name for i in self.inits} else f'hadamard.signs.{width}'
        # block = a × b; the block's transform is H_a · X · H_b with X its a × b view (H symmetric).
        a = 1 << ((self.block.bit_length() - 1) // 2)
        b = self.block // a
        have = {i.name for i in self.inits}
        ha, hb = f'hadamard.matrix.{a}', f'hadamard.matrix.{b}'
        for size, h in ((a, ha), (b, hb)):
            if h not in have:
                self.init(h, hadamard(size))
                have.add(h)
        y = self.node('Cast', [x], f'{name}/Cast', to=TensorProto.FLOAT)
        if not inverse:
            y = self.node('Mul', [y, signs], f'{name}/signs')
        # Any rank: blocks as a × b matrices, then back to x's shape.
        shape = self.node('Shape', [x], f'{name}/Shape')
        y = self.node('Reshape', [y, self.const(f'{name}/blocks', [-1, a, b])], f'{name}/Reshape')
        y = self.node('MatMul', [y, hb], f'{name}/MatMul_b')
        y = self.node('MatMul', [ha, y], f'{name}/MatMul_a')
        y = self.node('Reshape', [y, shape], f'{name}/Flat')
        if inverse:
            y = self.node('Mul', [y, signs], f'{name}/signs')
        y = self.node('Cast', [y], f'{name}/Cast_f16', to=TensorProto.FLOAT16)
        self.transformed[key] = y
        return y

    def node(self, op: str, inputs: list[str], name: str, **attrs) -> str:
        out = f'{name}/output_0'
        self.nodes.append(helper.make_node(op, inputs, [out], name=name, **attrs))
        return out

    def const(self, name: str, values) -> str:
        return self.init(name, np.array(values, np.int64))

    def matmul_ternary(self, x: str, gguf_name: str, stem: str, output: str, name: str):
        codes, scales = self.ternary(gguf_name, stem)
        n, k = codes.shape
        if gguf_name in self.folded or (self.tied and gguf_name == 'token_embd.weight'):
            x = self.rotate(x, k)
        q = (codes + 1).astype(np.uint8)
        packed = (q.reshape(n, k // 4, 4) << (2 * np.arange(4, dtype=np.uint8))).sum(-1).astype(np.uint8).reshape(n, k // BLOCK, BLOCK // 4)
        b = self.init(f'{name}.weight_Q2', packed)
        s = self.init(f'{name}.weight_scales', scales.astype(np.float16).reshape(-1))
        self.nodes.append(helper.make_node('MatMulNBits', [x, b, s], [output], name=name, domain='com.microsoft', K=k, N=n, bits=2, block_size=BLOCK, accuracy_level=0))

    def decoder(self, base: onnx.ModelProto) -> onnx.ModelProto:
        g = self.g
        skip = set()
        for node in base.graph.node:
            if node.op_type == 'MatMulNBits':
                skip.update(node.input[1:])
        for node in base.graph.node:
            if node.op_type != 'MatMulNBits':
                self.nodes.append(node)
                continue
            path = node.name.removesuffix('/MatMul_Quant').removesuffix('/MatMul_Q4')
            if path == '/lm_head':
                head = 'token_embd.weight' if self.tied else 'output.weight'
                self.matmul_ternary(node.input[0], head, head, node.output[0], node.name)
                continue
            layer, kind = path.split('/', 3)[2].removeprefix('layers.'), '/'.join(path.split('/')[3:])
            if kind in LINEARS:
                stem = f'{LINEARS[kind]}.weight'
                self.matmul_ternary(node.input[0], f'blk.{layer}.{stem}', stem, node.output[0], node.name)
            elif kind in FLOAT_LINEARS:
                stem = f'{FLOAT_LINEARS[kind]}.weight'
                w = self.reorder(g.array(f'blk.{layer}.{stem}'), stem)
                wt = self.init(f'{node.name}.weight_T', w.T.astype(np.float16).copy())
                self.nodes.append(helper.make_node('MatMul', [node.input[0], wt], list(node.output), name=node.name))
            else:
                raise ValueError(f'Unmapped linear layer {node.name}')
        layers = g.meta['qwen35.block_count']
        for init in base.graph.initializer:
            if init.name in skip:
                continue
            name = init.name
            replaced = None
            if name.startswith('model.layers.'):
                layer, rest = name.removeprefix('model.layers.').split('.', 1)
                if int(layer) == layers and rest == 'final_norm_layernorm.weight':
                    replaced = g.array('output_norm.weight')
                elif rest in SMALL:
                    stem = SMALL[rest]
                    replaced = self.reorder(g.array(f'blk.{layer}.{stem}'), stem)
                elif rest in ZERO:
                    if np.abs(numpy_helper.to_array(init)).max() != 0:
                        raise ValueError(f'{name} is not zero')
            if replaced is not None:
                old = numpy_helper.to_array(init)
                if old.size != replaced.size:
                    raise ValueError(f'{name}: {old.shape} vs {replaced.shape}')
                self.init(name, replaced.reshape(old.shape).astype(old.dtype))
            elif name.startswith('model.layers.') and name.split('.', 3)[3] not in ZERO:
                raise ValueError(f'Unmapped weight {name}')
            else:
                self.inits.append(init)
        graph = helper.make_graph(self.nodes, base.graph.name, list(base.graph.input), list(base.graph.output), self.inits)
        return helper.make_model(graph, opset_imports=list(base.opset_import), ir_version=base.ir_version)

    def embeddings(self, base: onnx.ModelProto) -> onnx.ModelProto:
        """The graph's lookup in the token table (a GatherBlockQuantized, or a Gather of a float table) becomes a 4-bit
        GatherBlockQuantized of the ternary codes followed by the inverse transform; every other node stays."""
        codes, scales = self.g.ternary('token_embd.weight')
        h = codes.shape[1]
        q = (codes + 7).astype(np.uint8)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        tables = {i.name for i in base.graph.initializer if len(i.dims) == 2 and i.dims[0] == codes.shape[0]}
        lookup = next(n for n in base.graph.node if n.op_type in ('Gather', 'GatherBlockQuantized') and n.input[0] in tables)
        b = self.init('embed_tokens.weight_Q4', packed)
        s = self.init('embed_tokens.weight_scales', scales.astype(np.float16))
        ids = lookup.input[1]
        gathered = f'{lookup.name}/ternary'
        nodes = [helper.make_node('GatherBlockQuantized', [b, ids, s], [gathered], name=f'{lookup.name}/Quant', domain='com.microsoft',
                                  bits=4, block_size=BLOCK, gather_axis=0, quantize_axis=1)]
        self.nodes = []
        rotated = self.rotate(gathered, h, inverse=True)
        nodes += self.nodes
        nodes.append(helper.make_node('Identity', [rotated], [lookup.output[0]], name=f'{lookup.name}/out'))
        dropped = set(lookup.input) - {ids}
        rest = [n for n in base.graph.node if n is not lookup]
        inits = self.inits + [i for i in base.graph.initializer if i.name not in dropped]
        graph = helper.make_graph(nodes + rest, base.graph.name, list(base.graph.input), list(base.graph.output), inits)
        return helper.make_model(graph, opset_imports=list(base.opset_import), ir_version=base.ir_version)

def model_card(size: str, out: Path) -> str:
    spec = MODELS[size]
    name = out.name
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name != 'README.md')
    qwen = f'Qwen/Qwen3.5-{size.upper()}'
    return f"""---
license: apache-2.0
base_model:
- {spec['source']}
tags:
- onnx
- webgpu
- ternary
- bonsai
- quick-inference
---

# {OWN_REPO}/{name}

[{spec['source']}](https://huggingface.co/{spec['source']}) (PrismML's Ternary Bonsai 2, built from
[{qwen}](https://huggingface.co/{qwen})) converted to ONNX for running in a web browser with ONNX Runtime Web's WebGPU
backend, as used by Quick Inference. Text only: the vision tower is not included.

Created using Bonsai by Prism ML.

## Changes from the original

- The graph is the Qwen3.5 ONNX graph of [{spec['base']}](https://huggingface.co/{spec['base']}), with every weight
  replaced by Bonsai's from `{spec['gguf']}`.
- The ternary matrices (−1, 0 or +1 times a float16 scale per 128 weights) are stored as 2-bit `MatMulNBits` with
  the same values: the weights are exact.
- Bonsai stores its matrices in a rotated basis (a blockwise Walsh–Hadamard transform with sign vectors); the graph
  applies the matching transform to each layer's input, as PrismML's runtime does. The token embeddings are a 4-bit
  `GatherBlockQuantized` of the same ternary values followed by the inverse transform.
- The linear attention's alpha and beta projections (float32 in the GGUF) are float16 here.

These are modified files. Checked against PrismML's MLX runtime: on a test prompt the last position's logits agree to
a cosine similarity of 0.99999 (same top tokens).

## Licence

Apache 2.0, as the original: see LICENSE and NOTICE.txt.

## Files

{chr(10).join(f'- `{f}`' for f in files)}
"""


def export(size: str, out: Path):
    spec = MODELS[size]
    gguf = Gguf(CACHE / f'gguf-{size}' / spec['gguf'])
    base = CACHE / f'onnx-{size}'
    if spec['layout'] == 'amd':
        for source, _ in FILES['amd']:
            slim_base(spec['base'], spec['revision'], source, base / source)
        for f in ('tokenizer.json', 'config.json'):
            if not (base / f).exists():
                shutil.copy(ROOT / 'tools' / 'reference' / '.cache' / spec['base'].replace('/', '__') / spec['revision'] / f, base / f)
    out.mkdir(parents=True, exist_ok=True)
    for kind, (source, target) in zip(('decoder', 'embeddings'), FILES[spec['layout']]):
        model = onnx.load(str(base / source))
        c = Converter(gguf)
        converted = c.decoder(model) if kind == 'decoder' else c.embeddings(model)
        path = out / target
        path.parent.mkdir(parents=True, exist_ok=True)
        data = path.name + ('.data' if spec['layout'] == 'amd' else '_data')
        # onnx appends to an existing data file rather than replacing it.
        (path.parent / data).unlink(missing_ok=True)
        onnx.save(converted, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=1024)
        print('saved', target, (path.parent / data).stat().st_size, flush=True)
    for f in ('tokenizer.json', 'config.json'):
        shutil.copy(base / f, out / f)
    shutil.copy(CACHE / f'gguf-{size}' / 'LICENSE', out / 'LICENSE')
    notice = (CACHE / f'gguf-{size}' / 'NOTICE.txt').read_text().rstrip()
    (out / 'NOTICE.txt').write_text(f'{notice}\n\nCreated using Bonsai by Prism ML. Converted to ONNX by Quick Inference (see README.md).\n')
    (out / 'README.md').write_text(model_card(size, out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', choices=sorted(MODELS))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    export(args.size, args.out or ROOT / 'exports' / f'Ternary-Bonsai-2-{args.size.upper()}-ONNX-WebGPU')
