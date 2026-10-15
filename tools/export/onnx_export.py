"""Shared pieces of our own ONNX decoder exporters (export_glm.py, export_nemotron.py): checkpoint tensors read one at
a time (optionally downloading each shard when first needed and deleting it when done), a graph whose initializers
stream to one external data file, and the quantised building blocks ONNX Runtime's WebGPU backend runs."""

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper
from onnxruntime_genai.models.quantization.cuda_quantizer import CudaQuantizer
from safetensors import safe_open

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
MS = 'com.microsoft'


class Settings:
    """Weight quantisation: 4 bits for the exports, 8 for checking an export against the original."""

    bits = 4
    block = 32


class Weights:
    """Checkpoint tensors by name, read one at a time as float32. With `stream`, each shard is downloaded when first
    needed and deleted once every tensor the export uses from it has been read, so the checkpoint needs only a shard
    or two of disk space at a time."""

    def __init__(self, folder: Path, source: str, revision: str, stream=False, skip_prefix=None):
        index = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
        self.folder, self.source, self.revision = folder, source, revision
        self.index, self.files, self.stream = index, {}, stream
        self.pending = {}
        for name, file in index.items():
            if skip_prefix is None or not name.startswith(skip_prefix):
                self.pending.setdefault(file, set()).add(name)

    def __call__(self, name: str, native: bool = False) -> torch.Tensor:
        """The tensor as float32, or in its stored type with `native` (for big ones, converted in parts later)."""
        file = self.index[name]
        if file not in self.files:
            if not (self.folder / file).exists():
                hf_hub_download(self.source, file, revision=self.revision, local_dir=self.folder)
            self.files[file] = safe_open(str(self.folder / file), framework='pt')
        tensor = self.files[file].get_tensor(name)
        if not native:
            tensor = tensor.float()
        left = self.pending.get(file)
        if self.stream and left is not None:
            left.discard(name)
            if not left:
                del self.files[file]
                (self.folder / file).unlink()
                print('done with', file, flush=True)
        return tensor


def to_float16(t: torch.Tensor) -> np.ndarray:
    """A big tensor as float16, converted a few thousand rows at a time."""
    out = np.empty(t.shape, np.float16)
    for start in range(0, t.shape[0], ROWS):
        out[start:start + ROWS] = t[start:start + ROWS].float().numpy()
    return out


WINDOW = 2**30


class Graph:
    """Nodes plus initializers streamed to one external data file (tensors under 1 KB stay inline)."""

    def __init__(self, data_path: Path):
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self.data_path = data_path
        self.data = open(data_path, 'wb')
        self.offset = 0

    def init(self, name: str, array: np.ndarray, dtype: int) -> str:
        raw = np.ascontiguousarray(array).tobytes()
        t = TensorProto(name=name, data_type=dtype, dims=list(array.shape))
        if len(raw) < 1024:
            t.raw_data = raw
        else:
            pad = (-self.offset) % 64
            # The app reads weight files over 2 GiB through 2 GiB windows that start every 1 GiB, so a tensor must fit
            # inside one: when it would cross the end of its window, it starts at the next 1 GiB.
            start = self.offset + pad
            if start % WINDOW + len(raw) > 2 * WINDOW - 1:
                pad = -self.offset % WINDOW
                print(f'{name}: padded {pad} bytes to fit a 2 GiB window', flush=True)
            self.data.write(b'\0' * pad)
            self.offset += pad
            self.data.write(raw)
            t.data_location = TensorProto.EXTERNAL
            for k, v in (('location', self.data_path.name), ('offset', str(self.offset)), ('length', str(len(raw)))):
                t.external_data.add(key=k, value=v)
            self.offset += len(raw)
        self.inits.append(t)
        return name

    def const(self, name: str, values, dtype=I64) -> str:
        np_type = {I64: np.int64, F16: np.float16, F32: np.float32, I32: np.int32}[dtype]
        return self.init(name, np.array(values, dtype=np_type), dtype)

    def node(self, op: str, inputs, name: str, outputs=1, domain='', **attrs):
        outs = [f'{name}/output_{i}' for i in range(outputs)] if isinstance(outputs, int) else outputs
        self.nodes.append(helper.make_node(op, list(inputs), outs, name=name, domain=domain or None, **attrs))
        return outs[0] if len(outs) == 1 else outs

    def close(self):
        self.data.close()


ROWS = 8192


def matmul_nbits(g: Graph, x: str, w: torch.Tensor, name: str) -> str:
    """x @ w.T with w ([N, K], any float type) quantised (ONNX Runtime's signed-scale blockwise encoding, no zero
    points), a few thousand rows at a time so a big matrix never needs more than a slice of it in float32."""
    n, k = w.shape
    bits, block = Settings.bits, Settings.block
    q_parts, s_parts = [], []
    for start in range(0, n, ROWS):
        q, s = CudaQuantizer.matmulnbits_blockwise_quantize(w[start:start + ROWS].float(), bits, block, flatten_qweight=False)
        q_parts.append(q.numpy().astype(np.uint8))
        s_parts.append(s.numpy().astype(np.float16))
    qn = g.init(f'{name}.weight_Q{bits}G{block}', np.concatenate(q_parts), U8)
    sn = g.init(f'{name}.weight_scale', np.concatenate(s_parts).reshape(-1), F16)
    return g.node('MatMulNBits', [x, qn, sn], name, domain=MS, K=k, N=n, bits=bits, block_size=block, accuracy_level=0)


def rms_norm(g: Graph, x: str, w: torch.Tensor, name: str, eps: float) -> str:
    wn = g.init(f'{name}.weight', w.numpy().astype(np.float16), F16)
    return g.node('SimplifiedLayerNormalization', [x, wn], name, axis=-1, epsilon=eps, stash_type=1)


def skip_rms_norm(g: Graph, x: str, skip: str, w: torch.Tensor, name: str, eps: float):
    """(norm(x + skip), x + skip)."""
    wn = g.init(f'{name}.weight', w.numpy().astype(np.float16), F16)
    outs = [f'{name}/output_0', '', '', f'{name}/output_3']
    g.node('SkipSimplifiedLayerNormalization', [x, skip, wn], name, outputs=outs, domain=MS, epsilon=eps)
    return outs[0], outs[3]


def attention_lengths(g: Graph) -> dict:
    """GroupQueryAttention's lengths from the attention mask: tokens seen so far minus one, and the total."""
    m = '/model/attn_mask_reformat'
    total = g.node('ReduceSum', ['attention_mask', g.const(f'{m}/axes', [1])], f'{m}/ReduceSum')
    seqlens = g.node('Cast', [g.node('Sub', [total, g.const(f'{m}/one', [1])], f'{m}/Sub')], f'{m}/Sub/Cast', to=I32)
    shape = g.node('Shape', ['attention_mask'], f'{m}/Shape')
    total_len = g.node('Cast', [g.node('Gather', [shape, g.const(f'{m}/index', 1)], f'{m}/Gather', axis=0)], f'{m}/Gather/Cast', to=I32)
    return {'seqlens': seqlens, 'total': total_len}


def sigmoid_router(g: Graph, x: str, weight: torch.Tensor, bias: torch.Tensor, k: int, name: str) -> str:
    """Router input for QMoE from a DeepSeek-V3-style router (one group): experts are chosen by sigmoid score plus a
    correction bias and weighted by the plain scores, normalised over the chosen ones. QMoE applies a softmax to its
    router input, so it gets log(score) for the chosen experts and -10000 elsewhere: the softmax over that is each
    chosen score divided by their sum. Computed in float32, as the original does."""
    experts = weight.shape[0]
    x32 = g.node('Cast', [x], f'{name}/Cast', to=F32)
    w = g.init(f'{name}.weight_T', weight.T.contiguous().numpy(), F32)
    logits = g.node('MatMul', [x32, w], f'{name}/MatMul')
    logits = g.node('Reshape', [logits, g.const(f'{name}/shape', [-1, experts])], f'{name}/Reshape')
    scores = g.node('Sigmoid', [logits], f'{name}/Sigmoid')
    choice = g.node('Add', [scores, g.init(f'{name}.e_score_correction_bias', bias.numpy(), F32)], f'{name}/Add')
    _, idx = g.node('TopK', [choice, g.const(f'{name}/k', [k])], f'{name}/TopK', outputs=2, axis=-1)
    chosen = g.node('GatherElements', [scores, idx], f'{name}/GatherElements', axis=1)
    log_w = g.node('Log', [chosen], f'{name}/Log')
    floor = g.node('Expand', [g.const(f'{name}/floor', -10000.0, F32), g.node('Shape', [scores], f'{name}/Shape')], f'{name}/Expand')
    probs = g.node('ScatterElements', [floor, idx, log_w], f'{name}/ScatterElements', axis=1)
    return g.node('Cast', [probs], f'{name}/Cast_16', to=F16)


def qmoe_weights(w: torch.Tensor, bits: int, block: int):
    """One expert's [N, K] weight in QMoE's raw layout: [N, K * bits / 8] bytes and [N, K / block] scales."""
    q, s = CudaQuantizer.matmulnbits_blockwise_quantize(w, bits, block)
    return q[:, : w.shape[1] * bits // 8].numpy().astype(np.uint8), s.numpy().astype(np.float16)
