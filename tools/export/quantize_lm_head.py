"""Quantises an exported decoder's LM head to 4 bits after the builder has left it in float16.

The builder's k-quant needs tens of GB of memory for a 150k-row LM head, so export_llm.py excludes that MatMul from
it; this replaces it with MatMulNBits (round to nearest, block size 32, ONNX Runtime's signed-scale encoding),
quantising a few thousand rows at a time, and rewrites model.onnx.data without the float16 weight.

    tools/export/.venv/bin/python tools/export/quantize_lm_head.py exports/<repo>
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from onnxruntime_genai.models.quantization.cuda_quantizer import CudaQuantizer

BLOCK = 32
ROWS = 4096


def external(t: TensorProto) -> dict:
    return {e.key: e.value for e in t.external_data}


def main(folder: Path):
    model = onnx.load(str(folder / 'model.onnx'), load_external_data=False)
    graph = model.graph
    node = next(n for n in graph.node if n.op_type == 'MatMul' and n.name.startswith('/lm_head/'))
    inits = {t.name: t for t in graph.initializer}
    weight = inits[node.input[1]]
    if any(node.input[1] in n.input for n in graph.node if n is not node):
        raise SystemExit('the LM head weight is shared with another node')
    k, n = weight.dims
    data_path = folder / 'model.onnx.data'
    src = np.memmap(data_path, dtype=np.uint8, mode='r')
    info = external(weight)
    w = np.frombuffer(src[int(info['offset']):int(info['offset']) + int(info['length'])], dtype=np.float16).reshape(k, n)

    q_parts, s_parts = [], []
    for start in range(0, n, ROWS):
        rows = torch.from_numpy(np.ascontiguousarray(w[:, start:start + ROWS].T).astype(np.float32))
        q, s = CudaQuantizer.matmulnbits_blockwise_quantize(rows, 4, BLOCK, flatten_qweight=False)
        q_parts.append(q.numpy().astype(np.uint8))
        s_parts.append(s.numpy().astype(np.float16))
    q = np.concatenate(q_parts)
    s = np.concatenate(s_parts).reshape(-1)

    # Rewrite the data file: every other external tensor, then the new ones.
    tmp = data_path.with_suffix('.data.tmp')
    offset = 0
    with open(tmp, 'wb') as out:
        def put(raw: bytes) -> int:
            nonlocal offset
            pad = (-offset) % 64
            out.write(b'\0' * pad)
            offset += pad
            at = offset
            out.write(raw)
            offset += len(raw)
            return at

        for t in graph.initializer:
            if t is weight or t.data_location != TensorProto.EXTERNAL:
                continue
            e = external(t)
            at = put(src[int(e['offset']):int(e['offset']) + int(e['length'])].tobytes())
            for entry in t.external_data:
                if entry.key == 'offset':
                    entry.value = str(at)
        new = []
        for name, array, dtype in ((f'{node.name}.weight_Q4G32', q, TensorProto.UINT8), (f'{node.name}.weight_scale', s, TensorProto.FLOAT16)):
            raw = array.tobytes()
            t = TensorProto(name=name, data_type=dtype, dims=list(array.shape), data_location=TensorProto.EXTERNAL)
            for key, value in (('location', data_path.name), ('offset', str(put(raw))), ('length', str(len(raw)))):
                t.external_data.add(key=key, value=value)
            new.append(t)
    del src, w

    graph.initializer.remove(weight)
    graph.initializer.extend(new)
    replacement = helper.make_node(
        'MatMulNBits', [node.input[0], new[0].name, new[1].name], list(node.output), name=node.name + '_Q4',
        domain='com.microsoft', K=k, N=n, bits=4, block_size=BLOCK, accuracy_level=0,
    )
    index = list(graph.node).index(node)
    graph.node.remove(node)
    graph.node.insert(index, replacement)
    tmp.replace(data_path)
    onnx.save(model, str(folder / 'model.onnx'))
    print(f'LM head [{n} × {k}] → 4 bits; data file {offset} bytes')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
