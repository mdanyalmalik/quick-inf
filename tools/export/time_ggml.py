"""Timing graphs for MatMulGgml (kernels.html): x [1, 1, 7168] through 40 real square matrices of one type from
different layers of a GGUF (so they don't sit in cache), into tools/reference/.cache/micro/0/ggml_<type>_x40.onnx.

    tools/export/.venv/bin/python tools/export/time_ggml.py
    node tools/bench/bench.mjs --page kernels "x:graphs=ggml_iq2_s_x40:40,ggml_iq3_s_x40:40&k=7168"
"""

from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import TensorProto, helper

import ggml_pack

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'reference' / '.cache' / 'micro' / '0'
import sys
GGUF = HERE / '.cache' / 'hf' / 'nanbeige-gguf' / (sys.argv[1] if len(sys.argv) > 1 else 'Nanbeige_Nanbeige4.3-33B-IQ2_M.gguf')
COUNT = 40

found = {}
for t in gguf.GGUFReader(GGUF).tensors:
    if len(t.shape) == 2 and int(t.shape[0]) == 7168 and int(t.shape[1]) == 7168 and int(t.tensor_type) in ggml_pack.LAYOUTS:
        found.setdefault(int(t.tensor_type), []).append(t)
for kind, tensors in found.items():
    if len(tensors) < COUNT:
        continue
    name = f'ggml_{gguf.GGMLQuantizationType(kind).name.lower()}_x{COUNT}'
    nodes, inits = [], []
    grid = ggml_pack.grid(kind)
    if grid is not None:
        inits.append(helper.make_tensor('grid', TensorProto.UINT8, [grid.nbytes], grid.view(np.uint8).tobytes(), raw=True))
    x = 'x'
    for i, t in enumerate(tensors[:COUNT]):
        packed = ggml_pack.pack(kind, np.asarray(t.data).reshape(7168, -1), 7168, 7168)
        inits.append(helper.make_tensor(f'w{i}', TensorProto.UINT8, [7168, packed.shape[1] * 4], packed.view(np.uint8).tobytes(), raw=True))
        y = 'y' if i == COUNT - 1 else f'h{i}'
        nodes.append(helper.make_node('MatMulGgml', [x, f'w{i}'] + (['grid'] if grid is not None else []), [y], domain='com.microsoft', K=7168, N=7168, ggml_type=kind))
        x = y
    io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT16, [1, 'm', 7168])
    graph = helper.make_graph(nodes, name, [io('x')], [io('y')], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)], ir_version=10)
    onnx.save(model, str(OUT / f'{name}.onnx'))
    print(name, sum(i.ByteSize() for i in inits) / 1e6, 'MB')
