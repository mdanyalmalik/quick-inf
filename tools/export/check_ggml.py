"""Single-op cases for MatMulGgml and GatherGgml (opcheck.html): real rows of each llama.cpp type from the GGUFs in
.cache/hf/nanbeige-gguf, random float16 inputs, expected outputs from ggml_pack's numpy decoding (float64 sums).
Written to the dev server's model cache (tools/reference/.cache/micro/0/); prints the case names.

    tools/export/.venv/bin/python tools/export/check_ggml.py
    node tools/bench/bench.mjs --page opcheck "x:cases=$(tools/export/.venv/bin/python tools/export/check_ggml.py --names)"
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import TensorProto, helper

import ggml_pack

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'reference' / '.cache' / 'micro' / '0'
SOURCES = ['Nanbeige4.3-33B-GSQ-RCO-IQ2_M.gguf', 'Nanbeige_Nanbeige4.3-33B-IQ2_M.gguf', 'Nanbeige_Nanbeige4.3-33B-Q6_K.gguf']
ROWS = 200  # not a multiple of the kernels' row tiles
ROWS_OF_A = [1, 3, 8, 9, 40]


def tensors():
    """(type, K, rows bytes) for one tensor of each type, the widest K found."""
    found = {}
    for name in SOURCES:
        path = HERE / '.cache' / 'hf' / 'nanbeige-gguf' / name
        if not path.exists():
            continue
        for t in gguf.GGUFReader(path).tensors:
            kind, k = int(t.tensor_type), int(t.shape[0])
            if kind in ggml_pack.LAYOUTS and (kind not in found or k > found[kind][0]):
                found[kind] = (k, np.asarray(t.data).reshape(int(t.shape[1]), -1)[:ROWS].copy())
    return found


def save(name: str, nodes, inits, inputs: dict, outputs: dict):
    types = {np.float16: (TensorProto.FLOAT16, 'float16'), np.int32: (TensorProto.INT32, 'int32')}
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(k, types[v.dtype.type][0], list(v.shape)) for k, v in inputs.items()],
        [helper.make_tensor_value_info(k, TensorProto.FLOAT16, list(v.shape)) for k, v in outputs.items()],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)], ir_version=10)
    onnx.save(model, str(OUT / f'{name}.onnx'))
    for k, v in inputs.items():
        v.tofile(OUT / f'{name}.{k}.bin')
    for k, v in outputs.items():
        v.astype(np.float32).tofile(OUT / f'{name}.{k}.bin')
    spec = {
        'inputs': [{'name': k, 'type': types[v.dtype.type][1], 'dims': list(v.shape)} for k, v in inputs.items()],
        'outputs': [{'name': k, 'type': 'float16', 'dims': list(v.shape)} for k, v in outputs.items()],
    }
    (OUT / f'{name}.json').write_text(json.dumps(spec))


def grid_init(kind: int) -> list:
    values = ggml_pack.grid(kind)
    if values is None:
        return []
    return [helper.make_tensor('grid', TensorProto.UINT8, [values.nbytes], values.view(np.uint8).tobytes(), raw=True)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', action='store_true', help='only print the case names (comma-separated)')
    args = parser.parse_args()
    names = []
    rng = np.random.default_rng(0)
    if not args.names:
        OUT.mkdir(parents=True, exist_ok=True)
    for kind, (k, data) in sorted(tensors().items()):
        type_name = gguf.GGMLQuantizationType(kind).name.lower()
        n = len(data)
        packed = ggml_pack.pack(kind, data, n, k)
        ggml_pack.check(kind, data, n, k, packed)
        w = ggml_pack.dequantize(kind, packed, k).astype(np.float64)
        grid = grid_init(kind)
        b = helper.make_tensor('b', TensorProto.UINT8, [n, packed.shape[1] * 4], packed.view(np.uint8).tobytes(), raw=True)
        for m in ROWS_OF_A:
            name = f'ggml_{type_name}_m{m}'
            names.append(name)
            if args.names:
                continue
            a = (rng.standard_normal((1, m, k)) * 0.5).astype(np.float16)
            y = (a.astype(np.float64) @ w.T).astype(np.float16)
            node = helper.make_node('MatMulGgml', ['a', 'b'] + (['grid'] if grid else []), ['y'], domain='com.microsoft', K=k, N=n, ggml_type=kind)
            save(name, [node], [b] + grid, {'a': a}, {'y': y})
        name = f'ggml_{type_name}_gather'
        names.append(name)
        if not args.names:
            ids = rng.integers(0, n, (2, 5)).astype(np.int32)
            y = w[ids].astype(np.float16)
            node = helper.make_node('GatherGgml', ['b', 'ids'] + (['grid'] if grid else []), ['y'], domain='com.microsoft', K=k, ggml_type=kind)
            save(name, [node], [b] + grid, {'ids': ids}, {'y': y})
    print(','.join(names))


if __name__ == '__main__':
    sys.exit(main())
